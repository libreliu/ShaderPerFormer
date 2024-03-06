import os, sys
sys.path.append(os.path.join(os.path.abspath(''), '../'))
import numpy as np

from misc.Directory import (
  getIntermediateDir,
  getVkPredictRootDir
)
from dataset.FragmentPerformanceTracedSnapshotDataset import FragmentPerformanceTracedSnapshotDataset
from dataset.DatasetBuilder import build_dataset
from dataset.MapDataset import MapDataset
from model.ModelBuilder import build_model
from misc.TokenizerBuilder import build_tokenizer
from misc.normalization import (
    LogNormalizer,
    DummyNormalizer
)
from train import (
    TracedDatasetPostProcessor,
    DataCollatorWithPaddingAndTraceEmbedding
)
from compete.TracedLinearRegression import TracedLinearRegression
from compete.TracedPerInstLinearRegression import TracedPerInstLinearRegression
from misc.metric import compute_metrics
from typing import Union
from collections.abc import Mapping
import torch
import tqdm
from toyDb.utils.spv.SpvContext import getDefaultGrammar
from compete.TracedPerInstLinearRegression import TracedPerInstLinearRegression
import argparse


def buildModelAndTokenizer(targetModel, modelLoadDir, tokenizerType):
    modelLoadDir = os.path.join(
        getVkPredictRootDir(), modelLoadDir
    )

    tokenizer = build_tokenizer(tokenizerType)
    model = build_model(
        targetModel, "mse", 4096,
        tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id,
        modelLoadDir
    )

    return model, tokenizer


def atleast_1d(tensor_or_array: Union[torch.Tensor, np.ndarray]):
    if isinstance(tensor_or_array, torch.Tensor):
        if hasattr(torch, "atleast_1d"):
            tensor_or_array = torch.atleast_1d(tensor_or_array)
        elif tensor_or_array.ndim < 1:
            tensor_or_array = tensor_or_array[None]
    else:
        tensor_or_array = np.atleast_1d(tensor_or_array)
    return tensor_or_array

def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    tensor1 = atleast_1d(tensor1)
    tensor2 = atleast_1d(tensor2)

    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    # Let's figure out the new shape
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]

    # Now let's fill the result tensor
    result = tensor1.new_full(new_shape, padding_index)
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0] :, : tensor2.shape[1]] = tensor2
    return result

def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    array1 = atleast_1d(array1)
    array2 = atleast_1d(array2)

    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), axis=0)

    # Let's figure out the new shape
    new_shape = (array1.shape[0] + array2.shape[0], max(array1.shape[1], array2.shape[1])) + array1.shape[2:]

    # Now let's fill the result tensor
    result = np.full_like(array1, padding_index, shape=new_shape)
    result[: array1.shape[0], : array1.shape[1]] = array1
    result[array1.shape[0] :, : array2.shape[1]] = array2
    return result

def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples/dict of tensors.
    """
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, Mapping):
        return type(tensors)(
            {k: nested_concat(t, new_tensors[k], padding_index=padding_index) for k, t in tensors.items()}
        )
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")


def buildDataLoader(datasetPath, model, tokenizer, batch_size=4):
    valDataset = build_dataset(datasetPath, "val")
    valDatasetMapped = MapDataset(valDataset, TracedDatasetPostProcessor, [tokenizer, LogNormalizer(), DummyNormalizer()])

    dataCollator = DataCollatorWithPaddingAndTraceEmbedding(
        tokenizer=tokenizer,
        trace_embedding_method="onehot-base2",
        trace_embedding_dim=model.config.hidden_size,
        padding='longest',
        pad_to_multiple_of=8
    )
    
    dataloader = torch.utils.data.DataLoader(
        valDatasetMapped,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataCollator
    )

    return dataloader

def SHValidate(datasetName, traced: 'bool', target: 'str'):
    """Uses trace"""
    trainDataset = build_dataset(datasetName, 'train')
    valDataset = build_dataset(datasetName, 'val')

    regressor = TracedLinearRegression(1, True, traced, False)
    regressor.train(trainDataset)

    valReal, valPred = regressor.evaluate(valDataset)

    np.save(f"./validation/{target}_base_labels.npy", valReal)
    np.save(f"./validation/{target}_base_preds.npy", valPred)


def PILRValidate(datasetName, traced: 'bool', target: 'str'):
    """Uses trace"""
    trainDataset = build_dataset(datasetName, 'train')
    valDataset = build_dataset(datasetName, 'val')

    grammar = getDefaultGrammar()

    regressor = TracedPerInstLinearRegression(grammar, True, traced)
    regressor.train(trainDataset)

    valReal, valPred = regressor.evaluate(valDataset)

    np.save(f"./validation/{target}_pilr_labels.npy", valReal)
    np.save(f"./validation/{target}_pilr_preds.npy", valPred)


def validate(model, valLoader, target):
    model.eval()

    INFERENCE_DEVICE = 'cuda'
    model.to(INFERENCE_DEVICE)

    labels = np.asarray([], dtype=np.float32)
    preds = np.asarray([], dtype=np.float32)

    with torch.no_grad():
        for model_inputs in tqdm.tqdm(valLoader):
            for k in model_inputs.keys():
                model_inputs[k] = model_inputs[k].to(INFERENCE_DEVICE)

            outputs = model(**model_inputs)
            labels = nested_concat(labels, model_inputs['labels'].cpu().numpy())
            preds = nested_concat(preds, outputs['logits'].squeeze(1).cpu().numpy())

    np.save(f"./validation/{target}_preds.npy", preds)
    np.save(f"./validation/{target}_labels.npy", labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="perfformer-layer9-regression-trace-input-embedding-xformers-memeff")
    parser.add_argument('--tokenizer-type', type=str, default="HfTracedSpvTokenizer-multiple-entrypoint")
    parser.add_argument('--model-load-dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--traced', type=bool, default=True)
    args = parser.parse_args()

    if not os.path.exists("./validation"):
        os.makedirs("./validation")

    model, tokenizer = buildModelAndTokenizer(args.model, args.model_load_dir, args.tokenizer_type)
    valLoader = buildDataLoader(args.dataset, model, tokenizer, args.batch_size)

    validate(model, valLoader, args.target)
    SHValidate(args.dataset, args.traced, args.target)
    PILRValidate(args.dataset, args.traced, args.target)
