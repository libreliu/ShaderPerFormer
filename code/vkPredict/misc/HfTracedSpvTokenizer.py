import vkExecute.spv
import os
import torch
from typing import Any, List, Optional, Union, Dict
import tokenizers, tokenizers.decoders
from transformers import PreTrainedTokenizer
from transformers.utils import logging, PaddingStrategy
from transformers.tokenization_utils_base import (
    EncodedInput,
    BatchEncoding,
    TruncationStrategy,
    TensorType
)
from transformers.utils.generic import is_tf_tensor, is_torch_tensor, to_py_obj
import numpy as np
import itertools
from typing import List, Tuple
from collections.abc import Mapping, Sized

TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]

class HfTracedSpvTokenizer:
    """Mimic the behaviour of Huggingface tokenizer, but with vkExecute's tokenizer"""
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(
            self,
            padding_side="left",
            single_entrypoint=False,
            compact_types=False,
            convert_ext_insts=False,
            relative_inst_id_pos=False
        ):
        self.padding_side = padding_side
        self.name = "HfSpvTokenizer"
        # inline Tokenizer(
        #   bool compactTypes,
        #   bool entrypointOnly,
        #   bool convertExtInsts,
        #   bool relativeInstIdPos
        # ) 
        self.tokenizer = vkExecute.spv.Tokenizer(
            compact_types,
            single_entrypoint,
            convert_ext_insts,
            relative_inst_id_pos
        )
        self.spvProc = vkExecute.SpvProcessor()

        # check SpecialSymbols inside Tokenize.hpp 
        self.specialTokens = {
            "[PAD]": 1000+0,
            "[BOS]": 1000+1,
            "[EOS]": 1000+2,
            "[MASK]": 1000+3,
            "[SEP]": 1000+4,
            "[UNK]": 1000+5,
            "[CLS]": 1000+6,
        }

        self.specialTokenIds = {
            1000+0: "[PAD]",
            1000+1: "[BOS]",
            1000+2: "[EOS]",
            1000+3: "[MASK]",
            1000+4: "[SEP]",
            1000+5: "[UNK]",
            1000+6: "[CLS]",
        }


    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens_with_trace([], token_ids_0, token_ids_1 if pair else None)[1]
        )

    def get_command(self, token):
        res = self.specialTokens[token]
        assert(res is not None)
        return res

    @property
    def pad_token(self) -> str:
        return "[PAD]"

    @property
    def pad_token_id(self):
        return self.get_command("[PAD]")

    @property
    def eos_token(self) -> str:
        return "[EOS]"

    @property
    def eos_token_id(self):
        return self.get_command("[EOS]")
    
    @property
    def bos_token(self) -> str:
        return "[BOS]"

    @property
    def bos_token_id(self):
        return self.get_command("[BOS]")

    @property
    def mask_token(self) -> str:
        return "[MASK]"

    @property
    def mask_token_id(self):
        return self.get_command("[MASK]")

    @property
    def vocab_size(self):
        """todo: grab via reflection"""
        return 40000

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, spvText, **kwargs) -> List[str]:
        raise NotImplementedError("This is now effectively unused, should not be called")

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.specialTokens:
            return self.specialTokens[token]
        else:
            return int(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.specialTokenIds:
            return self.specialTokenIds[index]
        else:
            return str(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        tokenIds = [self._convert_token_to_id(token) for token in tokens]
        return self.tokenizer.deTokenize(tokenIds)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.
        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.
        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        raise NotImplementedError("Not expected to be called")

        return None

    def get_prefix_tokens(self):
        prefix_tokens = [self.get_command("[BOS]")]
        return prefix_tokens

    def build_prompt(self, query, history=None):
        assert(False)

    def build_inputs_with_special_tokens_with_trace(
            self, trace_labels: List[int], token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        prefix_tokens = self.get_prefix_tokens()
        token_ids_0 = prefix_tokens + token_ids_0
        trace_labels = [0] * len(prefix_tokens) + trace_labels
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("[EOS]")]
            trace_labels = trace_labels + [0]
        return trace_labels, token_ids_0

    def _pad(
            self,
            encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)
        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.
                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:
                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        assert self.padding_side == "left"

        required_input = encoded_inputs[self.model_input_names[0]]
        seq_length = len(required_input)

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * seq_length

        if "position_ids" not in encoded_inputs:
            encoded_inputs["position_ids"] = list(range(seq_length))

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]

            if "trace_labels" in encoded_inputs:
                encoded_inputs["trace_labels"] = [0] * difference + encoded_inputs["trace_labels"]

            # (should not matter if I use <PAD> or sth)
            if "mlm_labels" in encoded_inputs:
                encoded_inputs["mlm_labels"] = [self.pad_token_id] * difference + encoded_inputs["mlm_labels"]
            
            if "label_dest_heads" in encoded_inputs:
                # Do nothing - label_dest_heads is an scalar int
                pass

            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

        # print(f"padded: {len(encoded_inputs)}")
        return encoded_inputs
    
    def _encode_plus(
        self,
        spvBinaryRepr: bytes,
        id2TraceIdxMap: Dict[int, int],
        traceCounters: List[int],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        def get_input_ids_and_trace_labels(
                spvBinaryRepr: bytes,
                id2TraceIdxMap: Dict[int, int],
                traceCounters: List[int]    
            ):
            """text: spvBlob of type vector<char> on C++ side
            and it's bytes object on python side
            """
            
            if not isinstance(spvBinaryRepr, bytes):
                raise ValueError(f"Input {spvBinaryRepr} is not valid. Should be bytes for inlined spv module")

            spvListRepr = [i for i in map(lambda x: chr(x), spvBinaryRepr)]

            self.tokenizer.loadSpv(spvListRepr)
            tokenVector, tokenTraceVector, errMsg = self.tokenizer.tokenizeWithTrace(
                id2TraceIdxMap,
                traceCounters
            )
            assert(errMsg == "")

            return (tokenTraceVector, tokenVector)

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        assert(not is_split_into_words)
        trace_labels, first_ids = get_input_ids_and_trace_labels(spvBinaryRepr, id2TraceIdxMap, traceCounters)
        second_ids = None

        return self.prepare_for_model_with_trace(
            trace_labels=trace_labels,
            ids=first_ids,
            pair_ids=None,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

    # copy code from PreTrainedTokenizer.prepare_for_model to here
    # to give encoded_input['trace_labels']
    def prepare_for_model_with_trace(
        self,
        trace_labels: List[int],
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs,
    ) -> BatchEncoding:

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation is eliminated
        assert(truncation == TruncationStrategy.DO_NOT_TRUNCATE)
        assert(not return_overflowing_tokens)

        # Add special tokens
        if add_special_tokens:
            trace_sequence, sequence = self.build_inputs_with_special_tokens_with_trace(trace_labels, ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            # NOTE: not supported pairing for now
            assert(not pair)
            sequence = ids + pair_ids if pair else ids
            trace_sequence = trace_labels
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

        assert(len(trace_sequence) == len(sequence))

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        encoded_inputs["trace_labels"] = trace_sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        assert((not hasattr(self, "model_max_length")) or len(encoded_inputs["input_ids"]) <= self.model_max_length)

        # Padding
        if padding != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs
    
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def __call__(
        self,
        spvBinaryRepr: bytes,
        id2TraceIdxMap: Dict[int, int],
        traceCounters: List[int],
        add_special_tokens: bool = True,
        padding: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        # To avoid duplicating
        all_kwargs = {
            "add_special_tokens": add_special_tokens,
            "padding": padding,
            "truncation": truncation,
            "max_length": max_length,
            "stride": stride,
            "is_split_into_words": is_split_into_words,
            "pad_to_multiple_of": pad_to_multiple_of,
            "return_tensors": return_tensors,
            "return_token_type_ids": return_token_type_ids,
            "return_attention_mask": return_attention_mask,
            "return_overflowing_tokens": return_overflowing_tokens,
            "return_special_tokens_mask": return_special_tokens_mask,
            "return_offsets_mapping": return_offsets_mapping,
            "return_length": return_length,
            "verbose": verbose,
        }
        all_kwargs.update(kwargs)

        assert(padding in [PaddingStrategy.DO_NOT_PAD, PaddingStrategy.LONGEST, PaddingStrategy.MAX_LENGTH])
        assert(truncation in [
            TruncationStrategy.ONLY_FIRST,
            TruncationStrategy.ONLY_SECOND,
            TruncationStrategy.LONGEST_FIRST,
            TruncationStrategy.DO_NOT_TRUNCATE
        ])

        return self._encode_plus(
            spvBinaryRepr, id2TraceIdxMap, traceCounters, **all_kwargs
        )

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if required_input is None or (isinstance(required_input, Sized) and len(required_input) == 0):
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_tf_tensor(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_tensor(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    "Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in encoded_inputs.items()}
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)