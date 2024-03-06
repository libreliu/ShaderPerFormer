import torch
import numpy as np
import argparse
from functools import reduce
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from PerformanceNet import PerformanceNet
from encoder_trainer import Trainer
from example_tasks import SumDataset

def test_dataset():
    train_dataset = SumDataset('train', 'seq2float', dataset_size=10000, seed=1)
    test_dataset = SumDataset('test', 'seq2float', dataset_size=10000, seed=None)
    x, y = train_dataset.getItem(1)
    print(f"test x: {x}")
    print(f"test y: {y}")

    x, y = test_dataset.getItem(1)
    print(f"test x: {x}")
    print(f"test y: {y}")

def eval_sumtask(model: torch.nn.Module, args):
    model.eval()

    train_dataset = SumDataset(
        'train',
        mode='seq2float',
        max_sequence_length=1,
        seed=args.train_seed
    )
    test_dataset = SumDataset(
        'test',
        mode='seq2float',
        max_sequence_length=1
    )

    def eval_split(split):
        dataset = {'train':train_dataset, 'test':test_dataset}[split]
        
        correct = 0
        total = 0
        errors_reported = 0
        for i in tqdm(range(0, len(dataset))):
            x, y = dataset.getItem(i, verbose=False)
            # print(x)
            # print(y)
            
            in_tensor = x
            in_batch = in_tensor.reshape((1, in_tensor.shape[0]))
            in_str = train_dataset.dictionary.string(in_tensor)

            result_seq = model(in_batch)
            result_seq = result_seq.to('cpu')

            if result_seq - y < 0.5:
                correct += 1
            else:
                if errors_reported < 5:
                    print(f"Wrong! Got result: {result_seq.tolist()}, full: {y.tolist()}")
                    errors_reported += 1
                    
            
            total += 1

        print(f"{split}: {correct} / {total} ({100 * correct / total}%)")

    # run a lot of examples from both train and test through the model and verify the output correctness
    with torch.no_grad():
        model.cpu()
        train_score = eval_split('train')
        test_score  = eval_split('test')


# TODO: add pad from short to long, and implement speed up under mingpt
# TODO: num_workers >= 2; serialize pad info correctly
def padded_collate(batch: list):
    # print(batch)
    # Let's assume that each element in "batch" is a tuple (data, label).
    # Sort the batch in the descending order
    sorted_batch = sorted(
        batch,
        key=lambda x: x['item'][0].shape[0],
        reverse=True
    )
    # Get each sequence and pad it
    sequences = [x['item'][0] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=batch[0]['pad_idx']
    )

    targets = torch.stack([x['item'][1] for x in sorted_batch])

    # print(sequences_padded)
    # print(targets)
    return sequences_padded, targets

def padding_mask(batch: torch.Tensor, pad_idx: int):
    mask = (batch == pad_idx).bool()
    return mask

def do_training(
    model: PerformanceNet,
    train_dataset: SumDataset,
    batch_size=256,
    num_workers=2,
    begin_iter=0,
    max_iters=5000,
    grad_norm_clip=1e4,
    weight_decay=0.1,
    learning_rate=3e-4,
    betas=(0.9, 0.95)
):

    writer = SummaryWriter()
    writer.add_graph(
        model,
        # give the extra 1 dim on batch_size
        tuple(map(lambda x: x.reshape((1, x.shape[0])), train_dataset.getItem(1))),
        False
    )
    writer.add_hparams({
        'lr': learning_rate,
        'bsize': batch_size,
        'dataset_size': train_dataset.dataset_size,
        'max_number': train_dataset.max_number,
        'max_sequence_length': train_dataset.max_sequence_length
    }, {})

    model.to('cuda')

    # setup the optimizer
    optimizer = model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        betas=betas
    )

    # setup the dataloader
    train_loader = DataLoader(
        train_dataset,
        sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=padded_collate
    )
    model.train()
    data_iter = iter(train_loader)

    pbar = tqdm(range(begin_iter, max_iters))
    for iter_num in pbar:
        # fetch the next batch (x, y) and re-init iterator if needed
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        # print(batch)
        batch = [t.to('cuda') for t in batch]
        x, y = batch

        # generate padding mask
        batch_padding_mask = padding_mask(x, train_dataset.dictionary.pad())
        # print(batch_padding_mask)
        # print(x)
        # print(y)

        # recovered = train_dataset.dictionary.string(x[0])
        # print(recovered)

        # forward the model
        logits, loss = model(x, y, src_key_padding_mask=batch_padding_mask)
        
        if iter_num % 10 == 0:
            pbar.set_postfix_str(f"Train Loss: {loss.item():.5f}")
            writer.add_scalar('Loss/train', loss.item(), iter_num)

        # backprop and update the parameters
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        optimizer.step()

    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-checkpoint", type=str)
    parser.add_argument("--save-checkpoint", type=str)
    parser.add_argument("--test-dataset", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--eval-sets", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--begin-iter", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)


    args = parser.parse_args()

    train_dataset = SumDataset('train', seed=args.train_seed, max_sequence_length=1)

    # mimic the params of gpt-mini (dict(n_layer=6, n_head=6, n_embd=192))
    model = PerformanceNet(
        ntoken=len(train_dataset.dictionary),
        d_model=192,
        d_mlp=128,
        nhead=6,
        d_hid=4 * 192,
        nlayers=6,
        dropout=0.5,
        output_dim=1
    )

    if args.export_onnx == True:
        model.eval()
        example_input = tuple(map(lambda x: x.reshape((1, x.shape[0])), train_dataset.getItem(1)))
        print(example_input)
        print(torch.jit.script(model, example_inputs=[example_input]).graph)
        
        torch.onnx.export(model, example_input, 'rnn.onnx', input_names=[], output_names=[])

    if args.test_dataset == True:
        test_dataset()
        return

    # load from checkpoint
    if args.load_checkpoint is not None:
        ckpt = torch.load(args.load_checkpoint)
        model.load_state_dict(ckpt['model_state_dict'])

    if args.train == True:
        model.train()
        do_training(
            model,
            train_dataset,
            batch_size=args.batch_size,
            num_workers=2,
            begin_iter=args.begin_iter,
            max_iters=args.max_iter,
            grad_norm_clip=10000,
            weight_decay=0.1,
            learning_rate=args.learning_rate,
            betas=(0.9, 0.95)
        )

        # move model back
        model.to('cpu')

    if args.save_checkpoint:
        torch.save({
            'model_state_dict': model.state_dict()
        }, args.save_checkpoint)

    if args.interactive:
        model.eval()
        assert(not args.eval_sets)
        with torch.no_grad():

            while True:
                in_str = None
                try:
                    in_str = input('Please enter question: ')
                except KeyboardInterrupt:
                    print("Bye.")
                    break
                except EOFError:
                    print("Bye.")
                    break

                in_tensor = train_dataset.dictionary.encode_line(in_str)

                in_batch = torch.reshape(in_tensor, (1, in_tensor.size(0)))
                print(in_batch)
                result_seq = model(in_batch)
                print(result_seq)
                
                print(f"Model output: {result_seq}")
    
    if args.eval_sets:
        eval_sumtask(model, args)




if __name__ == '__main__':
    main()