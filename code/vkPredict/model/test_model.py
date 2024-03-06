import torch
import numpy as np
import argparse
from functools import reduce
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import random
import tqdm

from mingpt.model import GPT
from mingpt.trainer import Trainer

class SumDataset(Dataset):
    """ 
    Dataset for the Sum problem. E.g. for problem length 6:
    Input: 1 2 3 4 5 6 <eos> -> Output: 21 <eos>
    Which will feed into the transformer concatenated as:
    input:  1 2 3 4 5 6 <eos>   21
    output: I I I I I I  21   <eos>
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self, split, max_length=10, max_number=200, seed_offset=0):
        assert split in {'train', 'test'}
        self.split = split
        self.max_length = max_length
        self.max_number = max_number
        self.seed_offset = seed_offset
    
    def __len__(self):
        return 10000 # ...
    
    @staticmethod
    def get_vocab_size():
        """Byte level encoding for the input; 256 is for <eos>, 257 is for the pad"""
        return 258
    
    @staticmethod
    def get_eos():
        return 256
    
    @staticmethod
    def get_pad():
        return 257

    @staticmethod
    def get_num_of_digits(number):
        return str(number).__len__()

    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output.

        input_digits = self.get_num_of_digits(self.max_number) * self.max_length
        input_spaces = self.max_length - 1
        output_digits = self.get_num_of_digits(self.max_number * self.max_length)
        output_eos = 1
        output_spaces = 1

        # additional 1 is between <eos> and the last input
        total = input_digits + input_spaces + output_digits + output_eos + output_spaces + 1
        return total + 10

    def __getitem__(self, idx):
        return self.getItem(idx, verbose=False)

    def getItem(self, idx, verbose=True):
        """For debugging turn the verbose on"""
        
        # random.seed(idx + self.seed_offset)
        

        # generate some random integers
        in_length = random.randint(1, self.max_length)

        in_list = [random.randint(0, self.max_number) for _ in range(0, in_length)]
        in_bytes = " ".join(map(lambda x: str(x), in_list)).encode("utf-8")

        in_tensor = torch.from_numpy(np.frombuffer(in_bytes, dtype=np.uint8).copy()).long()
        
        # solve the task: i.e. sort
        sol_str = str(reduce(lambda x, y: x + y, in_list))
        sol_bytes = sol_str.encode("utf-8")

        sol = torch.from_numpy(np.frombuffer(sol_bytes, dtype=np.uint8).copy()).long()

        if verbose:
            print(f"in_list: {in_list}")
            print(f"in_bytes: {in_bytes}")
            print(f"in_tensor: {in_tensor}")
            print(f"sol_bytes: {sol_bytes}")
            print(f"sol: {sol}")

        # concatenate the problem specification and the solution
        cat = torch.cat((in_tensor, torch.LongTensor([self.get_eos()]), sol, torch.LongTensor([self.get_eos()])), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y_select = torch.arange(0, in_tensor.size()[0], 1)
        y[y_select] = -1

        return x, y

# train_dataset = SumDataset('train', seed_offset=10000)
# test_dataset = SumDataset('test', seed_offset=20000)
# x, y = train_dataset.getItem(1)
# for a, b in zip(x, y):
#     print(a, b)


# now let's perform some evaluation


def eval_sumtask(model: torch.nn.Module, train_offset=10000, test_offset=20000):
    model.eval()

    train_dataset = SumDataset('train', seed_offset=10000,max_length=1)
    test_dataset = SumDataset('test', seed_offset=20000,max_length=1)

    def eval_split(split, max_batches):
        dataset = {'train':train_dataset, 'test':test_dataset}[split]
        
        correct = 0
        total = 0
        errors_reported = 0
        for i in tqdm.tqdm(range(0, len(dataset))):
            x, y = dataset[i]

            full_seq = torch.cat([x, y[-1:]])
            eos_index = full_seq.tolist().index(dataset.get_eos())
            
            in_byte_list = full_seq[:eos_index + 1]
            in_tensor = torch.LongTensor(in_byte_list)
            in_batch = torch.reshape(in_tensor, (1, in_tensor.size()[0])).to('cpu')

            result_seq = model.generate_lm(in_batch, 10, end_token=dataset.get_eos())
            result_seq = result_seq.to('cpu')

            if result_seq.size(1) == full_seq.size(0) and torch.all(torch.flatten(result_seq) == full_seq):
                correct += 1
            else:
                if errors_reported < 5:
                    print(f"Wrong! Got result: {result_seq.tolist()}, full: {full_seq.tolist()}")
                    errors_reported += 1
            
            total += 1

        print(f"{split}: {correct} / {total} ({100 * correct / total}%)")

    # run a lot of examples from both train and test through the model and verify the output correctness
    with torch.no_grad():
        train_score = eval_split('train', max_batches=50)
        test_score  = eval_split('test',  max_batches=50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-checkpoint", type=str)
    parser.add_argument("--save-checkpoint", type=str)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--eval-sets", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train-offset", type=int, default=50000)

    args = parser.parse_args()

    train_dataset = SumDataset('train', seed_offset=args.train_offset, max_number=200, max_length=1)

    # create a GPT instance
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-mini'
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    model = GPT(model_config)

    # load from checkpoint
    if args.load_checkpoint is not None:
        ckpt = torch.load(args.load_checkpoint)
        model.load_state_dict(ckpt['model_state_dict'])

    if args.train == True:
        # create a Trainer object
        train_config = Trainer.get_default_config()
        train_config.learning_rate = 1e-4 # the model we're using is so small that we can go a bit faster
        train_config.max_iters = 5000
        train_config.num_workers = 8
        train_config.batch_size = 256
        trainer = Trainer(train_config, model, train_dataset)

        def batch_end_callback(trainer):
            if trainer.iter_num % 100 == 0:
                print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        trainer.set_callback('on_batch_end', batch_end_callback)

        trainer.run()

        # move model back
        model.to('cpu')

    if args.save_checkpoint:
        torch.save({
            'model_state_dict': model.state_dict()
        }, args.save_checkpoint)

    if args.interactive:
        model.eval()
        assert(not args.eval_sets)

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

            in_bytes = in_str.encode('utf-8')
            in_tensor = torch.from_numpy(np.frombuffer(in_bytes, dtype=np.uint8).copy()).long()
            in_tensor = torch.cat([in_tensor, torch.LongTensor([SumDataset.get_eos()])])

            in_batch = torch.reshape(in_tensor, (1, in_tensor.size(0)))
            print(in_batch)
            result_seq = model.generate_lm(in_batch, 10, end_token=SumDataset.get_eos())
            print(result_seq)

            model_output = result_seq.flatten()[in_tensor.size(0):].tolist()
            print(model_output)
            model_str_output = ""
            for ch in model_output:
                if ch == SumDataset.get_eos():
                    model_str_output += " <eos>"
                else:
                    model_str_output += chr(ch)
            
            print(f"Model output: {model_str_output}")
    
    if args.eval_sets:
        eval_sumtask(model, args.train_offset)




if __name__ == '__main__':
    main()