import torch
import numpy as np
import argparse
from functools import reduce
from torch.utils.data import Dataset
import random
import tqdm
import time

from dictionary import ByteOnlyDictionary

class SumDataset(Dataset):
    """ 
    Dataset for the Sum problem.
    
    E.g. for problem length 6:
    Input: 1 2 3 4 5 6 <eos> -> Output: 21 <eos>
    Which will feed into the transformer concatenated as:
    input:  1 2 3 4 5 6 <eos>   21
    output: I I I I I I  21   <eos>
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self,
                 split,
                 mode='seq2float',
                 dictionary=ByteOnlyDictionary(),
                 dataset_size=10000,
                 # max length of total additions
                 max_sequence_length=10,
                 # [1, max_number]
                 max_number=200,
                 seed=None
        ):
        assert split in {'train', 'test'}
        assert mode in {'seq2float', 'seq2seq-gpt'}
        self.split = split
        self.mode = mode
        self.dictionary = dictionary
        self.dataset_size = dataset_size

        self.max_sequence_length = max_sequence_length
        self.max_number = max_number

        if seed is None:
            print(f"A None seed is used, reproducibility may suffer")

        self.random_state = np.random.RandomState(seed)
        self.data = []
        for i in range(self.dataset_size):
            this_len = self.random_state.randint(1, self.max_sequence_length+1)
            self.data.append(
                self.random_state.randint(1, self.max_number + 1, size=this_len).tolist()
            )

    def __len__(self):
        return self.dataset_size

    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output.
        return 5000

    def __getitem__(self, idx):
        return {
            "item": self.getItem(idx, verbose=False),
            "pad_idx": self.dictionary.pad()
        }

    def getItem(self, idx, verbose=True):
        """For debugging turn the verbose on"""
        # random.seed(idx + self.seed_offset)

        in_list = self.data[idx]
        in_str = " ".join(map(lambda x: str(x), in_list))
        in_tensor = self.dictionary.encode_line(in_str)
        
        if self.mode == 'seq2float':
            # solve the task: i.e. sort
            sol_float = float(reduce(lambda x, y: x + y, in_list))

            if verbose:
                print(f"in_list: {in_list}")
                print(f"in_tensor: {in_tensor}")
                print(f"sol_float: {sol_float}")

            return in_tensor, torch.FloatTensor([sol_float])
        elif self.mode == 'seq2seq-gpt':
            raise NotImplementedError()    


