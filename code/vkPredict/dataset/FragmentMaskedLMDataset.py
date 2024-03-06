import torch.utils.data
import typing
import json
import copy
import random
import numpy as np

class FragmentMaskedLMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizedSnapshotFilePath: str,
        maxSeqLen: int,
        epochSize: int,
        mlm_prob: float,
        maskTokenId: int,
        padTokenId: int,
        vocabSize: int
    ):
        self.data = None
        with open(tokenizedSnapshotFilePath, "r") as fp:
            self.data = json.load(fp)
        
        self.epochSize = epochSize
        self.dataSize = len(self.data)
        assert(self.dataSize >= 1)

        self.maxSeqLen = maxSeqLen
        self.mlm_prob = mlm_prob
        assert(0 < self.mlm_prob < 1)

        self.maskTokenId = maskTokenId
        self.padTokenId = padTokenId
        self.vocabSize = vocabSize

    def __len__(self):
        return self.epochSize

    def __getitem__(self, idx: int):
        """Doesn't support slicing at present"""
        nextIdx = random.randint(0, self.dataSize - 1)

        elemTokenInput = self.data[nextIdx]["input_ids"]

        # do random splitting
        # TODO: see if this is the best practice
        if len(elemTokenInput) > self.maxSeqLen:
            startOffset = random.randint(0, len(elemTokenInput)-self.maxSeqLen)
            elemTokens = elemTokenInput[startOffset:startOffset+self.maxSeqLen]
        elif len(elemTokenInput) == self.maxSeqLen:
            elemTokens = elemTokenInput
        else:
            elemTokens = elemTokenInput + [self.padTokenId for i in range(self.maxSeqLen - len(elemTokenInput))]
        
        # print(elemTokens)
        inputs = np.asarray(elemTokens, dtype=np.int64).copy()
        labels = np.asarray(elemTokens, dtype=np.int64).copy()

        # TODO: mask out the prob matrix if necessary
        probability_matrix = np.full(labels.shape, self.mlm_prob)

        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html
        # Returns shape (numTokens,)
        masked_indices = np.random.binomial(
            1, probability_matrix, size=probability_matrix.shape
        ).astype(bool)

        # -100 is used in CrossEntropyLoss as ignore_index in PyTorch code
        # Ignore loss for non masked tokens
        labels[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # 0.8 * mlm_prob
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
        inputs[indices_replaced] = self.maskTokenId

        # 10% of the time, we replace masked input tokens with random word
        # 0.5 * mlmprob * (1-0.8) = 0.1 * mlmprob
        indices_random = (
            np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced
        )
        random_words = np.random.randint(
            low=0, high=self.vocabSize, size=np.count_nonzero(indices_random), dtype=np.int64
        )
        inputs[indices_random] = random_words

        # and for 10% of the time, the words are unchanged
        return {
            "input_ids": inputs,
            "labels": labels
        }

