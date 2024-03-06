import vkExecute.spv
import os
import torch
from typing import List, Optional, Union, Dict
import tokenizers, tokenizers.decoders
from transformers import PreTrainedTokenizer
from transformers.utils import logging, PaddingStrategy
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding

class HfSpvTokenizer(PreTrainedTokenizer):
    """Mimic the behaviour of Huggingface tokenizer, but with vkExecute's tokenizer"""
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(self, padding_side="left", single_entrypoint=False, **kwargs):
        super().__init__(padding_side=padding_side, **kwargs)
        self.name = "HfSpvTokenizer"
        # inline Tokenizer(bool compactTypes, bool entrypointOnly, bool convertExtInsts) 
        self.tokenizer = vkExecute.spv.Tokenizer(True, single_entrypoint, True)
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
        success, errMsg = self.spvProc.assemble(spvText)
        assert(success)

        self.tokenizer.loadSpv(self.spvProc.exportSpv())
        tokenizedSpv, errMsgs = self.tokenizer.tokenize()
        assert(len(tokenizedSpv) > 0)

        return [str(i) for i in tokenizedSpv]

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

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
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
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("[EOS]")]
        return token_ids_0

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
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

        # print(f"padded: {len(encoded_inputs)}")
        return encoded_inputs