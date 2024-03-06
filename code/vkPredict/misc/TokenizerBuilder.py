from .HfBpeTokenizer import HfBpeTokenizer
from .HfSpvTokenizer import HfSpvTokenizer
from .HfTracedSpvTokenizer import HfTracedSpvTokenizer
from transformers import PreTrainedTokenizer
from typing import Union

import os
rootDir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")

def build_tokenizer(tokenizerName: str) -> Union[PreTrainedTokenizer, HfTracedSpvTokenizer]:
    tokenizer = None
    
    if tokenizerName == 'HfBpeTokenizer':
        tokenizer = HfBpeTokenizer(os.path.join(rootDir, "./SpvBpeTokenizer.json"))
    elif tokenizerName == 'HfSpvTokenizer':
        tokenizer = HfSpvTokenizer()
    elif tokenizerName == 'HfTracedSpvTokenizer-single-entrypoint':
        # Only useful when inlined to only function to be called from entrypoint
        # That is - no other "useful" function exists, only one big giant function
        tokenizer = HfTracedSpvTokenizer(single_entrypoint=True)
    elif tokenizerName == 'HfTracedSpvTokenizer-multiple-entrypoint':
        tokenizer = HfTracedSpvTokenizer(single_entrypoint=False)
    elif tokenizerName == 'HfTracedSpvTokenizer-multiple-entrypoint-relative-id':
        tokenizer = HfTracedSpvTokenizer(single_entrypoint=False, relative_inst_id_pos=True)

    assert(tokenizer is not None)

    return tokenizer