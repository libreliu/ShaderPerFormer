import unittest
import json
import math
import logging
import numpy as np
import typing
from misc.HfSpvTokenizer import HfSpvTokenizer
from test.exampleImageOnlyShader import fragmentSpv

logger = logging.getLogger(__name__)

class HfSpvTokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = HfSpvTokenizer()
    
    def test_tokenize(self):
        tokenized = self.tokenizer(fragmentSpv)
        print(tokenized)
