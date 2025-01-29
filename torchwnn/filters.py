# MIT License

# Copyright (c) 2025 Leandro Santiago de Araújo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import math
import numpy as np
from torchwnn.cpp import functional
from torchwnn.functional import h3_generate

__all__ = [
    "BloomFilter",    
]

class BloomFilter:
    def __init__(self, input_size, data_size, n_hashes):
        self.input_size = input_size
        self.data_size = data_size
        self.n_hashes = n_hashes
        self.hash_matrix = h3_generate(input_size, data_size, n_hashes)

        self.data = torch.zeros((1, self.data_size), dtype=torch.uint8)

    def add(self, input):
        hash_results = functional.h3_hash(input, self.hash_matrix)        
        self.data.scatter_(1, hash_results, 1, reduce = "add")

    def check_member(self, input):
        hash_results = functional.h3_hash(input, self.hash_matrix)
        selected = torch.clamp(torch.gather(self.data, 1, hash_results), 0, 1)        
        return selected.all()

    @classmethod
    def calculate_num_bits(cls, capacity: int, error: float) -> int:
        nbits = math.floor((-capacity * (math.log(error) / math.pow(math.log(2), 2))) + 1)
        return 1 << math.ceil(math.log2(nbits))        
    
    @classmethod
    def calculate_num_hashes(cls, capacity: int, error: float) -> int:
        nbits = math.floor((-capacity * (math.log(error) / math.pow(math.log(2), 2))) + 1)
        return math.floor((nbits * (math.log(2)/capacity)) + 1)
    
    
