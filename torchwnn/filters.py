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
    "CuckooFilter",
    "QuotientFilter", 
    "RSQuotientFilter",
    "CountMinSketch",   
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
    
# Implementation based on the following paper:
# Cuckoo Filter: Practically Better Than Bloom
# Link: <https://dl.acm.org/doi/10.1145/2674005.2674994>     
class CuckooFilter:
    def __init__(self, input_size, n_buckets, bucket_size = 4, fingerprint_size = 3, maxKick = 50):
        self.input_size = input_size
        self.n_buckets = n_buckets
        self.n_hashes = 2
        self.bucket_size = bucket_size
        self.fingerprint_size = fingerprint_size # number of bits for fingerprint
        self.hash_matrix = h3_generate(input_size, n_buckets,  self.n_hashes)
        self.fingerprint_matrix = h3_generate(fingerprint_size, n_buckets,  1)
        self.fingerprint_mask = (1 << fingerprint_size) - 1 
        self.maxKick = maxKick

        self.data = torch.zeros((self.bucket_size, self.n_buckets), dtype=torch.uint8)
        self.indexes = torch.zeros((1, self.n_buckets), dtype=torch.uint32)

    def add(self, input):
        hash_results = functional.h3_hash(input, self.hash_matrix) 
        i1 = hash_results[:, 0].item()
        
        fingerprint = hash_results[:, 1].item() & self.fingerprint_mask 
        

        if self.indexes[:,i1] == self.bucket_size:
            fingerprint_array = torch.tensor(np.unpackbits(np.array([fingerprint], dtype=np.uint8))[-self.fingerprint_size:])                
            fingerprint_array.unsqueeze_(0)        
            hash_results = functional.h3_hash(fingerprint_array, self.fingerprint_matrix) 
            i2 = i1 ^ hash_results[:, 0].item() 

            if self.indexes[:,i2] == self.bucket_size:
                i = np.random.choice((i1, i2))
                
                for _ in range(self.maxKick):
                    bucket_i = np.random.choice(self.bucket_size)
                    fingerprint2 = self.data[bucket_i, i] 
                    self.data[bucket_i, i] = fingerprint  
                    fingerprint = fingerprint2

                    fingerprint_array = torch.tensor(np.unpackbits(np.array([fingerprint], dtype=np.uint8))[-self.fingerprint_size:])                
                    fingerprint_array.unsqueeze_(0)        
                    hash_results = functional.h3_hash(fingerprint_array, self.fingerprint_matrix) 
                    i = i ^ hash_results[:, 0].item() 

                    if self.indexes[:,i].item() < self.bucket_size:
                        bucked_id = self.indexes[:,i].item()
                        self.data[bucked_id, i] = fingerprint
                        self.indexes[:,i] = bucked_id + 1
                        return i                       
            else:   
                bucked_id = self.indexes[:,i2].item()
                self.data[bucked_id, i2] = fingerprint
                self.indexes[:,i2] = bucked_id + 1
                return i2    
        else:
            bucked_id = self.indexes[:,i1].item()
            self.data[bucked_id, i1] = fingerprint
            self.indexes[:,i1] = bucked_id + 1                
            return i1
        

    def check_member(self, input):
        hash_results = functional.h3_hash(input, self.hash_matrix)
        i1 = hash_results[:, 0].item()        
        fingerprint = hash_results[:, 1].item() & self.fingerprint_mask 
        fingerprint_array = torch.tensor(np.unpackbits(np.array([fingerprint], dtype=np.uint8))[-self.fingerprint_size:])                
        fingerprint_array.unsqueeze_(0)        
        hash_results = functional.h3_hash(fingerprint_array, self.fingerprint_matrix) 
        i2 = i1 ^ hash_results[:, 0].item()  
        
        selected = torch.isin(self.data[...,i1], fingerprint)
        selected2 = torch.isin(self.data[...,i2], fingerprint)        
        return selected.any() | selected2.any()
    
    @classmethod
    def calculate_fingerprint_size(cls, bucket_size: int, error: float) -> int:
        return int(math.ceil(math.log(1.0 / error, 2) + math.log(2 * bucket_size, 2)))

# Implementation based on original Quotient Filter proposed in the following paper:
# Don't thrash: how to cache your hash on flash
# Link: <https://dl.acm.org/doi/10.14778/2350229.2350275> 
class QuotientFilter:
    def __init__(self, input_size, remainder_size, data_size = None):
        self.input_size = input_size
        self.remainder_size = remainder_size
        self.quotient_size = self.input_size - self.remainder_size

        if data_size:
            self.data_size = data_size
        else:
            self.data_size = 1 << self.quotient_size

        self.metadata = torch.zeros((1, self.data_size), dtype=torch.uint8)
        self.data = torch.zeros((1, self.data_size), dtype=torch.uint8)
        self.tidx = torch.arange(input_size).flip(dims=(0,))
        self.remainder_mask = (1 << remainder_size) - 1
        self.bit_occupieds = 1
        self.bit_runends = 2
        self.bit_continuations = 2
        self.bit_shifteds = 4
        self.bit_empty = 7

    def rank_select(self, index):
        count_occupieds = 0
        count_runends = 0
        runends_shift = self.bit_runends - 1
        last_run_index = -1

        for i in range(index + 1):
            count_occupieds += self.metadata[:, i] & self.bit_occupieds
            runends_val = (self.metadata[:, i] & self.bit_runends) >> runends_shift
            count_runends += runends_val
            last_run_index = i if runends_val > 0 else last_run_index

        if count_occupieds > count_runends:
            while count_occupieds > count_runends:
                runends_val = (self.metadata[:, i] & self.bit_runends) >> runends_shift
                count_runends += runends_val                
                i += 1
            
            last_run_index = i - 1
        elif count_occupieds < count_runends:
            while count_occupieds < count_runends:
                runends_val = (self.metadata[:, i] & self.bit_runends) >> runends_shift
                count_runends -= runends_val
                i -= 1
            
            runends_val = (self.metadata[:, i] & self.bit_runends) >> runends_shift    
            while runends_val == 0:    
                i -= 1
                runends_val = (self.metadata[:, i] & self.bit_runends) >> runends_shift                
            
            last_run_index = i 

        return last_run_index

    def next_free_slot(self, index):
        free_index = index
        last_index = self.rank_select(index)
        while free_index <= last_index:
            free_index = last_index + 1
            last_index = self.rank_select(free_index)
        return free_index

    def add(self, input):
        figerprint = (input << self.tidx).sum(dim=1).item()
        quotient = figerprint >> self.remainder_size
        remainder = figerprint & self.remainder_mask
        
        i = quotient    
       
        # Walk back to find the beginning of cluster
        while ((self.metadata[:, i] & self.bit_shifteds) > 0):
            i -= 1
        
        # Walk forward to find the start of the run
        j = i
        while (i < quotient):
            j += 1
            while ((j < self.data_size) and ((self.metadata[:, j] & self.bit_continuations) > 0)):
                j += 1

            i += 1
            while ((i < j) and ((self.metadata[:, i] & self.bit_occupieds) == 0)):
                i += 1

            if (j == self.data_size):
                return 0
        
        ## Found start of a run
       
        # First case if slot is empty
        if ((self.metadata[:, j] & self.bit_empty) == 0):
            # Mark canonical slot as occupied
            self.metadata[:, quotient] |= self.bit_occupieds
            
            if (j != quotient):
                self.metadata[:, j] |= self.bit_shifteds
            self.data[:, j] = remainder            
            return 1
        
        # Check if the start of run has to be shifted. If so, then IS_CONTINUATION is set in the shifted remainder.
        old_remainder = self.data[:, j]

        if (remainder < old_remainder.item()):
            # Old remainder sets CONTINUATION
            self.metadata[:, j] |= self.bit_continuations

            if (self.right_shift(j)):
                #print("Shifted: ", self.metadata)
                # Mark canonical slot as occupied
                self.metadata[:, quotient] |= self.bit_occupieds
                # New remainder unsets CONTINUATION
                self.metadata[:, j] &= ~self.bit_continuations
                self.data[:, j] = remainder
                return 1
            else:
                self.metadata[:, j] &= ~self.bit_continuations
                return 0
            
        elif (remainder == old_remainder):
            self.metadata[:, quotient] |= self.bit_occupieds
            return 1

        j += 1

        # Case that look the subsequent slots to insert new remainder in order.
        while ((j < self.data_size) and ((self.metadata[:, j] & self.bit_continuations) > 0)):
            old_remainder = self.data[:, j]

            if (remainder < old_remainder.item()):
                if (self.right_shift(j)):
                    # Mark canonical slot as occupied
                    self.metadata[:, quotient] |= self.bit_occupieds
                    self.data[:, j] = remainder
                    return 1
                else:
                    return 0
                
            elif (remainder == old_remainder):
                self.metadata[:, quotient] |= self.bit_occupieds
                return 1
                
            j += 1


        if (j == self.data_size):
            return 0

        # Case where was walked the whole run and not found appropriate slot.
        # A new slot will be added to the run. If the slot is full, it is shifted first.
        if (((self.metadata[:, j] & self.bit_empty) == 0) or (self.right_shift(j))):
            # Mark canonical slot as occupied
            self.metadata[:, quotient] |= self.bit_occupieds
            self.metadata[:, j] |= self.bit_continuations
            self.metadata[:, j] |= self.bit_shifteds
            self.data[:, j] = remainder
            return 1

        return 0


    def right_shift(self, index) -> int:
        for i in range(index + 1, self.data_size):
            if ((self.metadata[:, i] & self.bit_empty) == 0):
                for j in range(i-1, index-1, -1):
                    occupied = self.metadata[:, i] & self.bit_occupieds
                    self.metadata[:, i] = self.metadata[:, j]
                    self.data[:, i] = self.data[:, j]

                    if (occupied):
                        self.metadata[:, i] |= self.bit_occupieds
                    else:
                        self.metadata[:, i] &= ~self.bit_occupieds

                    self.metadata[:, i] |= self.bit_shifteds

                    i -= 1
                    if (i <= 0):
                        break
                return 1        
        return 0

    def check_member(self, input):
        fingerprint = (input << self.tidx).sum(dim=1).item()
        quotient = fingerprint >> self.remainder_size 

        if self.metadata[:, quotient] & self.bit_occupieds:
            remainder = fingerprint & self.remainder_mask

            i = quotient
            # Walk back to find the beginning of cluster
            while ((self.metadata[:, i] & self.bit_shifteds) > 0):
                i -= 1
            
            # Walk forward to find the start of the run
            j = i
            while (i < quotient):
                j += 1
                while ((j < self.data_size) and ((self.metadata[:, j] & self.bit_continuations) > 0)):
                    j += 1

                i += 1
                while ((i < j) and ((self.metadata[:, i] & self.bit_occupieds) == 0)):
                    i += 1                
                
                if (j == self.data_size):
                    return 0


            if self.data[:, j].item() == remainder: 
                return 1
            j += 1

            while ((self.metadata[:,j] & self.bit_continuations) > 0):
                if self.data[:, j].item() == remainder: 
                    return 1
                j += 1

        return 0
        
# Implementation based on RS-QF (Rank-Select Quotient Filter) proposed in the following paper:
# A General-Purpose Counting Filter: Making Every Bit Count
# Link: <https://dl.acm.org/doi/10.1145/3035918.3035963> 
class RSQuotientFilter:
    def __init__(self, input_size, remainder_size):
        self.input_size = input_size
        self.remainder_size = remainder_size
        self.quotient_size = self.input_size - self.remainder_size
        self.data_size = 1 << self.quotient_size
        self.metadata = torch.zeros((1, self.data_size), dtype=torch.uint8)
        self.data = torch.zeros((1, self.data_size), dtype=torch.uint8)
        self.tidx = torch.arange(input_size).flip(dims=(0,))
        self.remainder_mask = (1 << remainder_size) - 1
        self.bit_occupieds = 1
        self.bit_runends = 2

    def rank_select(self, index):
        count_occupieds = 0
        count_runends = 0
        runends_shift = self.bit_runends - 1
        last_run_index = -1

        for i in range(index + 1):
            count_occupieds += self.metadata[:, i] & self.bit_occupieds
            runends_val = (self.metadata[:, i] & self.bit_runends) >> runends_shift
            count_runends += runends_val
            last_run_index = i if runends_val > 0 else last_run_index

        if count_occupieds > count_runends:
            while count_occupieds > count_runends:
                runends_val = (self.metadata[:, i] & self.bit_runends) >> runends_shift
                count_runends += runends_val                
                i += 1
            
            last_run_index = i - 1
        elif count_occupieds < count_runends:
            while count_occupieds < count_runends:
                runends_val = (self.metadata[:, i] & self.bit_runends) >> runends_shift
                count_runends -= runends_val
                i -= 1
            
            runends_val = (self.metadata[:, i] & self.bit_runends) >> runends_shift    
            while runends_val == 0:    
                i -= 1
                runends_val = (self.metadata[:, i] & self.bit_runends) >> runends_shift                
            
            last_run_index = i 

        return last_run_index

    def next_free_slot(self, index):
        free_index = index
        last_index = self.rank_select(index)
        while free_index <= last_index:
            free_index = last_index + 1
            last_index = self.rank_select(free_index)
        return free_index

    def add(self, input):
        figerprint = (input << self.tidx).sum(dim=1).item()
        quotient = figerprint >> self.remainder_size
        remainder = figerprint & self.remainder_mask

        last_index = self.rank_select(quotient)
        #print(quotient, remainder, last_index)

        if quotient > last_index:
            self.data[:, quotient] = remainder
            self.metadata[:, quotient] |= self.bit_runends 
        else:
            last_index += 1
            free_index = self.next_free_slot(last_index)

            while free_index > last_index:
                self.data[:, free_index] = self.data[:, free_index - 1]
                self.metadata[:, free_index] |= (self.metadata[:, free_index - 1] & self.bit_runends)
                free_index -= 1

            self.data[:, last_index] = remainder
            if self.metadata[:, quotient] & self.bit_occupieds:
                 self.metadata[:, last_index - 1] -= self.bit_runends 
            
            self.metadata[:, last_index] |= self.bit_runends 

        self.metadata[:, quotient] |= self.bit_occupieds

    def check_member(self, input):
        figerprint = (input << self.tidx).sum(dim=1).item()
        quotient = figerprint >> self.remainder_size        

        if self.metadata[:, quotient] & self.bit_occupieds:
            last_index = self.rank_select(quotient)
            remainder = figerprint & self.remainder_mask
            #print(quotient, remainder, last_index)

            if self.data[:, last_index] == remainder:
                return 1
            
            last_index -= 1      
            runend = (self.metadata[:,last_index] & self.bit_runends) == 0
            
            while quotient <= last_index and runend:
                #print(last_index, self.data[:, last_index], runend)
                if self.data[:, last_index] == remainder:
                    return 1
                last_index -= 1      
                runend = (self.metadata[:,last_index] & self.bit_runends) == 0

        return 0

class CountMinSketch:
    def __init__(self, input_size, data_size, n_hashes):
        self.input_size = input_size
        self.data_size = data_size
        self.n_hashes = n_hashes
        self.hash_matrix = h3_generate(input_size, data_size, n_hashes)

        self.data = torch.zeros((n_hashes, self.data_size), dtype=torch.uint8)

    def add(self, input):
        hash_results = functional.h3_hash(input, self.hash_matrix)
        #print(hash_results, hash_results.squeeze(0).unsqueeze(1))        
        self.data.scatter_(1, hash_results.squeeze(0).unsqueeze(1), 1, reduce = "add")

    def check_member(self, input):
        hash_results = functional.h3_hash(input, self.hash_matrix)
        selected = torch.gather(self.data, 1, hash_results.squeeze(0).unsqueeze(1))    
        return selected.min()