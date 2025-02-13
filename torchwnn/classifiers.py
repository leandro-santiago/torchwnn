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
import torch.nn as nn
from torch import Tensor
from torchwnn.filters import BloomFilter
from torchwnn.cpp import functional
from torchwnn.functional import h3_generate

from sklearn.metrics import accuracy_score

__all__ = [
    "Discriminator",
    "BloomDiscriminator",
    "CuckooDiscriminator",
    "QuotientDiscriminator",
    "SketchDiscriminator",
    "Wisard",
    "BloomWisard",
    "CuckooWisard",
    "QuotientWisard",  
    "SketchWisard",  
]

class Discriminator:

    neuron_class = dict

    def __init__(self, n_neurons: int, bleaching = False) -> None:
        self.n_neurons = n_neurons
        self.neurons = [self.neuron_class() for _ in range(n_neurons)]
        self.bleach = 0
        self.bleaching = bleaching

    def set_bleach(self, value: int) -> None:
        self.bleach = value
    
    def get_minmax_val(self) -> tuple:
        max_val = 0
        min_val = 1 << 31

        for neuron in self.neurons:
            for val in neuron.values():
                if val > max_val:
                    max_val = val
                if val < min_val:
                    min_val = val
        return min_val, max_val

    def fit(self, data: Tensor) -> None:       
        data.transpose_(0,1)

        if self.bleaching:
            for neuron, addresses in enumerate(data):
                for addr in addresses:
                    if not addr.item() in self.neurons[neuron]:
                        self.neurons[neuron][addr.item()] = 1
                    
                    self.neurons[neuron][addr.item()] = self.neurons[neuron][addr.item()] + 1
        else:
            for neuron, addresses in enumerate(data):
                for addr in addresses:
                    self.neurons[neuron][addr.item()] = 1

    def rank(self, data: Tensor) -> Tensor:
        response = torch.zeros((data.shape[0],), dtype=torch.int8)
        data.transpose_(0,1)

        if self.bleach > 0:
            for neuron, addresses in enumerate(data):
                trained_tuples = torch.tensor(list(self.neurons[neuron].keys()))                
                selected = torch.isin(addresses, trained_tuples)

                for i in range(selected.shape[0]):
                    if selected[i]:
                        response[i] = response[i] + (self.neurons[neuron][addresses[i].item()] > self.bleach)                
                
        else:
            for neuron, addresses in enumerate(data):
                trained_tuples = torch.tensor(list(self.neurons[neuron].keys()))
                response += torch.isin(addresses, trained_tuples).int()


        return response
    
class BloomDiscriminator:
    
    def __init__(self, n_neurons: int, array_size: int) -> None:        
        self.n_neurons = n_neurons
        self.array_size = array_size
        self.neurons = torch.zeros((n_neurons, array_size), dtype=torch.uint8)
        self.bleach = 0

    def fit(self, data: Tensor) -> None:           
        functional.filter_multi_add(self.neurons, data)        

    def rank(self, data: Tensor) -> Tensor:
        response = functional.filter_multi_rank(self.neurons, data, self.bleach)
        return response    
    
    def set_bleach(self, value: int) -> None:
        self.bleach = value
        
    def get_minmax_val(self) -> tuple:
        max_val = 0
        min_val = 1 << 31

        for i in range(self.neurons.shape[0]):
            for j in range(self.neurons.shape[1]):
                if self.neurons[i, j].item() > max_val:
                    max_val = self.neurons[i, j].item()

                if (self.neurons[i, j].item() > 0) and (self.neurons[i, j].item() < min_val):
                    min_val = self.neurons[i, j].item()
        return min_val, max_val
    
class CuckooDiscriminator:
    
    def __init__(self, n_neurons: int, array_size: int, bucket_size: int, maxKick: int, fingerprint_matrix: Tensor, bleaching: bool = False) -> None:        
        self.n_neurons = n_neurons
        self.array_size = array_size
        self.bucket_size = bucket_size
        self.maxKick = maxKick
        self.fingerprint_matrix = fingerprint_matrix
        self.neurons = torch.full((n_neurons, bucket_size, array_size), -1, dtype=torch.int16)
        self.bucket_indexes = torch.zeros((n_neurons, self.array_size), dtype=torch.int32) 
        self.bleaching = bleaching
        self.bleach = 0
        
        if (self.bleaching):
            self.counters = torch.zeros((n_neurons, array_size), dtype=torch.uint32)     

    def fit(self, data: Tensor) -> None:
        # data: (n_samples x n_filters x 3), where last dimension has [i1, i2, fingerprint]  
        if (self.bleaching):
            functional.cuckoo_filter_multi_add_bleach(data, self.neurons, self.bucket_indexes, self.fingerprint_matrix, self.counters, self.maxKick)
        else:
            functional.cuckoo_filter_multi_add(data, self.neurons, self.bucket_indexes, self.fingerprint_matrix, self.maxKick)                        
    
    def fit_python(self, data: Tensor) -> None:
        # data: (n_samples x n_filters x 3), where last dimension has [i1, i2, fingerprint]  
        self.multi_add(data)
        
    def rank(self, data: Tensor) -> Tensor:
        # data: (n_samples x n_filters x 3), where last dimension has [i1, i2, fingerprint]  
        if (self.bleaching):
            response = functional.cuckoo_filter_multi_rank_bleach(self.neurons, self.counters, data, self.bleach)
        else:
            response = functional.cuckoo_filter_multi_rank(self.neurons, data)
        return response  

    def multi_add(self, data: Tensor) -> None:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                i1 = data[i][j][0].item()
                i2 = data[i][j][1].item()
                fingerprint = data[i][j][2].item()
                #print(i1, i2, fingerprint)
                bucket_index1 = self.bucket_indexes[j][i1].item()
                bucket_index2 = self.bucket_indexes[j][i2].item()
                
                if bucket_index1 < self.bucket_size:
                    selected = torch.isin(self.neurons[j,:, i1], fingerprint).any().item()
                    if not selected:
                        self.neurons[j][bucket_index1][i1] = fingerprint
                        self.bucket_indexes[j][i1] = self.bucket_indexes[j][i1].item() + 1
                elif bucket_index2 < self.bucket_size:
                    selected = torch.isin(self.neurons[j,:, i2], fingerprint).any().item()
                    if not selected:
                        self.neurons[j][bucket_index2][i2] = fingerprint
                        self.bucket_indexes[j][i2] = self.bucket_indexes[j][i2].item() + 1
                else:                
                    index = torch.randperm(2)[0]
                    index = data[i][j][index]
                    
                    for _ in range(self.maxKick):
                        bucket_i = torch.randperm(self.bucket_size)[0]     
                        fingerprint2 = self.neurons[j][bucket_i][index].item()              
                        self.neurons[j][bucket_i][index] = fingerprint
                        fingerprint = fingerprint2
        
                        hash_results = functional.h3_hash_int(torch.tensor([fingerprint]), self.fingerprint_matrix) 
                        index = index ^ hash_results[:, 0].item() 
                        bucket_index = self.bucket_indexes[j][index].item()
                        
                        if bucket_index < self.bucket_size:
                            self.neurons[j][bucket_index][index] = fingerprint
                            self.bucket_indexes[j][index] = self.bucket_indexes[j][index].item() + 1
                            break
    
    def set_bleach(self, value: int) -> None:
        self.bleach = value

    def get_minmax_val(self) -> tuple:
        max_val = 0
        min_val = 1 << 31

        if self.bleaching:
            for i in range(self.counters.shape[0]):
                for j in range(self.counters.shape[1]):
                    if self.counters[i, j].item() > max_val:
                        max_val = self.counters[i, j].item()

                    if (self.counters[i, j].item() > 0) and (self.counters[i, j].item() < min_val):
                        min_val = self.counters[i, j].item()
        return min_val, max_val
    
class QuotientDiscriminator:
    
    def __init__(self, n_neurons: int, array_size: int, remainder_size: int, bleaching: bool = False) -> None:        
        self.n_neurons = n_neurons
        self.array_size = array_size
        self.remainder_size = remainder_size
        self.remainder_mask = (1 << remainder_size) - 1
        self.neurons = torch.zeros((n_neurons, array_size), dtype=torch.uint8)
        self.metadata = torch.zeros((n_neurons, array_size), dtype=torch.uint8)
        self.bit_occupieds = 1 # metadata bit
        self.bit_runends = 2 # metadata bit used only by RS Quotient Filter operations
        self.bit_continuations = 2 # metadata bit
        self.bit_shifteds = 4 # metadata bit
        self.bit_empty = 7 # metadata bit
        self.bleaching = bleaching
        self.bleach = 0
        
        if (self.bleaching):
            self.counters = torch.zeros((n_neurons, array_size), dtype=torch.uint32)

    def fit(self, data: Tensor) -> None:  
        # data: (n_samples x n_filters)
        if (self.bleaching):
            functional.quotient_filter_multi_add_bleach(self.neurons, self.metadata, self.counters, data, self.remainder_size)
        else:
            functional.quotient_filter_multi_add(self.neurons, self.metadata, data, self.remainder_size)

    def fit_python(self, data: Tensor) -> None:  
        # data: (n_samples x n_filters)
        self.multi_add(data)        

    # multi_add using rank_select operations according to RS-QF (Rank-Select Quotient Filter)
    # This method is slower than self.multi_add
    def multi_add_rs(self, data: Tensor) -> None:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                quotient = data[i][j].item() >> self.remainder_size
                remainder = data[i][j].item() & self.remainder_mask
                last_index = self.rank_select(j, quotient)

                if quotient > last_index:
                    self.neurons[j, quotient] = remainder
                    self.metadata[j, quotient] |= self.bit_runends 
                else:
                    last_index += 1
                    free_index = self.next_free_slot(j, last_index)

                    while free_index > last_index:
                        self.neurons[j, free_index] = self.neurons[j, free_index - 1]
                        self.metadata[j, free_index] |= (self.metadata[j, free_index - 1] & self.bit_runends)
                        free_index -= 1

                    self.neurons[j, last_index] = remainder
                    if self.metadata[j, quotient] & self.bit_occupieds:
                        self.metadata[j, last_index - 1] -= self.bit_runends 
                    
                    self.metadata[j, last_index] |= self.bit_runends 

                self.metadata[j, quotient] |= self.bit_occupieds
    
    def multi_add(self, data: Tensor) -> None:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                quotient = data[i][j].item() >> self.remainder_size
                remainder = data[i][j].item() & self.remainder_mask    
                self.add(j, quotient, remainder)

    def right_shift(self, neuron: int, index: int) -> int:
        for i in range(index + 1, self.array_size):
            if ((self.metadata[neuron, i] & self.bit_empty) == 0):
                for j in range(i-1, index-1, -1):
                    occupied = self.metadata[neuron, i] & self.bit_occupieds
                    self.metadata[neuron, i] = self.metadata[neuron, j]
                    self.neurons[neuron, i] = self.neurons[neuron, j]

                    if (occupied):
                        self.metadata[neuron, i] |= self.bit_occupieds
                    else:
                        self.metadata[neuron, i] &= ~self.bit_occupieds

                    self.metadata[neuron, i] |= self.bit_shifteds

                    i -= 1
                    if (i <= 0):
                        break
                return 1        
        return 0
    
    def add(self, neuron: int, quotient: int, remainder: int) -> int:
        start_index = quotient    
            
        # Walk back to find the beginning of cluster
        while ((self.metadata[neuron, start_index] & self.bit_shifteds) > 0):
            start_index -= 1
        
        # Walk forward to find the start of the run
        run_index = start_index
        while (start_index < quotient):
            run_index += 1
            while ((run_index < self.array_size) and ((self.metadata[neuron, run_index] & self.bit_continuations) > 0)):
                run_index += 1

            start_index += 1
            while ((start_index < run_index) and ((self.metadata[neuron, start_index] & self.bit_occupieds) == 0)):
                start_index += 1

            if (run_index == self.array_size):
                return 0
        
        ## Found start of a run
    
        # First case if slot is empty
        if ((self.metadata[neuron, run_index] & self.bit_empty) == 0):
            # Mark canonical slot as occupied
            self.metadata[neuron, quotient] |= self.bit_occupieds
            
            if (run_index != quotient):
                self.metadata[neuron, run_index] |= self.bit_shifteds
            self.neurons[neuron, run_index] = remainder            
            return 1
        
        # Check if the start of run has to be shifted. If so, then IS_CONTINUATION is set in the shifted remainder.
        old_remainder = self.neurons[neuron, run_index]

        if (remainder < old_remainder.item()):
            # Old remainder sets CONTINUATION
            self.metadata[neuron, run_index] |= self.bit_continuations

            if (self.right_shift(neuron, run_index)):
                #print("Shifted: ", self.metadata)
                # Mark canonical slot as occupied
                self.metadata[neuron, quotient] |= self.bit_occupieds
                # New remainder unsets CONTINUATION
                self.metadata[neuron, run_index] &= ~self.bit_continuations
                self.neurons[neuron, run_index] = remainder
                return 1
            else:
                self.metadata[neuron, run_index] &= ~self.bit_continuations
                return 0
            
        elif (remainder == old_remainder):
            self.metadata[neuron, quotient] |= self.bit_occupieds
            return 1

        run_index += 1

        # Case that look the subsequent slots to insert new remainder in order.
        while ((run_index < self.array_size) and ((self.metadata[neuron, run_index] & self.bit_continuations) > 0)):
            old_remainder = self.neurons[neuron, run_index]

            if (remainder < old_remainder.item()):
                if (self.right_shift(neuron, run_index)):
                    # Mark canonical slot as occupied
                    self.metadata[neuron, quotient] |= self.bit_occupieds
                    self.neurons[neuron, run_index] = remainder
                    return 1
                else:
                    return 0
                
            elif (remainder == old_remainder):
                self.metadata[neuron, quotient] |= self.bit_occupieds
                return 1
                
            run_index += 1


        if (run_index == self.array_size):
            return 0

        # Case where was walked the whole run and not found appropriate slot.
        # A new slot will be added to the run. If the slot is full, it is shifted first.
        if (((self.metadata[neuron, run_index] & self.bit_empty) == 0) or (self.right_shift(neuron, run_index))):
            # Mark canonical slot as occupied
            self.metadata[neuron, quotient] |= self.bit_occupieds
            self.metadata[neuron, run_index] |= self.bit_continuations
            self.metadata[neuron, run_index] |= self.bit_shifteds
            self.neurons[neuron, run_index] = remainder
            return 1

        return 0
    
    def rank_select(self, neuron: int, index: int):
        count_occupieds = 0
        count_runends = 0
        runends_shift = self.bit_runends - 1
        last_run_index = -1

        for i in range(index + 1):
            count_occupieds += self.metadata[neuron, i] & self.bit_occupieds
            runends_val = (self.metadata[neuron, i] & self.bit_runends) >> runends_shift
            count_runends += runends_val
            last_run_index = i if runends_val > 0 else last_run_index

                      
        if count_occupieds > count_runends:
            while count_occupieds > count_runends:
                runends_val = (self.metadata[neuron, i] & self.bit_runends) >> runends_shift
                count_runends += runends_val
                i += 1

            last_run_index = i - 1

        elif count_occupieds < count_runends:
            while count_occupieds < count_runends:
                runends_val = (self.metadata[neuron, i] & self.bit_runends) >> runends_shift
                count_runends -= runends_val
                i -= 1
            
            runends_val = (self.metadata[neuron, i] & self.bit_runends) >> runends_shift    
            while runends_val == 0:    
                i -= 1
                runends_val = (self.metadata[neuron, i] & self.bit_runends) >> runends_shift                
            
            last_run_index = i                 
           
        return last_run_index
    
    def next_free_slot(self, neuron: int, index: int):
        free_index = index
        last_index = self.rank_select(neuron, index)
        
        while free_index <= last_index:
            free_index = last_index + 1        
            last_index = self.rank_select(neuron, free_index)
        return free_index
    
    def rank(self, data: Tensor) -> Tensor:
        if (self.bleaching):
            response = functional.quotient_filter_multi_rank_bleach(self.neurons, self.metadata, self.counters, data, self.remainder_size, self.bleach) 
        else:
            response = functional.quotient_filter_multi_rank(self.neurons, self.metadata, data, self.remainder_size) 
        return response         
    
    def rank_python(self, data: Tensor) -> Tensor:
        return self.multi_rank(data)

    def multi_rank(self, data: Tensor) -> Tensor:
        all_response = torch.zeros((data.shape[0]), dtype=torch.int32)

        for i in range(data.shape[0]):            
            for j in range(data.shape[1]):
                quotient = data[i][j].item() >> self.remainder_size                  
                remainder = data[i][j].item() & self.remainder_mask
                #print("RANK - Fingerprint: ", data[i][j], ", ", quotient, ", ", remainder)
                all_response[i] = all_response[i] + self.check_member(j, quotient, remainder)           

        return all_response  

    def check_member(self, neuron: int, quotient: int, remainder: int) -> int:        

        if self.metadata[neuron, quotient] & self.bit_occupieds:
            start_index = quotient
            # Walk back to find the beginning of cluster
            while ((self.metadata[neuron, start_index] & self.bit_shifteds) > 0):
                start_index -= 1
            
            # Walk forward to find the start of the run
            run_index = start_index
            while (start_index < quotient):
                run_index += 1
                while ((run_index < self.array_size) and ((self.metadata[neuron, run_index] & self.bit_continuations) > 0)):
                    run_index += 1

                start_index += 1
                while ((start_index < run_index) and ((self.metadata[neuron, start_index] & self.bit_occupieds) == 0)):
                    start_index += 1                
                
                if (run_index == self.array_size):
                    return 0


            if self.neurons[neuron, run_index].item() == remainder: 
                return 1
            
            run_index += 1

            while (run_index < self.array_size) and ((self.metadata[neuron,run_index] & self.bit_continuations) > 0):
                if self.neurons[neuron, run_index].item() == remainder: 
                    return 1
                run_index += 1      
        
        return 0

    def set_bleach(self, value: int) -> None:
        self.bleach = value
        
    def get_minmax_val(self) -> tuple:
        max_val = 0
        min_val = 1 << 31
        
        if self.bleaching:
            for i in range(self.counters.shape[0]):
                for j in range(self.counters.shape[1]):
                    if self.counters[i, j].item() > max_val:
                        max_val = self.counters[i, j].item()

                    if (self.counters[i, j].item() > 0) and (self.counters[i, j].item() < min_val):
                        min_val = self.counters[i, j].item()
        return min_val, max_val

class SketchDiscriminator:
    
    def __init__(self, n_neurons: int, array_size: int) -> None:        
        self.n_neurons = n_neurons
        self.array_size = array_size
        self.neurons = torch.zeros((n_neurons, array_size), dtype=torch.uint8)
        self.bleach = 0

    def fit(self, data: Tensor) -> None:    
        # data: (n_samples x n_filters x n_hashes)       
        functional.filter_multi_add(self.neurons, data.transpose(1, 2))                

    def rank(self, data: Tensor) -> Tensor:
        # data: (n_samples x n_filters x n_hashes)
        response = functional.sketch_multi_rank(self.neurons, data.transpose(1, 2), self.bleach)        
        return response  

    def set_bleach(self, value: int) -> None:
        self.bleach = value
    
    def get_minmax_val(self) -> tuple:
        max_val = 0
        min_val = 1 << 31

        for i in range(self.neurons.shape[0]):
            for j in range(self.neurons.shape[1]):
                if self.neurons[i, j].item() > max_val:
                    max_val = self.neurons[i, j].item()

                if (self.neurons[i, j].item() > 0) and (self.neurons[i, j].item() < min_val):
                    min_val = self.neurons[i, j].item()
        return min_val, max_val
        
    
class Wisard(nn.Module):

    discriminator_class = Discriminator 

    def __init__(
        self,
        entry_size: int,
        n_classes: int,
        tuple_size: int,
        bleaching: bool = False              
    ) -> None:
        super().__init__()
        
        #assert (entry_size % tuple_size) == 0
        
        self.entry_size = entry_size
        self.n_classes = n_classes
        self.tuple_size = tuple_size
        self.n_neurons = (entry_size // tuple_size) + ((entry_size % tuple_size) > 0)
        self.total_entry_size = self.n_neurons * self.tuple_size
        self.pad_bits = self.total_entry_size - self.entry_size
        self.bleaching = bleaching  
            
                
        self.tuple_mapping = torch.empty((n_classes, self.total_entry_size), dtype=torch.long)
        for i in range(n_classes):      
            self.tuple_mapping[i] = torch.randperm(self.total_entry_size)

        self.tidx = torch.arange(tuple_size).flip(dims=(0,))        

        self.create_discriminators()
    
    def create_discriminators(self) -> None:
        self.discriminators = [self.discriminator_class(self.n_neurons, bleaching=self.bleaching) for _ in range(self.n_classes)] 
        
    def fit(self, input: Tensor, target: Tensor) -> None:
        if self.pad_bits > 0:
            input = torch.nn.functional.pad(input, (0, self.pad_bits))

        # Sort input by class id to perform random mapping once per class
        target, target_indices = torch.sort(target) 
        input = input[target_indices]

        # Recover number of samples by class
        target_outputs, target_counts = torch.unique_consecutive(target, return_counts = True)
        
        start_class = 0
        end_class = 0
        for i in range(target_outputs.shape[0]):
            end_class += target_counts[i].item()
            label = target_outputs[i].item()

            # Apply random mapping to all samples of class i
            mapped_input = torch.index_select(input[start_class:end_class], 1, self.tuple_mapping[label])

            # Transform all tuples into numeric value for all samples of class i
            tuple_shape = (mapped_input.shape[0], self.n_neurons, self.tuple_size)
            mapped_input = mapped_input.view(tuple_shape)
            mapped_input = self.transform(mapped_input)  
            
            # Fit all mapped samples of class i
            self.discriminators[label].fit(mapped_input)            
            
            start_class = end_class
    
    def fit_bleach(self, input: Tensor, target: Tensor) -> None:
        max_val = 0
        min_val = 1 << 31
        
        for discriminator in self.discriminators:
            val0, val1 = discriminator.get_minmax_val()
            if val1 > max_val:
                max_val = val1
            
            if val0 < min_val:
                min_val = val0            
        
        best_bleach = max_val // 2
        step = max(max_val // 4, 1)
        bleach_accuracies = {}

        self.bleach = 0
        acc_without_bleach = self.accuracy(input, target)

        while True:
            bleach_values = [best_bleach-step, best_bleach, best_bleach+step]
            accuracies = []
            for bleach in bleach_values:
                if bleach in bleach_accuracies:
                    accuracies.append(bleach_accuracies[bleach])
                elif bleach < 1:
                    accuracies.append(0)
                else:
                    self.set_bleach(bleach)
                    acc = self.accuracy(input, target)
                    bleach_accuracies[bleach] = acc 
                    accuracies.append(acc)                    

            new_best_bleach = bleach_values[accuracies.index(max(accuracies))]
            
            if (new_best_bleach == best_bleach) and (step == 1):
                break
            best_bleach = new_best_bleach
            if step > 1:
                step //= 2
        
        if acc_without_bleach < max(accuracies):        
            self.set_bleach(best_bleach)
            self.bleach = best_bleach
        else:
            self.set_bleach(0)
            self.bleach = 0
        
    def forward(self, samples: Tensor) -> Tensor:
        if self.pad_bits > 0:
            samples = torch.nn.functional.pad(samples, (0, self.pad_bits))

        response = torch.empty((self.n_classes, samples.shape[0]), dtype=torch.int8)
        
        for i in range(self.n_classes):
            mapped_input = torch.index_select(samples, 1, self.tuple_mapping[i])

            # Transform all tuples into numeric value for all samples of class i
            tuple_shape = (mapped_input.shape[0], self.n_neurons, self.tuple_size)
            mapped_input = mapped_input.view(tuple_shape)
            mapped_input = self.transform(mapped_input)            
            
            # Rank all mapped samples of class i
            response[i] = self.discriminators[i].rank(mapped_input)                      

        return response.transpose_(0,1)

    def predict(self, samples: Tensor) -> Tensor:
        return torch.argmax(self(samples), dim=-1)

    def transform(self, mapped_data: Tensor) -> Tensor:
        # Transform all tuples into numeric value for all samples of class i
        return (mapped_data << self.tidx).sum(dim=2)  

    def set_bleach(self, value: int) -> None:        
        for discriminator in self.discriminators:
            discriminator.set_bleach(value)  

    def accuracy(self, samples: Tensor, target: Tensor) -> Tensor:
        predictions = self.predict(samples)
        return accuracy_score(predictions, target)

class BloomWisard(Wisard):
    
    discriminator_class = BloomDiscriminator              

    def __init__(
        self,
        entry_size: int,
        n_classes: int,
        tuple_size: int,        
        filter_size: int = None,   
        n_hashes: int = None,
        capacity: int = 100,
        error: float = 0.5,
    ) -> None:
                
        self.capacity = capacity
        self.error = error

        assert (filter_size is None and n_hashes is None) or (filter_size > 0 and n_hashes > 0)

        if (not filter_size):
            self.filter_size = BloomFilter.calculate_num_bits(capacity, error)            
            self.n_hashes = BloomFilter.calculate_num_hashes(capacity, error)
        else :    
            self.filter_size = filter_size     
            self.n_hashes = n_hashes

        super().__init__(entry_size, n_classes, tuple_size) 

        self.hash_matrix = h3_generate(self.tuple_size, self.filter_size, self.n_hashes)

    def create_discriminators(self) -> None:
        self.discriminators = [self.discriminator_class(self.n_neurons, self.filter_size) for _ in range(self.n_classes)] 

    def transform(self, mapped_data: Tensor) -> Tensor:
        # Generate hashed values for all samples of class i
        return functional.h3_multi_hash(mapped_data, self.hash_matrix)


# Cuckoo Filter implementation was based on the following paper:
# Cuckoo Filter: Practically Better Than Bloom
# Link: <https://dl.acm.org/doi/10.1145/2674005.2674994>  
class CuckooWisard(Wisard):
    
    discriminator_class = CuckooDiscriminator              

    def __init__(
        self,
        entry_size: int,
        n_classes: int,
        tuple_size: int,        
        filter_size: int = 128,
        bucket_size: int = 4, 
        fingerprint_size: int = 3, 
        maxKick: int = 50,
        bleaching: bool = False
    ) -> None:                
        
        self.filter_size = filter_size
        self.bucket_size = bucket_size
        self.fingerprint_size = fingerprint_size
        self.fingerprint_mask = (1 << fingerprint_size) - 1 
        self.maxKick = maxKick

        self.fingerprint_matrix = h3_generate(self.fingerprint_size, filter_size,  1)        

        super().__init__(entry_size, n_classes, tuple_size, bleaching=bleaching) 
        
        self.hash_matrix = h3_generate(self.tuple_size, self.filter_size, 2)        

    def create_discriminators(self) -> None:
        self.discriminators = [self.discriminator_class(self.n_neurons, self.filter_size, self.bucket_size, self.maxKick, self.fingerprint_matrix, bleaching=self.bleaching) for _ in range(self.n_classes)] 

    def transform(self, mapped_data: Tensor) -> Tensor:
        # Generate hashed values for all samples of class i
        # Results of two hash functions:
        # 1: i1 = hash1(x)
        # 2: fingerprint = hash2(x)
        hash_results = functional.h3_multi_hash(mapped_data, self.hash_matrix)

        # Clipping fingerprint
        fingerprint = hash_results[..., 1] & self.fingerprint_mask                 
        #fingerprint2 = (mapped_data[..., :self.fingerprint_size] << self.fidx).sum(dim=2)         

        # Generate hashed values of fingerprints
        hash_fingerprint = functional.h3_multi_hash_int(fingerprint, self.fingerprint_matrix)
        
        # i2 = i1 xor hash(fingerprint), i1 is in h[:, :, 0]
        hash_fingerprint.bitwise_xor_(hash_results[:, :, :-1])        
        
        # transform_result has dimension (n_samples x n_filters x 3). 
        # Last dimension has [i1, i2, figerprint] 
        transform_result = torch.cat((hash_results[:, :, :-1], hash_fingerprint, fingerprint.unsqueeze(2)), 2)
        
        return transform_result
        
# Quotient Filter has two implementations:
# 1) Quotient Filter: 
# Paper: Don't thrash: how to cache your hash on flash <https://dl.acm.org/doi/10.14778/2350229.2350275>
# 
# 2) RS-QF (Rank-Select Quotient Filter):
# Paper: A General-Purpose Counting Filter: Making Every Bit Count <https://dl.acm.org/doi/10.1145/3035918.3035963> 
class QuotientWisard(Wisard):
    
    discriminator_class = QuotientDiscriminator              

    def __init__(
        self,
        entry_size: int,
        n_classes: int,
        tuple_size: int,        
        filter_size: int = 128,
        remainder_size: int = 4, 
        bleaching: bool = False        
    ) -> None:
                
        assert(filter_size >= (1 << (tuple_size - remainder_size)))
        
        self.filter_size = filter_size
        self.remainder_size = remainder_size                   
        
        super().__init__(entry_size, n_classes, tuple_size, bleaching=bleaching)         

    def create_discriminators(self) -> None:
        self.discriminators = [self.discriminator_class(self.n_neurons, self.filter_size, self.remainder_size, bleaching=self.bleaching) for _ in range(self.n_classes)] 

    def transform(self, mapped_data: Tensor) -> Tensor:
        # Generate hashed values for all samples of class i      
        return (mapped_data << self.tidx).sum(dim=2)


class SketchWisard(Wisard):
    
    discriminator_class = SketchDiscriminator              

    def __init__(
        self,
        entry_size: int,
        n_classes: int,
        tuple_size: int,        
        filter_size: int = 128,           
    ) -> None:
                
        self.filter_size = filter_size             

        super().__init__(entry_size, n_classes, tuple_size) 

        self.hash_matrix = h3_generate(self.tuple_size, self.filter_size, self.n_neurons)

    def create_discriminators(self) -> None:
        self.discriminators = [self.discriminator_class(self.n_neurons, self.filter_size) for _ in range(self.n_classes)] 

    def transform(self, mapped_data: Tensor) -> Tensor:
        # Generate hashed values for all samples of class i
        return functional.h3_multi_hash(mapped_data, self.hash_matrix)