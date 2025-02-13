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
    "Wisard",
    "BloomWisard",    
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
    