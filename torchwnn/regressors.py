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
from tqdm import trange
import math

__all__ = [
    "Discriminator",
    "RegressionWisard",    
]

class Discriminator:

    neuron_class = dict

    def __init__(self, n_neurons: int) -> None:
        self.n_neurons = n_neurons
        self.neurons = [self.neuron_class() for _ in range(n_neurons)]        
    
    def fit(self, data: Tensor, target: Tensor, minZero: Tensor, minOne: Tensor) -> None:       
        if minZero is None:
            for neuron, addresses in enumerate(data):
                for i in range(data.shape[1]):
                    addr = addresses[i].item()
                    y = target[i].item()

                    if not addr in self.neurons[neuron]:
                        self.neurons[neuron][addr] = [1, y, 0.0]
                    else:
                        self.neurons[neuron][addr][0] += 1
                        self.neurons[neuron][addr][1] += y 
        else:            
            for neuron, addresses in enumerate(data):
                for i in range(data.shape[1]):
                    if minZero[neuron][i] or minOne[neuron][i]:                        
                        continue
                    
                    addr = addresses[i].item()
                    y = target[i].item()

                    if not addr in self.neurons[neuron]:
                        self.neurons[neuron][addr] = [1, y, 0.0]
                    else:
                        self.neurons[neuron][addr][0] += 1
                        self.neurons[neuron][addr][1] += y 
    
    def fit_adapt(self, data: Tensor, errors: Tensor) -> None:       
        # Fit errors in each neuron
        for neuron, addresses in enumerate(data):
            for i in range(data.shape[1]):
                if addr in self.neurons[neuron]:
                    addr = addresses[i].item()
                    self.neurons[neuron][addr][2] += errors[i].item() 
        
        # Update neurons with errors
        for i in range(self.n_neurons):
           for addr in self.neurons[i].keys():  
               self.neurons[i][addr][1] += (self.neurons[i][addr][2] / self.neurons[i][addr][0])
               self.neurons[i][addr][2] = 0

    def predict(self, data: Tensor, centrality = "mean", power = 2) -> Tensor:
        if centrality == "powermean":
            return self.calculate_powermean(data, power)
        elif centrality == "median":
            return self.calculate_median(data)
        elif centrality == "harmonicmean":
            return self.calculate_harmonicmean(data)
        elif centrality == "harmonicpowermean":
            return self.calculate_harmonicpowermean(data, power)
        elif centrality == "geometricmean":
            return self.calculate_geometricmean(data)
        elif centrality == "exponentialmean":
            return self.calculate_exponentialmean(data)
        else:
            return self.calculate_mean(data)

    def calculate_mean(self, data: Tensor) -> Tensor:
        response = torch.zeros((data.shape[1],), dtype=torch.float64)
        counters = torch.zeros((data.shape[1],), dtype=torch.int64)        

        for neuron, addresses in enumerate(data):
            trained_tuples = torch.tensor(list(self.neurons[neuron].keys()))                
            selected = torch.isin(addresses, trained_tuples)

            for i in range(selected.shape[0]):
                if selected[i]:
                    content_neuron = self.neurons[neuron][addresses[i].item()] 
                    response[i] += content_neuron[1]
                    counters[i] += content_neuron[0]
        
        response = torch.nan_to_num(response / counters)                  
        return response
    
    def calculate_powermean(self, data: Tensor, power: int) -> Tensor:
        response = torch.zeros((data.shape[1],), dtype=torch.float64)
        counters = torch.zeros((data.shape[1],), dtype=torch.int64)                

        for neuron, addresses in enumerate(data):
            trained_tuples = torch.tensor(list(self.neurons[neuron].keys()))                
            selected = torch.isin(addresses, trained_tuples)

            for i in range(selected.shape[0]):
                if selected[i]:
                    content_neuron = self.neurons[neuron][addresses[i].item()] 
                    
                    if (content_neuron[0] != 0):
                        response[i] += pow(content_neuron[1]/content_neuron[0], power)                        
                        counters[i] += 1
        
        
        response = torch.nan_to_num(torch.pow(response / counters, 1/power))                  
        #print(f"Response: {response}, Counters: {counters}")
        return response

    def calculate_median(self, data: Tensor) -> Tensor:
        medians = torch.zeros((data.shape[1], self.n_neurons), dtype=torch.float64)

        for neuron, addresses in enumerate(data):
            trained_tuples = torch.tensor(list(self.neurons[neuron].keys()))                
            selected = torch.isin(addresses, trained_tuples)

            for i in range(selected.shape[0]):
                if selected[i]:
                    content_neuron = self.neurons[neuron][addresses[i].item()] 
                    medians[i, neuron] = (content_neuron[1] / content_neuron[0])

        values, _ = torch.median(medians, 1)
        return values

    def calculate_harmonicmean(self, data: Tensor) -> Tensor:
        response = torch.zeros((data.shape[1],), dtype=torch.float64)
        counters = torch.zeros((data.shape[1],), dtype=torch.int64)        

        for neuron, addresses in enumerate(data):
            trained_tuples = torch.tensor(list(self.neurons[neuron].keys()))                
            selected = torch.isin(addresses, trained_tuples)

            for i in range(selected.shape[0]):
                if selected[i]:
                    content_neuron = self.neurons[neuron][addresses[i].item()] 
                    response[i] += 1/(content_neuron[1]/content_neuron[0])
                    counters[i] += 1
        
        response = torch.nan_to_num(counters/response)                  
        return response

    def calculate_harmonicpowermean(self, data: Tensor, power: int) -> Tensor:
        response = torch.zeros((data.shape[1],), dtype=torch.float64)
        counters = torch.zeros((data.shape[1],), dtype=torch.int64)        

        for neuron, addresses in enumerate(data):
            trained_tuples = torch.tensor(list(self.neurons[neuron].keys()))                
            selected = torch.isin(addresses, trained_tuples)

            for i in range(selected.shape[0]):
                if selected[i]:
                    content_neuron = self.neurons[neuron][addresses[i].item()]                     
                    if (content_neuron[0] != 0):
                        response[i] += 1/pow(content_neuron[1]/content_neuron[0], power)
                        counters[i] += 1
                    
        
        response = torch.nan_to_num(torch.pow(counters/response, 1/power))                
        return response


    def calculate_geometricmean(self, data: Tensor) -> Tensor:
        response = torch.ones((data.shape[1],), dtype=torch.float64)
        counters = torch.zeros((data.shape[1],), dtype=torch.int64)        

        for neuron, addresses in enumerate(data):
            trained_tuples = torch.tensor(list(self.neurons[neuron].keys()))                
            selected = torch.isin(addresses, trained_tuples)

            for i in range(selected.shape[0]):
                if selected[i]:
                    content_neuron = self.neurons[neuron][addresses[i].item()] 

                    if (content_neuron[0] != 0):
                        response[i] *= content_neuron[1]/content_neuron[0]
                        counters[i] += 1
        
        response = torch.nan_to_num(torch.pow(response, 1/counters))                

        return response

    def calculate_exponentialmean(self, data: Tensor) -> Tensor:
        response = torch.zeros((data.shape[1],), dtype=torch.float64)
        counters = torch.zeros((data.shape[1],), dtype=torch.int64)        

        for neuron, addresses in enumerate(data):
            trained_tuples = torch.tensor(list(self.neurons[neuron].keys()))                
            selected = torch.isin(addresses, trained_tuples)

            for i in range(selected.shape[0]):
                if selected[i]:
                    content_neuron = self.neurons[neuron][addresses[i].item()] 
                    
                    if (content_neuron[0] != 0):
                        response[i] += math.exp(content_neuron[1]/content_neuron[0])
                        counters[i] += 1
        
        response = torch.nan_to_num(torch.log(response/counters))                  

        return response

# Regression implementation was based on the following paper:
# Extending the weightless WiSARD classifier for regression
# Link: <https://www.sciencedirect.com/science/article/abs/pii/S092523122030504X> 
class RegressionWisard(nn.Module):

    discriminator_class = Discriminator 

    def __init__(
        self,
        entry_size: int,
        tuple_size: int,
        minZero: int = 0,
        minOne: int = 0,
        epoch: int = 0
    ) -> None:
        super().__init__()
        
        self.entry_size = entry_size
        self.tuple_size = tuple_size
        self.n_regressors = 1        
        self.n_neurons = (entry_size // tuple_size) + ((entry_size % tuple_size) > 0)
        self.total_entry_size = self.n_neurons * self.tuple_size
        self.pad_bits = self.total_entry_size - self.entry_size
        self.minZero = minZero
        self.minOne = minOne
        self.epoch = epoch
                
        self.tuple_mapping = torch.empty((self.n_regressors, self.total_entry_size), dtype=torch.long)
        
        for i in range(self.n_regressors):      
            self.tuple_mapping[i] = torch.randperm(self.total_entry_size)

        self.tidx = torch.arange(tuple_size).flip(dims=(0,))        

        self.create_discriminators()
    
    def create_discriminators(self) -> None:
        self.discriminators = [self.discriminator_class(self.n_neurons) for _ in range(self.n_regressors)] 
        
    def fit(self, input: Tensor, target: Tensor) -> None:
        if self.total_entry_size > input.shape[1]:
            input = torch.nn.functional.pad(input, (0, self.pad_bits))

        # Apply random mapping to all samples of class i
        mapped_input = torch.index_select(input, 1, self.tuple_mapping[0])

        # Transform all tuples into numeric value for all samples of class i
        tuple_shape = (mapped_input.shape[0], self.n_neurons, self.tuple_size)
        mapped_input = mapped_input.view(tuple_shape)
        
        mapped_ones = None
        mapped_zeros = None
        if (self.minOne > 0 or self.minZero > 0):
            mapped_ones = mapped_input.sum(dim=2)        
            mapped_zeros = self.tuple_size - mapped_ones  
            mapped_ones = mapped_ones < self.minOne
            mapped_zeros = mapped_zeros < self.minZero
            mapped_ones.transpose_(0, 1)
            mapped_zeros.transpose_(0, 1)

        mapped_input = self.transform(mapped_input)  

        # Fit all mapped samples of class i
        mapped_input.transpose_(0, 1)        
        self.discriminators[0].fit(mapped_input, target, mapped_zeros, mapped_ones)  

        if self.epoch > 0:
            for _ in trange(0, self.epoch, desc="fit"):   
                pred = self.discriminators[0].predict(mapped_input)
                error = target - pred
                self.discriminators[0].fit_adapt(mapped_input, error) 
                
    
    def forward(self, samples: Tensor, centrality: str, power: int) -> Tensor:
        if self.total_entry_size > samples.shape[1]:
            samples = torch.nn.functional.pad(samples, (0, self.pad_bits))

        response = torch.empty((samples.shape[0]), dtype=torch.float64)        
        
        mapped_input = torch.index_select(samples, 1, self.tuple_mapping[0])

        # Transform all tuples into numeric value for all samples of class i
        tuple_shape = (mapped_input.shape[0], self.n_neurons, self.tuple_size)
        mapped_input = mapped_input.view(tuple_shape)
        mapped_input = self.transform(mapped_input)            
        
        # Rank all mapped samples of class i
        mapped_input.transpose_(0, 1)        
        response = self.discriminators[0].predict(mapped_input, centrality, power)                      

        return response

    def predict(self, samples: Tensor, centrality: str = "mean", power: int = 2) -> Tensor:
        return self(samples, centrality, power)

    def transform(self, mapped_data: Tensor) -> Tensor:
        # Transform all tuples into numeric value for all samples of class i
        return (mapped_data << self.tidx).sum(dim=2) 
    