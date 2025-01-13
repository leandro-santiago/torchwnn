import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "Wisard",    
]

class Wisard(nn.Module):
    def __init__(
        self,
        entry_size: int,
        n_classes: int,
        tuple_size: int              
    ) -> None:
        super().__init__()
        
        assert (entry_size % tuple_size) == 0
        
        self.entry_size = entry_size
        self.n_classes = n_classes
        self.tuple_size = tuple_size
        self.n_rams = entry_size // tuple_size        
        self.discriminators = [Discriminator(self.n_rams) for _ in range(n_classes)] 
        
        self.tuple_mapping = torch.empty((n_classes, entry_size), dtype=torch.long)
        for i in range(n_classes):      
            self.tuple_mapping[i] = torch.randperm(entry_size)

        self.tidx = torch.arange(tuple_size).flip(dims=(0,))        
        
    def fit(self, input: Tensor, target: Tensor):
        # Sort input by class id to perform random mapping once per class
        target, target_indices = torch.sort(target) 
        input = input[target_indices]

        # Recover number of samples by class
        _, target_counts = torch.unique_consecutive(target, return_counts = True)

        start_class = 0
        end_class = 0
        for i in range(self.n_classes):
            end_class += target_counts[i].item()
            
            # Apply random mapping to all samples of class i
            mapped_input = torch.index_select(input[start_class:end_class], 1, self.tuple_mapping[i])

            # Transform all tuples into numeric value for all samples of class i
            tuple_shape = (mapped_input.shape[0], self.n_rams, self.tuple_size)
            mapped_input = mapped_input.view(tuple_shape)
            mapped_input = (mapped_input << self.tidx).sum(dim=2)
            
            # Fit all mapped samples of class i
            self.discriminators[i].fit(mapped_input)            
            
            start_class = end_class
    
    def forward(self, samples: Tensor) -> Tensor:
        response = torch.empty((self.n_classes, samples.shape[0]), dtype=torch.int8)
        
        for i in range(self.n_classes):
            mapped_input = torch.index_select(samples, 1, self.tuple_mapping[i])

            # Transform all tuples into numeric value for all samples of class i
            tuple_shape = (mapped_input.shape[0], self.n_rams, self.tuple_size)
            mapped_input = mapped_input.view(tuple_shape)
            mapped_input = (mapped_input << self.tidx).sum(dim=2)            
            
            # Rank all mapped samples of class i
            response[i] = self.discriminators[i].rank(mapped_input)                      

        return response.transpose_(0,1)

    def predict(self, samples: Tensor) -> Tensor:
        return torch.argmax(self(samples), dim=-1)

class Discriminator:
    def __init__(self, n_rams: int):
        self.n_rams = n_rams
        self.rams = [{} for _ in range(n_rams)]

    def fit(self, data: Tensor):       
        data.transpose_(0,1)

        for ram, addresses in enumerate(data):
            for addr in addresses:
                self.rams[ram][addr.item()] = 1

    def rank(self, data: Tensor) -> Tensor:
        response = torch.zeros((data.shape[0],), dtype=torch.int8)
        data.transpose_(0,1)
        
        for ram, addresses in enumerate(data):
            trained_tuples = torch.tensor(list(self.rams[ram].keys()))
            response += torch.isin(addresses, trained_tuples).int()            

        return response
                
            