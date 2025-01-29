#
# Code adapted from DWN repository: https://github.com/alanbacellar/DWN
#
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

class Thermometer:
    def __init__(self, num_bits=1, feature_wise=True):
        
        assert num_bits > 0
        assert type(feature_wise) is bool

        self.num_bits = int(num_bits)
        self.feature_wise = feature_wise
        self.thresholds = None

    def get_thresholds(self, x):
        min_value = x.min(dim=0)[0] if self.feature_wise else x.min()
        max_value = x.max(dim=0)[0] if self.feature_wise else x.max()
        return min_value.unsqueeze(-1) + torch.arange(1, self.num_bits+1).unsqueeze(0) * ((max_value - min_value) / (self.num_bits + 1)).unsqueeze(-1)

    def fit(self, x):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        self.thresholds = self.get_thresholds(x)
        return self
    
    def binarize(self, x):
        if self.thresholds is None:
            raise 'need to fit before calling apply'
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        x = x.unsqueeze(-1)
        return (x > self.thresholds).int()

class GaussianThermometer(Thermometer):
    def __init__(self, num_bits=1, feature_wise=True):
        super().__init__(num_bits, feature_wise)

    def get_thresholds(self, x):
        std_skews = torch.distributions.Normal(0, 1).icdf(torch.arange(1, self.num_bits+1)/(self.num_bits+1))
        mean = x.mean(dim=0) if self.feature_wise else x.mean()
        std = x.std(dim=0) if self.feature_wise else x.std() 
        thresholds = torch.stack([std_skew * std + mean for std_skew in std_skews], dim=-1)
        return thresholds
    
class DistributiveThermometer(Thermometer):
    def __init__(self, num_bits=1, feature_wise=True):
        super().__init__(num_bits, feature_wise)

    def get_thresholds(self, x):
        data = torch.sort(x.flatten())[0] if not self.feature_wise else torch.sort(x, dim=0)[0]
        indicies = torch.tensor([int(data.shape[0]*i/(self.num_bits+1)) for i in range(1, self.num_bits+1)])
        thresholds = data[indicies]
        return torch.permute(thresholds, (*list(range(1, thresholds.ndim)), 0))
