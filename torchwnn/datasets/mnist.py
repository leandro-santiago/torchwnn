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
from torchvision import datasets, transforms

from torchwnn.datasets.dataset import Dataset

class Mnist(Dataset):
    name = "mnist"    
    categorical_features = []
    numeric_features = []
    numeric_range = (0, 255)

    def __init__(self, path = None):
        self.isimage = True

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(dtype=torch.int8),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])

        self.train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        self.X_train = self.train_dataset
        self.y_train = self.train_dataset.targets
        self.X_test = self.test_dataset
        self.y_test = self.test_dataset.targets
        
        self.num_features = len(self.train_dataset[0][0])
        self.num_classes = len(self.train_dataset.classes)

        #print(self.num_classes, self.data_size)
        #print(self.X_train)
        #print(self.y_train)

