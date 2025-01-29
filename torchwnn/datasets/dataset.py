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

from typing import Dict
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from torch.utils import data

class Dataset:
    isimage = False
    
    def load_uci_repo(self):
        dataset = fetch_ucirepo(id=self.id)            
        self.features = dataset.data.features
        self.targets = dataset.data.targets
        self.target_col = dataset.metadata.target_col[0] 
        self.num_features = dataset.metadata["num_features"]        

    def get_unique_categories_values(self) -> Dict[str, list]:
        all_unique_cat_values = {}
        
        for feature in self.categorical_features:
            unique_values = np.sort(self.features[feature].unique()).tolist()
            all_unique_cat_values[feature] = unique_values
        return all_unique_cat_values  

    def get_range_numeric_values(self):
        all_range_values = {}
        
        for feature  in self.numeric_features: 
            minVal = min(self.features[feature])
            maxVal = max(self.features[feature])            
            all_range_values[feature] = (minVal, maxVal)            
        
        return all_range_values
    
    def get_min_max_values(self):
        all_range_values = {"min": [], "max": []}
        
        for feature  in self.numeric_features: 
            minVal = min(self.features[feature])
            maxVal = max(self.features[feature])            
            all_range_values["min"].append(minVal)
            all_range_values["max"].append(maxVal)            
        
        return min(all_range_values["min"]), max(all_range_values["max"])
    
    def gen_class_ids(self):
        # Generating class ids
        self.classes = self.targets[self.target_col].unique()
        self.num_classes = len(self.classes)

        if isinstance(self.classes[0], str):
            label_encoder = LabelEncoder()
            self.labels = label_encoder.fit_transform(self.targets[self.target_col])
        else:
            self.labels_id = self.classes        
            self.labels = self.targets[self.target_col]

class BaseDataset(data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index: int):        
        return self.features[index], self.labels[index]
    