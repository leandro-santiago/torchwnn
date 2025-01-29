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

from pandas import read_csv
from torchwnn.datasets.dataset import Dataset

class Adult(Dataset):
    name = "adult"
    id = 2
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country' ]
    # education-num is removed because it is corresponding to education feature
    numeric_features = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

    def __init__(self, path = None):
        if not path:
            # Loading dataset from uci repo
            self.load_uci_repo()  
        else:
            names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']
            self.target_col = "class"
            data = read_csv(path, names=names, delimiter=" ")
            self.features = data[names[:-1]]
            self.targets = data[[self.target_col]]            
        
         # Correcting classes values ['<=50K' '>50K' '<=50K.' '>50K.']                        
        df = self.targets
        self.targets.loc[df[self.target_col]  == "<=50K.", self.target_col] = "<=50K"
        self.targets.loc[df[self.target_col]  == ">50K.", self.target_col] = ">50K"
        
        
        self.gen_class_ids()

         # Removing education-num features as it is the same that education feature
        self.features = self.features.drop(columns = ["education-num"])
        self.num_features = self.num_features - 1

        # Fill NaN values
        nan_str = "unknown"
        self.features["workclass"] = self.features["workclass"].fillna(nan_str)
        self.features["occupation"] = self.features["occupation"].fillna(nan_str)
        self.features["native-country"] = self.features["native-country"].fillna(nan_str)
   
            
   
