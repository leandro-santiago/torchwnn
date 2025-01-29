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

class Ecoli(Dataset):
    name = "ecoli"
    id = 39
    categorical_features = [] #'Sequence'
    numeric_features = ['mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2']

    def __init__(self, path = None):
        if not path:
            # Loading dataset from uci repo
            self.load_uci_repo()  
        else:
            names = ['Sequence', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
            self.target_col = "class"
            data = read_csv(path, names=names, delimiter=" ")
            self.features = data[names[:-1]]
            self.targets = data[[self.target_col]]            
        
        self.gen_class_ids()