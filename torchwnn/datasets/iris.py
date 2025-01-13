from pandas import read_csv
from torchwnn.datasets.dataset import Dataset

class Iris(Dataset):
    name = "iris"
    id = 53
    categorical_features = []
    numeric_features = ['sepal length', 'sepal width', 'petal length', 'petal width']

    def __init__(self, path = None):
        if not path:
            # Loading dataset from uci repo
            self.load_uci_repo()           
        else: 
            names = self.numeric_features + ["class"]
            self.target_col = "class"
            data = read_csv(path, names=names)
            self.features = data[self.numeric_features]
            self.targets = data[[self.target_col]]  
            self.num_features = len(self.numeric_features)
        
        self.gen_class_ids()
            
    
           
            