from typing import Dict
import numpy as np
from ucimlrepo import fetch_ucirepo

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
            self.labels_id = {k:v for k, v in zip(self.classes, range(len(self.classes)))}
            for index, row in self.targets.iterrows():  
                self.targets.at[index, self.target_col] = self.labels_id[row[self.target_col]]                  
            
            self.labels = self.targets[self.target_col]
        else:
            self.labels_id = self.classes
            self.labels = self.targets[self.target_col]
