import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from torchwnn.datasets.adult import Adult
from torchwnn.classifiers import Wisard
from torchwnn.encoding import Thermometer

import pandas as pd

# Use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

dataset = Adult()

X_categorical = dataset.features[dataset.categorical_features]
X_numeric = dataset.features[dataset.numeric_features]

X_categorical = pd.get_dummies(X_categorical, columns=dataset.categorical_features, dtype=pd.Int8Dtype())
X = pd.concat([X_numeric, X_categorical], axis=1)

X = torch.tensor(X.values).to(device)
y = dataset.labels
y = torch.tensor(y).squeeze().to(device)


bits_encoding = 20
encoding = Thermometer(bits_encoding).fit(X)    
X_bin = encoding.binarize(X).flatten(start_dim=1)

X_train, X_test, y_train, y_test = train_test_split(X_bin, y, test_size=0.3, random_state = 0)  

entry_size = X_train.shape[1]
tuple_size = 8
model = Wisard(entry_size, dataset.num_classes, tuple_size)

with torch.no_grad():
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)  
    acc = accuracy_score(predictions, y_test)
    print("Wisard: Accuracy = ", acc)

