# Torchwnn: *Weightless Neural Network*

Torchwnn is a Python library for *Weightless Neural Network* (also known as *RAM-based* and *N-tuple based Neural Network* ).

# Usage
## Installation

First, install PyTorch using their [installation instructions](https://pytorch.org/get-started/locally/). Then, use the following command to install Torchwnn:


```bash
pip install torchwnn
```

Requirements: PyTorch and ucimlrepo to load datasets from UCI repository.


## Quick Start

### Iris Example

To quickly get started with Torchwnn, here's an example using the Iris dataset. Full training code is available in the [examples/iris.py](examples/iris.py) file.

```python
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from torchwnn.datasets.iris import Iris
from torchwnn.classifiers import Wisard
from torchwnn.encoding import Thermometer

# Use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

iris = Iris()
X = iris.features
X = torch.tensor(X.values).to(device)
y = list(iris.labels)
y = torch.tensor(y).squeeze().to(device)

bits_encoding = 20
encoding = Thermometer(bits_encoding).fit(X)    
X_bin = encoding.binarize(X).flatten(start_dim=1)

X_train, X_test, y_train, y_test = train_test_split(X_bin, y, test_size=0.3, random_state = 0)  

entry_size = X_train.shape[1]
tuple_size = 8
model = Wisard(entry_size, iris.num_classes, tuple_size)

with torch.no_grad():
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)  
    acc = accuracy_score(predictions, y_test)
    print("Wisard: Accuracy = ", acc)

```

## Supported WNN models
Currently, the library supports the following WNN models:

- WiSARD - Neurons based on dictionary.
