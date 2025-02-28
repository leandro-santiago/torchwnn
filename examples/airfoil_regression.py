import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torchwnn.regressors import RegressionWisard
from torchwnn.datasets.airfoil import Airfoil
from torchwnn.encoding import Thermometer

# Use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

dataset = Airfoil()

X = dataset.features
X = torch.tensor(X.values).to(device)
y = dataset.labels
y = torch.tensor(y).squeeze().to(device)

bits_encoding = 20
encoding = Thermometer(bits_encoding).fit(X)    
X_bin = encoding.binarize(X).flatten(start_dim=1)

X_train, X_test, y_train, y_test = train_test_split(X_bin, y, test_size=0.3, random_state = 0)  

entry_size = X_train.shape[1]
tuple_size = 10
model = RegressionWisard(entry_size, tuple_size)
#model = RegressionWisard(entry_size, tuple_size, minOne=1, minZero=1)

with torch.no_grad():
    model.fit(X_train,y_train)
    
    predictions = model.predict(X_test)  
    mse = mean_squared_error(predictions, y_test)
    print("Wisard simple mean: MSE = ", mse)

    predictions = model.predict(X_test, centrality="powermean")  
    mse = mean_squared_error(predictions, y_test)
    print("Wisard power mean: MSE = ", mse)

    predictions = model.predict(X_test, centrality="median")  
    mse = mean_squared_error(predictions, y_test)
    print("Wisard median: MSE = ", mse)

    predictions = model.predict(X_test, centrality="harmonicmean")  
    mse = mean_squared_error(predictions, y_test)
    print("Wisard harmonic mean: MSE = ", mse)
    
    predictions = model.predict(X_test, centrality="harmonicpowermean")  
    mse = mean_squared_error(predictions, y_test)
    print("Wisard harmonic power mean: MSE = ", mse)

    predictions = model.predict(X_test, centrality="geometricmean")  
    mse = mean_squared_error(predictions, y_test)
    print("Wisard geometric mean: MSE = ", mse)

    predictions = model.predict(X_test, centrality="exponentialmean")  
    mse = mean_squared_error(predictions, y_test)
    print("Wisard exponential mean: MSE = ", mse)
    