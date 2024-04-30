from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from random import shuffle, seed
import sys, os
sys.path.append(os.path.abspath("."))
from FedRLS.FedRLS import FedRLS

def generate_local_agents(X, y, n_clients, random_seed=1):
    clients = []
    idx = list(range(len(y)))
    seed(random_seed)
    shuffle(idx)
    n = len(y) // n_clients
    for i in range(n_clients):
        last = i * n + n -1
        if last >= len(y):
            last = len(y)-1
        X_i = pd.DataFrame(X[idx[i*n:last]], columns=iris.feature_names)
        y_i = y[idx[i * n:last]]
        clients.append([X_i, y_i])

    return clients

def iris_xtrem():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target   # target_names str for the ints
    clients = []
    for i in np.unique(y):
        X_i = X[y==i]
        y_i = y[y == i]
        clients.append([X_i, y_i])
    return clients


iris = datasets.load_iris()
X = iris.data  # pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target  #   target_names str for the ints

clients_datasets = generate_local_agents(X, y, 3)
# print(clients_datasets)
# clients_datasets = iris_xtrem()
fedrls = FedRLS(clients_datasets)
fedrls.local()
