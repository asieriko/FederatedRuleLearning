from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from random import shuffle, seed
from collections import namedtuple
import sys, os
sys.path.append(os.path.abspath("."))
from FedRLS.FedRLS import FedRLS

Dataset = namedtuple("Dataset", "data, feature_names, target, target_names")

def load_magic_telescope(fpath="datasets/MagicTelescope.csv"):
    df = pd.read_csv(fpath, index_col=0)
    X = df.iloc[:,:-1] 
    y = df.iloc[:,-1] 
    feature_names = X.columns.to_list()
    target_names = y.to_list()
    data = X.to_numpy()
    le = LabelEncoder()
    le.fit(target_names)
    target = le.transform(target_names)
    return Dataset(data, feature_names, target, target_names) #{"data": data, "feature_names": feature_names, "target": target, "target_names": target_names}   

def generate_local_agents(dataset, n_clients, random_seed=1):
    clients = []
    X = dataset.data
    y = dataset.target
    idx = list(range(len(y)))
    seed(random_seed)
    shuffle(idx)
    n = len(y) // n_clients
    for i in range(n_clients):
        last = i * n + n -1
        if last >= len(y):
            last = len(y)-1
        X_i = pd.DataFrame(X[idx[i*n:last]], columns=dataset.feature_names)
        y_i = y[idx[i * n:last]]
        clients.append([X_i, y_i])
    return clients

def dataset_xtrem(dataset):
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target   # target_names str for the ints
    clients = []
    for i in np.unique(y):
        X_i = X[y==i]
        y_i = y[y == i]
        clients.append([X_i, y_i])
    return clients


iris = datasets.load_iris()
telescope = load_magic_telescope()
clients_datasets = generate_local_agents(iris, 3)
# print(clients_datasets)
# clients_datasets = dataset_xtrem(iris)
fedrls = FedRLS(clients_datasets)
fedrls.local()
