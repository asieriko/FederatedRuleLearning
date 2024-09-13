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
import ex_fuzzy.fuzzy_sets as fs
import ex_fuzzy.evolutionary_fit as GA
import ex_fuzzy.utils as utils
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

def non_federated(dataset,n_gen=30,n_pop=50,nRules=15,nAnts=4,fz_type_studied=fs.FUZZY_SETS.t1,tolerance=0.001,runner=1,ramdom_seed=23,**args):
    # ,**args: to avoid "unexpected keyword for parameters not needed in this function but which are compiled into model_parameters"
        n_gen = n_gen
        n_pop = n_pop

        class_names = dataset.target_names  # np.unique(dataset[1])
        X = dataset.data
        y = dataset.target
        precomputed_partitions = utils.construct_partitions(X, fz_type_studied)
        model = GA.BaseFuzzyRulesClassifier(
            nRules=nRules,
            nAnts=nAnts,
            fuzzy_type=fz_type_studied, 
            verbose=False,
            tolerance=tolerance, 
            linguistic_variables=precomputed_partitions,
            class_names=class_names,
            runner=runner)
        fl_classifier = model
        
        X_train = X
        y_train = y
        fl_classifier.fit(X_train, y_train, n_gen=n_gen, pop_size=n_pop, random_state = ramdom_seed)
        # str_rules = eval_tools.eval_fuzzy_model(fl_classifier, X_train, y_train, X_test, y_test,
        #                                         plot_rules=True, print_rules=True, plot_partitions=True,
        #                                         return_rules=True)

        performance = fl_classifier.performance
        print("-- Non Federated Case -- ")
        print(performance)
        print(fl_classifier.rule_base)
        print(f"LOG:,NonFederated,{performance}")

iris = datasets.load_iris()
bcancer = datasets.load_breast_cancer()
digits = datasets.load_digits()
telescope = load_magic_telescope()

dataset = bcancer

ramdom_seed = 23

model_parameters = {
    "n_gen":30,
    "n_pop":50,
    "nRules":15,
    "nAnts":4,
    "fz_type_studied":fs.FUZZY_SETS.t1,
    "tolerance":0.001,
    "runner":1,
    "ramdom_seed":23,
    "sim_threshold":0.7,
    "contradictory_factor":0.7
}


clients_datasets = generate_local_agents(dataset, 3)
non_federated(dataset,**model_parameters)
# print(clients_datasets)
# clients_datasets = dataset_xtrem(iris)
fedrls = FedRLS(clients_datasets, model_parameters)
fedrls.local()
