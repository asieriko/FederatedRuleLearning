import pandas as pd
import numpy as np
from random import shuffle, seed
from collections import namedtuple
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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
    return Dataset(data, feature_names, target, target_names) 
    #{"data": data, "feature_names": feature_names, "target": target, "target_names": target_names}   

# def generate_local_agents_old(dataset, n_clients, random_seed=1):
#     clients = []
#     X = dataset.data
#     y = dataset.target
#     idx = list(range(len(y)))
#     seed(random_seed)
#     shuffle(idx)
#     n = len(y) // n_clients
#     for i in range(n_clients):
#         last = i * n + n -1
#         if last >= len(y):
#             last = len(y)-1
#         X_i = pd.DataFrame(X[idx[i*n:last]], columns=dataset.feature_names)
#         y_i = y[idx[i * n:last]]
#         clients.append([X_i, y_i])
#     return clients

def generate_local_agents(dataset, n_clients=3, test_size=0.25, partition="homo",alpha=0.5, min_partition_ratio=0.5, random_seed=1):
    """
    n_clients:
        how many partitions
    test_size:
        proportion of the data to construct the test dataset
    partition:
        "homo" homegenous
        "hetero-dir" hetereogeneous
    min_partition_ratio:
         # Only for hetero-dir 1 -> n/n_clients, and 0.x fraction of that
    returns:
    clients: listz
    """
    label_names = {i: name for i,name in enumerate(dataset.target_names)}
    X, y, dataidx_map, cls_counts = partition_data(dataset,partition,n_clients,min_partition_ratio, alpha, random_seed)
    clients = []
    for idx_map in dataidx_map.values():
        Xi = pd.DataFrame(X[idx_map], columns=dataset.feature_names)
        yi = y[idx_map]
        Xi_train, Xi_test, yi_train, yi_test = train_test_split(Xi, yi, test_size=test_size, random_state=random_seed)
        clients.append([Xi_train, Xi_test, yi_train, yi_test])


    X_train = clients[0][0]
    X_test = clients[0][1]
    y_train = clients[0][2]
    y_test = clients[0][3]
    for cd in clients[1:]:
        X_train = pd.concat((X_train, cd[0]))
        X_test = pd.concat((X_test, cd[1]))
        y_train = np.hstack((y_train, cd[2]))
        y_test = np.hstack((y_test, cd[3]))
    one_client = [X_train, X_test, y_train, y_test]

    return clients, cls_counts, one_client

def dataset_xtrem(dataset):
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target   # target_names str for the ints
    clients = []
    for i in np.unique(y):
        X_i = X[y==i]
        y_i = y[y == i]
        clients.append([X_i, y_i])
    return clients

def load_dataset(datasetName):
    if datasetName == "iris":
        return datasets.load_iris()
    elif datasetName == "bcancer":
        return datasets.load_breast_cancer()
    elif datasetName == "digits":
        return datasets.load_digits()
    elif datasetName == "telescope":
        return load_magic_telescope()
    else:
        raise NameError

def partition_data(dataset, partition, n_clients, min_partition_ratio=0.5, alpha=0.5, random_seed=1):
    """
    dataset: Dataset
        dataset to be partitioned dataset.data for X and dataset.target for y
    partition : str
        homo or hetero-dir
    num_clients : int
        The total number of partitions that the data will be divided into.
    alpha : int, float > 0c
        Concentration parameter to the Dirichlet distribution
    min_partition_ratio: float
         # Only for hetero-dir 1 -> n/n_clients, and 0.x fraction of that
        The minimum number of samples that each partitions will have.
        If 1, it makes equal size partitions n/n_clients
    """
    # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py#L116
    X = dataset.data
    y = dataset.target
    
    n_train = X.shape[0]

    min_partition_size = min_partition_ratio * n_train/n_clients

    if min_partition_size == 0:
        min_partition_size = int(len(X)/n_clients)

    np.random.seed(42)
    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_clients)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_clients)}

    elif partition == "hetero-dir":
        min_size = 0
        K = len(np.unique(y))  # len(np.unique(y) # for names instead of numbers
        N = y.shape[0]
        net_dataidx_map = {}
        while min_size < min_partition_size:
            idx_batch = [[] for _ in range(n_clients)]
            for k in range(K): # for k in K:   # for names instead of numbers
                idx_k = np.where(y == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp


    return (X, y, net_dataidx_map, net_cls_counts)        


def plot_data_distribution(net_cls_counts, label_names=None, fileName=None):
    plt.style.use("tableau-colorblind10")
    plt.style.use("seaborn-v0_8-colorblind")
    
    # It can happen that some client doesn't have
    # all the classes
    values = set()
    for k1 in net_cls_counts.keys():
        for k2 in net_cls_counts[k1].keys():
            values.add(k2)

    n = {}
    for key in net_cls_counts:
        for ci in values:  
            if ci not in net_cls_counts[key]:
                net_cls_counts[key][ci] = 0
        n[key] = [v for k, v in sorted(net_cls_counts[key].items())]

    c = []
    v = []             
    for key, val in n.items():
        c.append(key)
        v.append(val)
    v = np.array(v)

    if not label_names:
        n_labels = max([len(cls_c) for cls_c in net_cls_counts.values()])
        label_names = {i:f"C{i+1}" for i in range(n_labels)}  # C for class
    
    fig=plt.figure()
    plt.bar(range(len(c)), v[:,0], label=label_names[0])
    bottom = v[:,0]
    for i in range(1,len(v[0])):
        plt.bar(range(len(c)), v[:,i], bottom=bottom, label=label_names[i])
        bottom = bottom + v[:,i]
    plt.xticks(range(len(c)), [f"Client {ci+1}" for ci in c])
    plt.title("Data distribution per client")
    plt.xlabel("Clients")
    plt.ylabel("Label amount")
    plt.legend()
    if fileName:
        plt.savefig(f"{fileName}.png")
    else:
        plt.show()

# To store the information...

def save_partiton(fileName, X, y, dataidx_map, cls_counts):
    all_clients = None
    for i in range(len(dataidx_map)):
        Xi = X[dataidx_map[i]]
        yi = y[dataidx_map[i]]
        Xy = np.column_stack((np.full_like(yi, i), Xi, yi))
        if all_clients is None:
            all_clients = Xy
        else:
            all_clients = np.concatenate((all_clients, Xy))
        cls_counts[i]
    
    # FIXME: Different file names!
    np.savetxt(fileName, all_clients, delimiter=';', header=f'Client;{";".join(list(f"Attr.{i}" for i in range(len(X[0]))))};Class')
    # with open(fileName, 'w') as f:
    #     f.write(str(cls_counts))

def read_partition(fileName):
    data = np.loadtxt(fileName, skiprows=1,delimiter=";")
    clients = np.unique(data[:,0])
    Xa = data[:,1:-1]
    ya = data[:,-1]
    dataidx_map = {}
    for ci in clients:
        Xi = data[data[:,0]==ci,1:-1]
        yi = data[data[:,0]==ci,-1]
        dataidx_map[int(ci)] = np.where(data[:,0]==ci)[0]
        
    cls_counts = {}  
    for net_i, dataidx in dataidx_map.items():
        unq, unq_cnt = np.unique(ya[dataidx], return_counts=True)
        tmp = {int(unq[i]): unq_cnt[i] for i in range(len(unq))}
        cls_counts[net_i] = tmp
    
    return Xa, ya, dataidx_map, cls_counts

if __name__ == "__main__"    :
    data = datasets.load_iris()
    label_names = {i: name for i,name in enumerate(data.target_names)}
    X, y, dataidx_map, cls_counts = partition_data(data,"homo",5)
    plot_data_distribution(cls_counts, label_names)    
    print(cls_counts)
    save_partiton("test.csv", X, y, dataidx_map, cls_counts)
    X, y, dataidx_map, cls_counts = read_partition("test.csv")
    print(cls_counts)
    X, y, dataidx_map, cls_counts = partition_data(data,"hetero-dir",5)
    plot_data_distribution(cls_counts)    
    X, y, dataidx_map, cls_counts = partition_data(data,"hetero-dir",5, min_partition_size=len(data.data)/5)
    plot_data_distribution(cls_counts)    



def reading(fileName):
    with open(fileName, 'r') as f:
        cls_counts = eval(f.read())
    return cls_counts
