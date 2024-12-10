import pandas as pd
import numpy as np
import json
from random import shuffle, seed
from pathlib import Path
from collections import defaultdict
from collections import namedtuple
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datasets.load_dataset import read_keel_dat, load_json_dataset

Dataset = namedtuple("Dataset", "data, feature_names, target, target_names")

# Attribute Skew
# Ideas:
# - Proportional stratified clustering
# - Constraint K-Means
# - Clusters los mas proporcional poisible respecto a las clases
# NIPS 2017 fair k-means
# Skewness coefficient
# Correlacion entre los datos de distintos clientes: Se pued comparar con
# Kolmogorov-Smirnoff o Kramer- comparar distribuciones)
# Autocorrelacion
# Mutual information
# https://arxiv.org/pdf/2102.02079
# https://github.com/algo-hhu/fair-kmeans
# https://github.com/imtiazziko/Variational-Fair-Clustering

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

def generate_one_client(clients):
    X_train = clients[0][0]
    X_test = clients[0][1]
    y_train = clients[0][2]
    y_test = clients[0][3]
    for cd in clients[1:]:
        X_train = np.concatenate((X_train, cd[0]))
        X_test = np.concatenate((X_test, cd[1]))
        y_train = np.hstack((y_train, cd[2]))
        y_test = np.hstack((y_test, cd[3]))
    one_client = [X_train, X_test, y_train, y_test]

    return one_client

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


def split_dataset_dirichlet_balanced(labels, n_clients, alpha=0.5, min_per_label=2, seed=None):
    """
    Splits dataset indices for federated learning clients using Dirichlet distribution,
    ensuring each client gets at least `min_per_label` samples per label and balanced total examples.
    
    Parameters:
    - labels: Array-like, labels of the dataset
    - n_clients: int, number of clients
    - alpha: float, Dirichlet distribution concentration parameter
    - min_per_label: int, minimum number of each label in each client
    - seed: int, random seed for reproducibility
    
    Returns:
    - clients_indices: list of lists, each sublist contains indices assigned to one client
    """
    if seed is not None:
        np.random.seed(seed)
        # random.seed(seed)
    
    # Dictionary to store indices of each label
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)
    
    # Shuffle indices of each label group
    for label in label_to_indices:
        np.random.shuffle(label_to_indices[label])
    
    # Initialize list to store client-specific indices
    clients_indices = [[] for _ in range(n_clients)]
    
    # Distribute minimum samples for each label to each client
    for label, indices in label_to_indices.items():
        # Check for sufficient samples to meet minimum requirement
        if len(indices) < min_per_label * n_clients:
            raise ValueError(f"Not enough samples for label {label} to meet min_per_label requirement.")
        
        # Minimum distribution
        start_idx = 0
        for client in range(n_clients):
            clients_indices[client].extend(indices[start_idx:start_idx + min_per_label])
            start_idx += min_per_label
        
        # Remaining samples after minimum distribution
        remaining_indices = indices[start_idx:]
        
        # Sample proportions from Dirichlet distribution for remaining samples
        proportions = np.random.dirichlet([alpha] * n_clients)
        samples_per_client = (proportions * len(remaining_indices)).astype(int)
        
        # Adjust to balance client sizes
        while sum(samples_per_client) < len(remaining_indices):
            samples_per_client[np.argmin(samples_per_client)] += 1
        while sum(samples_per_client) > len(remaining_indices):
            samples_per_client[np.argmax(samples_per_client)] -= 1
        
        # Distribute remaining samples according to adjusted proportions
        start_idx = 0
        for client, num_samples in enumerate(samples_per_client):
            clients_indices[client].extend(remaining_indices[start_idx:start_idx + num_samples])
            start_idx += num_samples

    # Balance total number of samples across clients
    target_size = len(labels) // n_clients
    client_sizes = [len(client) for client in clients_indices]
    for i in range(n_clients):
        while client_sizes[i] > target_size + 1:
            idx_to_move = clients_indices[i].pop()
            smallest_client = np.argmin(client_sizes)
            clients_indices[smallest_client].append(idx_to_move)
            client_sizes = [len(client) for client in clients_indices]

    return clients_indices


def partition_data_from_indices(dataset, indices):
    targets = dataset.target
    inputs = dataset.data
    partitions = []
    for idx in indices:#.values():
        partition = {
            "target": targets[idx],
            "data": inputs[idx],
            "feature_names": dataset.feature_names,
            "target_names": dataset.target_names
        }
        partitions.append(partition)

    return partitions


def generate_n_splits(dataset, n_clients, n_splits, alpha=0.5, min_samples_per_class=2, random_seed=1):
    """
    dataset: Dataset
        dataset to be partitioned dataset.data for X and dataset.target for y
    num_clients : int
        The total number of partitions that the data will be divided into.
    n_splits:
        number of different partitions to generate
    alpha : int, float > 0c
        Concentration parameter to the Dirichlet distribution
        alpha = 100 -> uniform
    min_samples_per_class: int
         # Only for heterogeneous distributions
        At least 2 to have them in train and test
        The minimum number of samples that each partitions will have.
    """
    targets = dataset.target
    inputs = dataset.data

    n_idx = []
    n_cls_counts = []
    done = 0
    random_seed = random_seed
    while done < n_splits:
    
        client_splits_idx = split_dataset_dirichlet_balanced(dataset.target, n_clients, alpha, min_samples_per_class, random_seed)
        random_seed += 1
        
        partitions = []
        for idx in client_splits_idx:
            partition = {
                "target": targets[idx],
                "data": inputs[idx],
                "feature_names": dataset.feature_names,
                "target_names": dataset.target_names
            }
            partitions.append(partition)

        net_cls_counts = []
        for client in partitions:
            net_cls_counts.append(np.unique(client["target"], return_counts=True)[1])

        if (len([len(x) for x in net_cls_counts if len(x)!=len(np.unique(targets))]) > 0) or (len([min(x) for x in net_cls_counts if min(x) < min_samples_per_class]) > 0):
            continue
        
        done += 1
        n_idx.append(client_splits_idx)
        n_cls_counts.append(net_cls_counts)

    return n_idx, n_cls_counts


def partition_data(dataset, n_clients, alpha=0.5, min_samples_per_class=2, random_seed=1):
    """
    dataset: Dataset
        dataset to be partitioned dataset.data for X and dataset.target for y
    num_clients : int
        The total number of partitions that the data will be divided into.
    alpha : int, float > 0c
        Concentration parameter to the Dirichlet distribution
        alpha = 100 -> uniform
    min_samples_per_class: int
         # Only for heterogeneous distributions
        At least 2 to have them in train and test
        The minimum number of samples that each partitions will have.
    """
    targets = dataset.target
    inputs = dataset.data
    
    client_splits_idx = split_dataset_dirichlet_balanced(dataset.target, n_clients, alpha, min_samples_per_class, random_seed)
    
    partitions = []
    for idx in client_splits_idx:
        partition = {
            "target": targets[idx],
            "data": inputs[idx],
            "feature_names": dataset.feature_names,
            "target_names": dataset.target_names
        }
        partitions.append(partition)

    net_cls_counts = []
    for client in partitions:
        net_cls_counts.append(np.unique(client["target"], return_counts=True)[1])

    return partitions, net_cls_counts

  


def plot_data_distribution_cgpt(data, fileName=None):
    num_rows, num_columns = data.shape
    x = np.arange(num_rows)

    fig, ax = plt.subplots()
    bottom = np.zeros(num_rows)
    plt.style.use("seaborn-v0_8-colorblind")
    if num_columns < 11:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  
    else:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, num_columns))
    
    for col in range(num_columns):
        ax.bar(x, data[:, col], bottom=bottom, color=colors[col], label=f'Category {col + 1}')
        bottom += data[:, col]  # Update the bottom position for the next stack

    ax.set_xlabel('Clients')
    ax.set_ylabel('Examples')
    ax.legend(title="Classes")

    plt.show()
    if fileName:
        plt.savefig(f"{fileName}.png")
    else:
        plt.show()

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
        plt.bar(range(len(c)), v[:,i], bottom=bottom, label=str(int(label_names[i])+1))
        bottom = bottom + v[:,i]
    plt.xticks(range(len(c)), [f"C{ci+1}" for ci in c])
    plt.title("Data distribution per client")
    plt.xlabel("Clients")
    plt.ylabel("Label amount")
    plt.legend()
    if fileName:
        plt.savefig(f"{fileName}.png")
        plt.close()
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


def generate_n_partitions(dataset, n_clients, min_samples, alpha, random_seed):
    # partition_data(dataset, n_clients, alpha=0.5, min_samples_per_class=2, random_seed=1)
    done = 0
    partitions = []
    rs = random_seed
    # for i in range(10):
    while done < 10:
        rs += 1
        partition, net_cls_counts = partition_data(dataset, n_clients, alpha=0.5, min_samples_per_class=min_samples, random_seed=rs)
        if (len([len(x) for x in net_cls_counts if len(x)!=len(np.unique(dataset.target))]) > 0) or (len([min(x) for x in net_cls_counts if min(x) < min_samples]) > 0):
            continue
        done += 1
        partitions.append(partition)
    return partitions


def generate_n_datasets(dataset, n_idx, fileName, test_size=0.25, random_seed=1):
    for j,partition_idx in enumerate(n_idx):
        partitions = {}
        for i, client_partition_idx in enumerate(partition_idx):
            train_idx, test_idx = train_test_split(client_partition_idx, test_size=test_size, random_state=random_seed)
            partitions[f"Client {i}"] = {"train":  train_idx, "test": test_idx}
        with open(f"{fileName}_f{j}.json","w") as f:
            json.dump(partitions, f, indent=4)
            
def generate_partitions_one_dataset(folder, datasetName, n_clients, alphas, min_samples=2, n_folds=5,random_seed=1):
    """
    folder where the datasets are stored
    dataset: name of the dataset
    n_clients: List with the number of clients to generate
    alphas: List wiht the values of alphas to generate
    n_folds: number of folds for each combination
    """
    main_directory = Path(folder) / datasetName
    try:
        dat_file = main_directory / f"{datasetName}.dat"
        if dat_file.is_file():
            for n_client in n_clients:
                for alpha in alphas:
                    dataset = read_keel_dat(datasetName)
                    # print(ds)
                    n_idx, n_cls = generate_n_splits(dataset, n_client, n_folds, alpha, min_samples_per_class=min_samples, random_seed=random_seed)
                    generate_n_datasets(dataset, n_idx, main_directory / f"{datasetName}_n{n_client}_a{alpha}",random_seed=random_seed)
    except Exception as inst:
        print(f"### Error: {dataset} ###")
        print(type(inst))  # the exception type
        print(inst.args)  # arguments stored in .args
        print(inst)
        # with open("errors.txt", "a") as err_file:
        #     err_file.write(f"Error: {datasetName=}: {type(inst)} -> {inst.args} = {inst}\n")

def generate_partitions_all_datasets(folder, n_clients, alphas, min_samples=2, n_folds=5,random_seed=1):
    """
    folder where the datasets are stored
    n_clients: List with the number of clients to generate
    alphas: List wiht the values of alphas to generate
    n_folds: number of folds for each combination
    """
    main_directory = Path(folder)
    for subdir in main_directory.iterdir():
        if subdir.name in ["crx","saheart","post-operative","monk-2"]: # Done
            continue
        if subdir.name in ["newthyroid", "optdigits"]: # Some problem
            continue
        if subdir.is_dir():
            print(f"{subdir.name}")
            generate_partitions_one_dataset(folder, subdir.name, n_clients, alphas, min_samples,random_seed)

def generate_partitions_all_datasets_OLD(folder, n_clients, alphas, n_folds=5,random_seed=1):
    """
    folder where the datasets are stored
    n_clients: List with the number of clients to generate
    alphas: List wiht the values of alphas to generate
    n_folds: number of folds for each combination
    """
    main_directory = Path(folder)
    for subdir in main_directory.iterdir():
        if subdir.name in ["crx","saheart","post-operative","monk-2"]: # Done
            continue
        if subdir.name in ["newthyroid", "optdigits"]: # Some problem
            continue
        try:
            if subdir.is_dir():
                dat_file = subdir / f"{subdir.name}.dat"
                if dat_file.is_file():
                    print(f"{subdir.name}")
                    for n_client in n_clients:
                        for alpha in alphas:
                            dataset = read_keel_dat(subdir.name)
                            # print(ds)
                            n_idx, n_cls = generate_n_splits(dataset, n_client, n_folds, alpha, min_samples_per_class=min_samples, random_seed=random_seed)
                            generate_n_datasets(dataset, n_idx, subdir / f"{subdir.name}_n{n_client}_a{alpha}",random_seed=random_seed)
        except Exception as inst:
            print(f"### Error: {subdir.name} ###")
            print(type(inst))  # the exception type
            print(inst.args)  # arguments stored in .args
            print(inst)
            # with open("errors.txt", "a") as err_file:
            #     err_file.write(f"Error: {datasetName=}: {type(inst)} -> {inst.args} = {inst}\n")
def generate_plots(directory_name: str, selected_dataset: str = None):
    """
    Parse all JSON files in the given directory structure and generate plots.
    If a specific dataset is provided, it will only parse files from that dataset.

    :param directory: The root directory containing the datasets.
    :param selected_dataset: The name of the dataset to parse (optional).
    """
    directory = Path(directory_name)
    for dataset_dir in directory.iterdir():
        # Check if we are in a dataset directory
        current_dataset = dataset_dir.name

        if current_dataset not in ["crx", "saheart", "post-operative", "monk-2"]:  # Done remove not...
            continue
        if current_dataset in ["newthyroid"]:  # Some problem
            continue
        if not dataset_dir.is_dir():
            continue

        # Skip if a dataset is selected and this is not the selected one
        if selected_dataset and current_dataset != selected_dataset:
            continue

        data = read_keel_dat(current_dataset)

        for json_file in dataset_dir.glob('*.json'):
            net_cls_counts = {}
            try:
                partitions = load_json_dataset(json_file)
                for i, client in enumerate(partitions.values()):
                    # labels, reps = np.unique(,return_counts=True)
                    labels, reps = np.unique(data.target[client["train"]+client["test"]], return_counts=True)
                    net_cls_counts[i] = {}
                    for l, r in zip(labels, reps):
                        net_cls_counts[i][l] = r
                plot_data_distribution(net_cls_counts, range(len(net_cls_counts[0])), str(json_file).replace(".json",""))
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON from {json_file}: {e}")

if __name__ == "__main__"    :
    data = datasets.load_iris()

    # Example usage
    dataset = data
    n_clients = 10
    alpha = 100
    n_splits = 5  # Number of different partitions (folds)
    min_samples = 2  # Ensure at least 2 samples per class in each client

    generate_partitions_one_dataset("datasets", "wine" , n_clients=[5,10], alphas=[0.5,100], min_samples=min_samples, n_folds=5, random_seed=1)
    # generate_partitions_all_datasets("datasets", n_clients=[5,10],alphas=[0.5,100], min_samples=min_samples)
    # generate_plots("datasets")

    # partitions = generate_n_partitions(dataset, n_clients, min_samples, alpha, random_seed=1)
    # n_idx, n_cls = generate_n_splits(dataset, n_clients, n_splits, alpha=0.5, min_samples_per_class=min_samples, random_seed=1)
    # generate_n_datasets(dataset, n_idx, f"_n{n_clients}_a{alpha}")
    # load_json_dataset(f"iris_n{n_clients}_a{alpha}_f1.json")
    #
    # print(n)

#     label_names = {i: name for i,name in enumerate(data.target_names)}
#     X, y, dataidx_map, cls_counts = partition_data(data,"homo",5)
#     plot_data_distribution(cls_counts, label_names)    
#     print(cls_counts)
#     save_partiton("test.csv", X, y, dataidx_map, cls_counts)
#     X, y, dataidx_map, cls_counts = read_partition("test.csv")
#     print(cls_counts)
#     X, y, dataidx_map, cls_counts = partition_data(data,"hetero-dir",5)
#     plot_data_distribution(cls_counts)    
#     X, y, dataidx_map, cls_counts = partition_data(data,"hetero-dir",5, min_partition_ratio=len(data.data)/5)
#     plot_data_distribution(cls_counts)    

    

# for cssi in css:
#     print(np.unique(cssi["target"], return_counts=True))


# def reading(fileName):
#     with open(fileName, 'r') as f:
#         cls_counts = eval(f.read())
#     return cls_counts
