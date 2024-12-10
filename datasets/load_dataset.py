import json
from pathlib import Path
from collections import namedtuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


Dataset = namedtuple("Dataset", "data, feature_names, target, target_names")

def read_keel_dat(file_path):
    """
    Reads a KEEL .dat file and returns a Pandas DataFrame.

    Args:
        file_path (str): Path to the .dat file.

    Returns:
        pd.DataFrame: DataFrame containing the dataset.
    """
    try:
        with open(f"datasets/{file_path}/{file_path}.dat", 'r') as file:
            lines = file.readlines()

        # Extract metadata lines (those starting with '@')
        metadata_lines = [line.strip() for line in lines if line.startswith('@')]

        # Extract attribute names from metadata
        columns = []
        for line in metadata_lines:
            if line.startswith('@attribute'):
                # The attribute name is the second element in the split string
                columns.append(line.split()[1])

        # Extract data lines (those not starting with '@')
        data_lines = [line.strip() for line in lines if not line.startswith('@') and line.strip()]

        # Split the data lines by commas
        rows = [line.split(',') for line in data_lines]

        # Ensure the number of columns matches the data
        if len(columns) == 0 or len(rows[0]) != len(columns):
            columns = [f'feature_{i}' for i in range(len(rows[0]))]

        # Create DataFrame
        df = pd.DataFrame(rows, columns=columns)
        
        # """ CUIDADO
        # Convert data types where possible
        for col in df.columns:
            # Convert columns that can be interpreted as numeric
            # df[col] = pd.to_numeric(df[col])
            #  FutureWarning: errors='ignore' is deprecated and will raise in a future version. 
            # Use to_numeric without passing `errors` and catch exceptions explicitly instead
            # It breaks with the class column that is often a string
            df[col] = pd.to_numeric(df[col], errors='ignore')

        # Encode categorical labels if they exist
        for col in df.columns[:-1]:
            if df[col].dtype == 'object':  # Check if column is of type object (usually strings)
                df[col] = df[col].astype('category').cat.codes
        #"""

        # Asier
        X = df.iloc[:,:-1] 
        y = df.iloc[:,-1] 
        feature_names = X.columns.to_list()
        target_names = y.to_numpy()
        data = X.to_numpy()
        # target = target_names   # -> categories
        le = LabelEncoder()
        le.fit(target_names)
        target = le.transform(target_names)
        target_names = le.inverse_transform(np.unique(target))
        # target_names = np.unique(target)  # -> categories
        return Dataset(data, feature_names, target, target_names) 

        # return df

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error


def get_partition(dataset, partitionFile):
    with open(partitionFile, "r") as f:
        partition_data = json.load(f)

    clients = []
    for client in partition_data:
        Xi_train = dataset.data[partition_data[client]["train"]]
        yi_train = dataset.target[partition_data[client]["train"]]
        Xi_test = dataset.data[partition_data[client]["test"]]
        yi_test = dataset.target[partition_data[client]["test"]]
        clients.append([Xi_train, Xi_test, yi_train, yi_test])

    cls_counts = [np.unique(np.concatenate((c[3],c[2])),return_counts=True)[1] for c in clients]

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

    return clients, cls_counts, one_client


def load_json_dataset(jfileName,dataset=None):
    print(jfileName)
    with open(jfileName, "r") as f:
        partition_data = json.load(f)
    return partition_data

def read_json_dataset(dataset_name, dataset_path):
    data = read_keel_dat(dataset_name)
    print(dataset_path)
    partitions = load_json_dataset(dataset_path)
    client_partitions = []
    for i, client in enumerate(partitions.values()):
        X_train = data.data[client["train"]]
        y_train = data.target[client["train"]]
        X_test = data.data[client["test"]]
        y_test = data.target[client["test"]]
        client_partitions.append([X_train, X_test, y_train, y_test])

    return client_partitions


def read_json_datasets(directory_name: str, selected_dataset: str = None):
    """
    Parse all JSON files in the given directory structure and generate clients data.
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
            client_partitions = {}
            try:
                partitions = load_json_dataset(json_file)
                for i, client in enumerate(partitions.values()):
                    X_train = data.data[client["train"]]
                    y_train = data.target[client["train"]]
                    X_test = data.data[client["test"]]
                    y_test = data.target[client["test"]]
                    client_partitions.append([X_train, X_test, y_train, y_test])
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON from {json_file}: {e}")

        return client_partitions


if __name__ == '__main__':
    dataset = "appendicitis"
    DataInfos = read_keel_dat(dataset)
    print(DataInfos)
    # Data = DataInfos.get_data()
    # totAtt = DataInfos.get_number_of_attributes()
