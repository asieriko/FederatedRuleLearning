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


if __name__ == '__main__':
    dataset = "appendicitis"
    DataInfos = read_keel_dat(dataset)
    print(DataInfos)
    # Data = DataInfos.get_data()
    # totAtt = DataInfos.get_number_of_attributes()
