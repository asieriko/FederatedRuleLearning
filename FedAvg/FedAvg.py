import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from local_mlp import LocalAgent
from global_mlp import GlobalAgent

dataset = '/data/asier/Ikerketa/Projects/FederatedRuleLearning/FedAvg/iris.csv'
df = pd.read_csv(dataset)
X = df[["sepal.length","sepal.width","petal.length","petal.width"]]

le = LabelEncoder()
df['label'] = le.fit_transform(df.variety.values)
y = df['label'].values

ga = GlobalAgent(X,y)
ga.fit()

