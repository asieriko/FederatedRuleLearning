# https://www.kaggle.com/code/fabriziobonavita/pytorch-mlp-classification
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from local_mlp import LocalAgent


class FeedForwardNN(nn.Module):
    def __init__(self, input_dim,hidden,out):
        super(FeedForwardNN,self).__init__()
        self.relu=nn.ReLU()
        self.layer1 = nn.Linear(input_dim, hidden)
        self.layer2 = nn.Linear(hidden,hidden)
        self.layer3 = nn.Linear(hidden,out)
        
    def forward(self,x): 
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x= self.relu(self.layer3(x))
        return x


class GlobalAgent():

    def __init__(self, X, y, model=FeedForwardNN):
        scaler = StandardScaler()
        self.X = scaler.fit_transform(X)
        self.y = y

        self.mlp = model(input_dim = self.X.shape[1],hidden = 25,out=3)
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=0.001)
        self.loss_fn   = nn.CrossEntropyLoss()

        self.epochs = 1000
        self.n_clients = 3

        self.clients = []

        self.init_clients()
        
    def init_clients(self):
        for i in range(self.n_clients):
            Xi = self.X
            yi = self.y
            self.clients.append(LocalAgent(Xi, yi))

    def fit(self):
        self.update_client_params()
            

    def running_model_avg(self, current, next, scale):
        if current == None:
            current = next
            for key in current:
                current[key] = current[key] * scale
        else:
            for key in current:
                current[key] = current[key] + (next[key] * scale)
        return current    

    def update_client_params(self):
        running_avg = None
        num_clients_per_round = len(self.clients)
        for client in self.clients:
            client.fit()
            cp = client.get_params()
            running_avg = self.running_model_avg(running_avg, cp, 1/num_clients_per_round)
        
        for client in self.clients:
            client.set_params(running_avg)


if __name__ == "__main__":

    dataset = '/data/asier/Ikerketa/Projects/FederatedRuleLearning/FedAvg/iris.csv'
    df = pd.read_csv(dataset)
    X = df[["sepal.length","sepal.width","petal.length","petal.width"]]

    le = LabelEncoder()
    df['label'] = le.fit_transform(df.variety.values)
    y = df['label'].values

    ga = GlobalAgent(X,y)
    ga.fit()