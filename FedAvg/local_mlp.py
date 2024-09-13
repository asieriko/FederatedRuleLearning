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
from sklearn.metrics import accuracy_score


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


class LocalAgent():

    def __init__(self, X, y, model=FeedForwardNN):
        X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.3)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        self.X_train = Variable(torch.from_numpy(X_train)).float()
        self.Y_train = torch.LongTensor(Y_train)
        self.X_test  = Variable(torch.from_numpy(X_test)).float()
        self.Y_test  = torch.LongTensor(Y_test)


        self.mlp = model(input_dim = self.X_train.shape[1],hidden = 25,out=3)
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=0.001)
        self.loss_fn   = nn.CrossEntropyLoss()

        self.epochs = 1000
        self.train_losses =np.zeros(self.epochs)
        self.test_losses = np.zeros(self.epochs)

    def get_params(self):
        return self.mlp.state_dict()

    def set_params(self, state_dict):
        self.mlp.load_state_dict(state_dict)

    def fit(self):
        for epoch in range(self.epochs):
            Y_pred = self.mlp(self.X_train)
            loss_train = self.loss_fn(Y_pred,self.Y_train)
            self.optimizer.zero_grad()
            loss_train.backward()
            self.optimizer.step()
            
            Y_pred_test = self.mlp(self.X_test)
            loss_test = self.loss_fn(Y_pred_test,self.Y_test)
            accuracy = accuracy_score(torch.argmax(Y_pred_test,dim=1),self.Y_test)
            self.train_losses[epoch] = loss_train.item()
            self.test_losses[epoch] = loss_test.item()

            
            if (epoch + 1) % 25 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {loss_train.item():.3f}, Test Loss: {loss_test.item():.3f}, Accuracy: {accuracy:.3f}")


if __name__ == "__main__":

    dataset = '/data/asier/Ikerketa/Projects/FederatedRuleLearning/FedAvg/iris.csv'
    df = pd.read_csv(dataset)
    X = df[["sepal.length","sepal.width","petal.length","petal.width"]]

    le = LabelEncoder()
    df['label'] = le.fit_transform(df.variety.values)
    y = df['label'].values

    la = LocalAgent(X,y)
    la.fit()