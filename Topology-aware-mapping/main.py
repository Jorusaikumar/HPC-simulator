# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 12:55:19 2021

@author: saikumar
"""
import os
import sys
import numpy as np
from math import ceil
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import xgboost as xgb

from load_data import DAGData
from aggregators import MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator
from model import GraphSAGE
import utils

class NeuralNetwork(nn.Module):
    def __init__(self,input_dim,hidden_layers):
        super(NeuralNetwork,self).__init__()
        self.layers=[nn.Linear(input_dim,hidden_layers[0])]
        self.layers.extend([nn.Linear(hidden_layers[i-1],hidden_layers[i]) for i in range(1,len(hidden_layers))])
        self.layers.append(nn.Linear(hidden_layers[-1],1))
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
    
    def forward(self,input):
        output=input
        for i in range(len(self.layers)):
            output = self.layers[i](output)
            if i<len(self.layers)-1:
                output = self.sigmoid(output)
        return output
    
class XGBoost(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(XGBoost,self).__init__()    
        self.model = xgb.XGBClassifier()
    def forward(self,input):
        output = self.model.predict(input.detach().numpy())
        return output

def main():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    batch_size=32
    dataset = DAGData(path = 'D:/clsim/database/gcn_graphs')
    loader = DataLoader(dataset=dataset, batch_size=batch_size)
    
    input_dims,output_dims = dataset.get_dims()
    gcn_layers = [16,16,16]
    downstream_model = nn.Linear(gcn_layers[-1],output_dims)
    agg_class = getattr(sys.modules[__name__], 'LSTMAggregator')
    
    model = GraphSAGE(input_dims,gcn_layers,downstream_model,agg_class,device)
    
    learning_rate = 0.01
    epochs = 100
    print_every = 1
    num_batches = int(ceil(len(dataset) / batch_size))
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        print('Epoch {} / {}'.format(epoch+1, epochs))
        running_loss = 0.0
        num_correct, num_examples = 0, 0
        for (idx, batch) in enumerate(loader):
            features,adj_matrix,labels,timings=batch
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(features,adj_matrix)
            loss = utils.weighted_loss(out,labels,timings,'D:/clsim/database/gcn_graphs/diverse_timings/')
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                    running_loss += loss.item()
                    predictions = out.round()
                    num_correct += torch.sum(predictions == labels).item()
                    num_examples += labels.shape[0]*labels.shape[1]*labels.shape[2]
            if (idx + 1) % print_every == 0:
                running_loss /= print_every
                accuracy = num_correct / num_examples
                print('    Batch {} / {}: loss {}, accuracy {}'.format(
                    idx+1, num_batches, running_loss, accuracy))
                running_loss = 0.0
                num_correct, num_examples = 0, 0
                
    torch.save(model.state_dict(),'model10.pth')
    
if __name__=='__main__':
    main()