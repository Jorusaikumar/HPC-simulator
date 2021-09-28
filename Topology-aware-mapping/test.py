# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:09:16 2021

@author: saikumar
"""
import sys
import os
import pandas as pd
import numpy as np
import torch
import utils
import json
import torch.nn as nn
import matplotlib.pyplot as plt

from model import GraphSAGE
from aggregators import MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator
from main import NeuralNetwork


gcn_layers = [16,16,16]
input_dims = 56
#downstream_model = NeuralNetwork(gcn_layers[-1], [2])
downstream_model = nn.Linear(gcn_layers[-1],1)
agg_class = getattr(sys.modules[__name__], 'MeanPoolAggregator')
model = GraphSAGE(input_dims,gcn_layers,downstream_model,agg_class)
model.load_state_dict(torch.load('model0.pth'))
model.eval()

###Model Testing
test_files = os.listdir('D:/clsim/database/gcn_graphs/temp_graphs')
df = pd.read_csv('D:/clsim/database/gcn_graphs/kernel_threads_features.csv')
tot_loss = 0.0
tot_accuracy = 0.0
tot_speedup = 0.0
test_size=0
speedup_list=[]
for F in test_files:
    lines=open('D:/clsim/database/gcn_graphs/temp_graphs/'+F).readlines()
    n=int(lines[0])
    feats_graph=[]
    for kernel in lines[1:n+1]:
        kernel=kernel.split('\n')[0]
        feature_vector = df.loc[df['kernel']==kernel]
        feature_vector = feature_vector.values.tolist()[0][1:]
        feats_graph.append(feature_vector)
    input_feature = (torch.tensor(feats_graph,dtype=torch.float32)).view(1,n,-1)
    adj_matrix = torch.zeros((n,n),dtype=torch.int32)
    for edge in lines[n+1:]:
        u,v = edge.split()
        adj_matrix[int(u),int(v)] = 1
    adj_matrix = adj_matrix.view(1,n,n)    
    out = model(input_feature,adj_matrix)
    #print(out)
    predictions = out.round()
    labels = torch.tensor(utils.optimal_makespan('D:/clsim/database/gcn_graphs/temp_timings/'+F[:-6]+'.json'),dtype=torch.float32)
    labels = labels.view(1,-1,1)
    loss = utils.weighted_loss(out, labels, [F[:-6]+'.json'],'D:/clsim/database/gcn_graphs/temp_timings/')
    tot_loss+=loss.item()
    num_correct = torch.sum(predictions == labels).item()
    tot_accuracy+=num_correct/out.shape[1]
    test_size+=1
    
    predicted_string = utils.string_representation_of_labels(predictions[0,:,0])
    target_string = utils.string_representation_of_labels(labels[0,:,0])
    timings = open('D:/clsim/database/gcn_graphs/temp_timings/'+F[:-6]+'.json')
    timings = json.load(timings)
    speedup = timings[target_string]/timings[predicted_string]
    speedup_list.append(speedup)
    tot_speedup+=speedup
    worst_speedup = timings[target_string]/max(timings.values())
    mean_speedup = [timings[target_string]/value for key,value in timings.items()]
    mean_speedup = sum(mean_speedup)/len(mean_speedup)
    
    print('{}: predicted speedup={}  ,worst_speedup={}, mean_speedup={}'.format(F,speedup,worst_speedup,mean_speedup))
    
tot_loss/=test_size
tot_accuracy/=test_size
tot_speedup/=test_size
print("Test loss: ",tot_loss," Test accuracy: ",tot_accuracy,"Avg speedup: ",tot_speedup)
bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
count,bin_edges = np.histogram(speedup_list,bins=bins)
print("count:",count)
histog = plt.hist(speedup_list,bins = bins)
plt.xlabel('speedups')
plt.ylabel('number of DAGs')
for i in range(len(count)):
    plt.text(histog[1][i],histog[0][i],'{0:.1f}'.format(100*count[i]/sum(count))+'%')
plt.show()