# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 19:46:32 2021

@author: saikumar
"""

import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader, Dataset

import utils

class DAGData(Dataset):
    def __init__(self,path=None):
        super(DAGData, self).__init__()
        self.path = path
        self.features=[]
        self.adj=[]
        self.labels=[]
        self.timings=[]
        df = pd.read_csv(path+'/kernel_threads_features.csv')
        graphs = os.listdir(path+'/diverse_graphs')
        random.shuffle(graphs)
        '''graphs=[]
        for i in range(10):
            for j in range(0,160,10):
                graphs.append('graph_'+str(i+j)+'.graph')
        print(len(graphs))'''
        for F in graphs:
            lines=open(path+'/diverse_graphs/'+F).readlines()
            n=int(lines[0])
            feats_graph=[]
            for kernel in lines[1:n+1]:
                kernel=kernel.split('\n')[0]
                feature_vector = df.loc[df['kernel']==kernel]
                feature_vector = feature_vector.values.tolist()[0][1:]
                feats_graph.append(feature_vector)
            self.features.append(torch.tensor(feats_graph,dtype=torch.float32))
            adj_matrix = torch.zeros((n,n),dtype=torch.int32)
            for edge in lines[n+1:]:
                u,v = edge.split()
                adj_matrix[int(u),int(v)] = 1
            self.adj.append(adj_matrix)
            
            
            devices = torch.tensor(utils.optimal_makespan(path+'/diverse_timings/'+F[:-6]+'.json'),dtype=torch.float32)
            devices = devices.view(-1,1)
            self.labels.append(devices)
            self.timings.append(F[:-6]+'.json')
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        return self.features[idx],self.adj[idx],self.labels[idx],self.timings[idx]
    
    def get_dims(self):
        return self.features[0].shape[1],self.labels[0].shape[1]