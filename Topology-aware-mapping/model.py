# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 09:24:50 2021

@author: saikumar
"""

import numpy as np
import torch
import torch.nn as nn

from aggregators import MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator
import utils

class GraphSAGE(nn.Module):
    def __init__(self,input_dim,gcn_layers,downstream_model,agg_class=MeanAggregator,device='cpu',dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.input_dim=input_dim
        self.gcn_layers=gcn_layers
        self.num_layers=len(gcn_layers)
        self.agg_class = agg_class
        
        self.aggregators = nn.ModuleList([agg_class(input_dim, input_dim, device)])
        self.aggregators.extend([agg_class(dim, dim, device) for dim in gcn_layers[:-1]])
        
        c = 3 if agg_class == LSTMAggregator else 2
        self.fcs = nn.ModuleList([nn.Linear(c*input_dim, gcn_layers[0])])
        self.fcs.extend([nn.Linear(c*gcn_layers[i-1], gcn_layers[i]) for i in range(1, len(gcn_layers))])
        self.downstream_model=downstream_model
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,features,adj_matrix):
        batch_parents = utils.get_parents(adj_matrix)
        batch_size,n,_ = features.shape
        batch_output = torch.zeros((batch_size,n,1))
        for batch in range(batch_size):
            out = features[batch,:,:]
            parents = batch_parents[batch]
            for k in range(self.num_layers):
                curr_out = torch.zeros(n,self.gcn_layers[k])
                for i in range(n):
                    if(len(parents[i])>0):
                        aggregate = self.aggregators[k](out[parents[i],:])
                    elif self.agg_class==LSTMAggregator:
                        aggregate = torch.zeros(2*out.shape[1])
                    else:
                        aggregate = torch.zeros(out.shape[1])
                    input_vec=torch.cat((out[i,:],aggregate))
                    output_vec=self.sigmoid(self.fcs[k](input_vec))
                    output_vec=self.dropout(output_vec)
                    curr_out[i,:]=output_vec
                out = curr_out.div(curr_out.norm(dim=1, keepdim=True)+1e-6)
            label = self.sigmoid(self.downstream_model(out))
            batch_output[batch,:,:] = label
        return batch_output        