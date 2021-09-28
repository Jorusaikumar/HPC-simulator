# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 22:05:55 2021

@author: saikumar
"""
import os
import json
import math
import torch
import torch.nn as nn

def get_kernel_features(path):
    lines=open(path).readlines()
    index=lines.index('---\n')
    lines=lines[index+1:index+81]
    x=[]
    for item in lines:
        split=item.split()
        x.append(eval(split[-1]))
    return x

def get_item_size(item_type,item_size):
    if item_type in ['int','unsigned int','float']:
        return 4*item_size
    if item_type in ['long','unsigned long','double']:
        return 8*item_size
    if item_type in ['short','unsigned short']:
        return 2*item_size
    if item_type in ['long double']:
        return 16*item_size 

def get_thread_parameters(kernel):
    kernel_name = kernel[:kernel.rfind('_')]
    dataset=eval(kernel[kernel.rfind('_')+1:])
    info=json.load(open('D:/clsim/database/info/'+kernel_name+'.json'))
    globalWorkSize=eval(info['globalWorkSize'])
    if isinstance(globalWorkSize,list):
        globalWorkSize=sum(globalWorkSize)
    L=[globalWorkSize]
    L.append(info['workDimension'])
    for buffertype in ['inputBuffers','outputBuffers','ioBuffers','varArguments']:
        buffersize=0
        for item in info[buffertype]:
            item_type=item['type']
            if buffertype == 'varArguments':
                num_of_items=eval(item['value'])
            else:
                num_of_items=eval(item['size'])
            item_size=get_item_size(item_type,num_of_items)
            buffersize+=item_size
        L.append(buffersize)
    return L

def get_parents(adj_matrix):
    b,n,n = adj_matrix.shape
    parent_list = []
    for i in range(b):
        adj = adj_matrix[i,:,:]
        parents = [[] for j in range(n)]
        for v in range(n):
            for u in range(n):
                if adj[u,v]==1:
                    parents[v].append(u)
        parent_list.append(parents)
    
    return parent_list

def optimal_makespan(path):
    file = open(path)
    timings = json.load(file)
    optimal_makespan = math.inf
    optimal_mapping = ""
    for key,value in timings.items():
        if value<optimal_makespan:
            optimal_makespan = value
            optimal_mapping = key
    
    L=[]
    for char in optimal_mapping:
        L.append(eval(char))
    
    return L

def string_representation_of_labels(label_vec):
    s=""
    for i,x in enumerate(label_vec):
        s+=str(int(x.item()))
    return s

def weighted_loss(out,labels,timings,path):
    tot_sum = 0.0
    weight_sum = 0.0
    criterion = nn.BCELoss()
    for i in range(out.shape[0]):
        loss = criterion(out[i,:,:],labels[i,:,:])
        timing_file=open(path+timings[i])
        makespans = json.load(timing_file)
        out_key = string_representation_of_labels((out[i,:,0]).round())
        label_key = string_representation_of_labels((labels[i,:,0]))
        print(out_key,label_key)
        weight = makespans[out_key]/makespans[label_key]
        weight_sum+=weight
        tot_sum+=loss*weight
    return tot_sum/weight_sum