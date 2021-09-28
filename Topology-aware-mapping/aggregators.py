# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 08:29:55 2021

@author: saikumar
"""

import math

import numpy as np
import torch
import torch.nn as nn

class Aggregator(nn.Module):

    def __init__(self, input_dim=None, output_dim=None, device='cpu'):
        super(Aggregator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

    def forward(self, features):

        n = len(features)
        if self.__class__.__name__ == 'LSTMAggregator':
            out = torch.zeros(n, 2*self.output_dim).to(self.device)
        else:
            out = torch.zeros(n, self.output_dim).to(self.device)
        out = self._aggregate(features)

        return out

    def _aggregate(self, features):
        raise NotImplementedError

class MeanAggregator(Aggregator):

    def _aggregate(self, features):
        return torch.mean(features, dim=0)

class PoolAggregator(Aggregator):

    def __init__(self, input_dim, output_dim, device='cpu'):
        super(PoolAggregator, self).__init__(input_dim, output_dim, device)

        self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def _aggregate(self, features):
        out = self.sigmoid(self.fc1(features))
        return self._pool_fn(out)

    def _pool_fn(self, features):
        raise NotImplementedError

class MaxPoolAggregator(PoolAggregator):

    def _pool_fn(self, features):
        return torch.max(features, dim=0)[0]

class MeanPoolAggregator(PoolAggregator):

    def _pool_fn(self, features):
        return torch.mean(features, dim=0)

class LSTMAggregator(Aggregator):

    def __init__(self, input_dim, output_dim, device='cpu'):
        # super(LSTMAggregator, self).__init__(input_dim, output_dim, device)
        super().__init__(input_dim, output_dim, device)

        self.lstm = nn.LSTM(input_dim, output_dim, bidirectional=True, batch_first=True)

    def _aggregate(self, features):
        perm = np.random.permutation(np.arange(features.shape[0]))
        features = features[perm, :]
        features = features.unsqueeze(0)

        out, _ = self.lstm(features)
        out = out.squeeze(0)
        out = torch.sum(out, dim=0)

        return out