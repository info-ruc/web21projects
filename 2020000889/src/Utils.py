#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import re
import csv
import time
import random
import glob
import tqdm
import copy
import math
import pandas as pd
import numpy as np
import itertools
import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MSELoss
from torch.nn import BCELoss,BCEWithLogitsLoss
from torch.optim import Adam,Adadelta,RMSprop

from dgl.nn.pytorch import GATConv

from scipy.sparse import coo_matrix
from scipy.sparse import vstack
from scipy import sparse
from math import exp
from collections import Counter


# In[2]:


class BPRLoss(nn.Module):
    def __init__(self,scorer):
        super().__init__()
        self.scorer = scorer
        
    def forward(self,u, pos, neg):
        xui = self.scorer(u,pos)
        xuj = self.scorer(u,neg)
        return -F.logsigmoid(xui - xuj).sum()
    
class Wrap_Dataset(Dataset):
    """Wrapper, convert <user, pos_item, neg_item> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, pos_item_tensor, neg_item_tensor):
        self.user_tensor = user_tensor
        self.pos_item_tensor = pos_item_tensor
        self.neg_item_tensor = neg_item_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.pos_item_tensor[index], self.neg_item_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


# 定义早停止类
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation metric Increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt') # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


# In[ ]:




