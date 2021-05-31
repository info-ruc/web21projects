#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

from Utils import BPRLoss


# In[4]:


def call_hit_ratio(df_eval,top_k):
    top_k = df_eval[df_eval['rank']<=top_k]
    truth_in_top_k  = top_k[top_k['rating']==1]
    return len(truth_in_top_k) * 1.0 / (df_eval['uid'].nunique() * 5.0)

def call_ndcg(df_eval,top_k):
    top_k = df_eval[df_eval['rank']<=top_k]
    truth_in_top_k  = top_k[top_k['rating']==1]
    truth_in_top_k['ndcg'] = truth_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
    return truth_in_top_k['ndcg'].sum() * 1.0 / (df_eval['uid'].nunique()*sum([ math.log(2) / math.log(1 + i) for i in range(1,6)]))

def evaluate(evaluate_data,input_data, model,loss_func,top_k,use_cuda=True):
    """
    input_data --> 计算验证损失
    evaluate_data --> 计算指标
    
    """
    model.eval()
    with torch.no_grad():
        if use_cuda:
            eval_user = torch.LongTensor(evaluate_data['uid']).cuda()
            eval_item = torch.LongTensor(evaluate_data['iid']).cuda()
            eval_rating = torch.FloatTensor(evaluate_data['rating']).cuda()
            
            eval_user_input = torch.LongTensor(input_data['uid']).cuda()
            eval_pos = torch.LongTensor(input_data['pos_iid']).cuda()
            eval_neg = torch.LongTensor(input_data['neg_iid']).cuda()
        else:
            eval_user = torch.LongTensor(evaluate_data['uid']).cpu()
            eval_item = torch.LongTensor(evaluate_data['iid']).cpu()
            eval_rating = torch.FloatTensor(evaluate_data['rating']).cpu()
            
            eval_user_input = torch.LongTensor(input_data['uid']).cpu()
            eval_pos = torch.LongTensor(input_data['pos_iid']).cpu()
            eval_neg = torch.LongTensor(input_data['neg_iid']).cpu()
            
        #print(eval_user.shape,eval_item.shape)    
        scores = model(eval_user, eval_item)
        val_loss = loss_func(eval_user_input,eval_pos,eval_neg)
            
            
        #把数据转存到cpu,从而使用pandas
        eval_user = torch.LongTensor(evaluate_data['uid']).cpu()
        eval_item = torch.LongTensor(evaluate_data['iid']).cpu()
        eval_rating = torch.FloatTensor(evaluate_data['rating']).cpu()
        scores = scores.cpu()
        
        df_eval = pd.DataFrame([], columns=['uid', 'iid','rating', 'score'])
        df_eval['uid'] = eval_user.data.view(-1).tolist()
        df_eval['iid'] = eval_item.data.view(-1).tolist()
        df_eval['rating'] = eval_rating.data.view(-1).tolist()
        df_eval['score'] = scores.data.view(-1).tolist()
        
        df_eval['rank'] = df_eval.groupby('uid')['score'].rank(method='first', ascending=False)
        df_eval.sort_values(['uid', 'rank'], inplace=True)
        
        return val_loss.item(),call_hit_ratio(df_eval,top_k),call_ndcg(df_eval,top_k)


# In[ ]:




