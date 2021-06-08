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


# In[14]:


class ACM_Dataset(object):
    def __init__(self,config_dat):
        self.dataset_name = config_dat['dataset_name']
        #self.model_name = model_name
        self.Load_flag = config_dat['isLoad']
        self.Store_flag =config_dat['store']
        self.filepaths = config_dat['filepaths']
        self.uNum,self.iNum = self.ACM_get_nums()
        self.train,self.dev,self.test = self.data_split(config_dat['neg_num4train'],
                                                   config_dat['neg_num4eval'])
        self.train['rating'] = [1 for i in range(len(self.train))]
        
        self.input_train = self.ACM_build_input_data(1)
        self.input_dev = self.ACM_build_input_data(2)
        self.input_test = self.ACM_build_input_data(3)
        self.eval_dev,self.eval_test = self.ACM_build_eval_data()
        
        self.a2a,self.p2p = self.ACM_relation_process()
        self.mp_graphs = self.ACM_create_mp_neighbor_graph()
        
        self.u2i = [self.ACM_buildLaplacianMat_u2i(),[self.uNum,self.iNum]]
        self.u2es = [[self.ACM_buildLaplacianMat_u2u(),0,True]]
        self.i2es = [[self.ACM_buildLaplacianMat_i2i(),0,True]]

        
    def ACM_get_nums(self):
        df_a2p = pd.read_csv(self.filepaths[1])
        return df_a2p['aid'].max()+1,df_a2p['pid'].max()+1
    
    def ACM_split(self,df_a2p):
        """ 
        interact>=3 ---> [N-2:1:1]
        interact==2 ---> [1:0:1]
        interact==1 ---> [1:0:0]
        
        """
        rd_val = []
        for i in range(len(df_a2p)):
            rd_val.append(random.random())
        df_a2p['random_val']=rd_val
        df_a2p['rank'] = df_a2p.groupby(['aid'])['random_val'].rank(method='first', ascending=False)

        grouped = df_a2p.groupby(['aid'])
        test1 = pd.DataFrame([], columns=['aid', 'pid','rating','random_val','rank'])
        dev1 = pd.DataFrame([], columns=['aid', 'pid','rating','random_val','rank'])
        train1 = pd.DataFrame([], columns=['aid', 'pid','rating','random_val','rank'])
        for name,group in tqdm.tqdm(grouped):
            if(len(group)==1):
                train1 = train1.append(group)
            if(len(group)==2):
                train1 = train1.append(group[group['rank'] ==1])
                test1 = test1.append(group[group['rank'] ==2])
            if(len(group)>2):
                train1 = train1.append(group[group['rank'] >2])
                dev1= dev1.append(group[group['rank']==1])
                test1 = test1.append(group[group['rank'] ==2])

        return train1[['aid', 'pid','rating']],dev1[['aid', 'pid','rating']], test1[['aid', 'pid','rating']]

    def get_prob(self,neg_items,counter_dict):
        """ get the prob of each neg_items"""
        neg_occurance = [counter_dict[i] for i in neg_items] 
        neg_num = sum(neg_occurance)
        neg_prob = [i/neg_num for i in neg_occurance]
        return neg_prob
    
    def ACM_get_negative_items(self,df_a2p):
        paper_pool = set(df_a2p['pid'].unique())
        counter_dict = Counter(df_a2p['pid'])
        interact_status = df_a2p.groupby('aid')['pid'].apply(set).reset_index().rename(columns={'pid': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: paper_pool - x)
        interact_status['negative_probs'] = interact_status['negative_items'].apply(lambda x:self.get_prob(x,counter_dict))
        return interact_status[['aid', 'negative_items', 'negative_probs']]

    def ACM_negative_sample(self,df,neg_num,neg):
        """依照文章出现的频率来采负样本【热门物品优先】"""
        full = pd.merge(df, neg, on=['aid'],how='left')
        full['negatives'] = full.apply(lambda row: np.random.choice(list(row['negative_items']),neg_num,replace=False,
                                                                  p = row['negative_probs']),axis=1)
        return full[['aid','pid','negatives']]            
    
    def ACM_split_and_sample(self,store, neg_num4train, neg_num4eval):
        df_a2p = pd.read_csv(self.filepaths[1])        
        #split
        train,dev,test = self.ACM_split(df_a2p)
        #sample
        neg = self.ACM_get_negative_items(df_a2p)
        train_w_neg = self.ACM_negative_sample(train,neg_num4train,neg)
        test_w_neg = self.ACM_negative_sample(test,neg_num4eval,neg)
        dev_w_neg = self.ACM_negative_sample(dev,neg_num4eval,neg)
        
        if store:
            train_w_neg.to_csv(self.filepaths[0]+'train1.csv')
            test_w_neg.to_csv(self.filepaths[0]+'test1.csv')
            dev_w_neg.to_csv(self.filepaths[0]+'dev1.csv')
        return train_w_neg,dev_w_neg,test_w_neg
    
    def data_split(self, neg_num4train, neg_num4eval):
        if self.Load_flag==True:
            #raise Exception("Dataset has been loaded.")
            train = pd.read_csv(self.filepaths[0]+'train.csv')
            dev = pd.read_csv(self.filepaths[0]+'dev.csv')
            test = pd.read_csv(self.filepaths[0]+'test.csv')
        else:
            train,dev,test = self.ACM_split_and_sample(self.Store_flag, neg_num4train, neg_num4eval)
        print('Dataset has been splited!')
        return train,dev,test
    ###############################################################################################################################
    def ACM_build_input_data(self,dat_type):
        authors, pos_papers, neg_papers = [], [], []
        input_dat= pd.DataFrame()
        
        if self.Load_flag==True:
            if dat_type==1:
                for row in self.train.itertuples():
                    for i in row.negatives[1:-2].split():
                        authors.append(int(row.aid))
                        pos_papers.append(int(row.pid))
                        neg_papers.append(int(i))
            if dat_type==2:
                for row in self.dev.itertuples():
                    for i in row.negatives[1:-2].split():
                        authors.append(int(row.aid))
                        pos_papers.append(int(row.pid))
                        neg_papers.append(int(i))
            if dat_type==3:
                for row in self.test.itertuples():
                    for i in row.negatives[1:-2].split():
                        authors.append(int(row.aid))
                        pos_papers.append(int(row.pid))
                        neg_papers.append(int(i))
        else:
            if dat_type==1:
                for row in self.train.itertuples():
                    for i in row.negatives:
                        authors.append(int(row.aid))
                        pos_papers.append(int(row.pid))
                        neg_papers.append(int(i))
            if dat_type==2:
                for row in self.dev.itertuples():
                    for i in row.negatives:
                        authors.append(int(row.aid))
                        pos_papers.append(int(row.pid))
                        neg_papers.append(int(i))
            if dat_type==3:
                for row in self.test.itertuples():
                    for i in row.negatives:
                        authors.append(int(row.aid))
                        pos_papers.append(int(row.pid))
                        neg_papers.append(int(i))
                        
        input_dat['uid']=authors
        input_dat['pos_iid']=pos_papers
        input_dat['neg_iid']=neg_papers
        return input_dat
    
    def ACM_build_eval_data(self):
        authors,papers,ratings = [],[],[]
        eval_dev=pd.DataFrame()
        if self.Load_flag==True:
            for row in self.dev.itertuples():
                for i in row.negatives[1:-2].split():
                    authors.append(int(row.aid))
                    papers.append(int(i))
                    ratings.append(float(0))  # negative samples get 0 rating
                authors.append(int(row.aid))
                papers.append(int(row.pid))
                ratings.append(float(1))
        else:
            for row in self.dev.itertuples():
                for i in row.negatives:
                    authors.append(int(row.aid))
                    papers.append(int(i))
                    ratings.append(float(0))  # negative samples get 0 rating
                authors.append(int(row.aid))
                papers.append(int(row.pid))
                ratings.append(float(1))
        eval_dev['uid']=authors
        eval_dev['iid']=papers
        eval_dev['rating']=ratings
        
        authors,papers,ratings = [],[],[]
        eval_test=pd.DataFrame()
        if self.Load_flag==True:
            for row in self.test.itertuples():
                for i in row.negatives[1:-2].split():
                    authors.append(int(row.aid))
                    papers.append(int(i))
                    ratings.append(float(0))  # negative samples get 0 rating
                authors.append(int(row.aid))
                papers.append(int(row.pid))
                ratings.append(float(1))
        else:
            for row in self.test.itertuples():
                for i in row.negatives:
                    authors.append(int(row.aid))
                    papers.append(int(i))
                    ratings.append(float(0))  # negative samples get 0 rating
                authors.append(int(row.aid))
                papers.append(int(row.pid))
                ratings.append(float(1))
        eval_test['uid']=authors
        eval_test['iid']=papers
        eval_test['rating']=ratings
        
        return eval_dev,eval_test
    ###############################################################################################################################
    
    def ACM_relation_process(self):
        """
        load relation data
        if relation is self interaction ---> drop self link
        
        """
        # load data
        df_a2a = pd.read_csv(self.filepaths[0]+'a2a.csv')
        df_p2p = pd.read_csv(self.filepaths[0]+'p2p.csv')
        
        # self link drop
        df_a2a = df_a2a[df_a2a['aid']!=df_a2a['aid_2']]
        df_p2p = df_p2p[df_p2p['pid']!=df_p2p['pid_2']]
        
        # add link flag
        df_a2a['link']=[1 for i in range(len(df_a2a))]
        df_p2p['link']=[1 for i in range(len(df_p2p))]
        return df_a2a,df_p2p
    
    ###############################################################################################################################
    def ACM_sample_mp_neighbor(self,df):
        sample_df = pd.DataFrame([],columns = ['aid','pid'])
        grouped_df = df.groupby('aid')
        for name,group in tqdm.tqdm(grouped_df):
            if len(group)> 100:
                    sample_df = sample_df.append(group.sample(n=100, replace=False))
            if 100>=len(group)>10:
                    sample_df = sample_df.append(group.sample(frac=0.1, replace=False))
            if len(group)<=10:
                    sample_df = sample_df.append(group)
        return sample_df
                    
    def ACM_create_mp_neighbor_graph(self):
        """
        creat metapath neighbor: APP, AAP
        and transform to DGL Graph
        """
        
        if self.Load_flag==True:
            df_app = pd.read_csv(self.filepaths[0]+'app.csv')
            df_aap = pd.read_csv(self.filepaths[0]+'aap.csv')
        else:
            df_app = pd.merge(self.train, self.p2p, on=['pid'])
            
            df_aap = pd.merge(self.a2a, self.train, on=['aid'])

            df_aap = df_aap[['aid','pid']].drop_duplicates().reset_index(drop=True)
            df_app = df_app[['aid','pid']].drop_duplicates().reset_index(drop=True)

            #sample (sample standard? sample strategy?)
            
            if len(df_app)>200000:
                print('length of app:',len(df_app),'start sampling...') 
                df_app = self.ACM_sample_mp_neighbor(df_app)

            if len(df_aap)>200000:
                print('length of aap:',len(df_aap),'start sampling...') 
                df_aap = self.ACM_sample_mp_neighbor(df_aap)
                
            if self.Store_flag==True:
                df_app.to_csv('ACM/app.csv')
                df_aap.to_csv('ACM/aap.csv')

        df_app['n_pid'] = df_app['pid'].apply(lambda x: x+self.uNum)        
        df_aap['n_pid'] = df_aap['pid'].apply(lambda x: x+self.uNum)
        self.train['n_pid'] = self.train['pid'].apply(lambda x: x+self.uNum)
        
        g_ap = dgl.DGLGraph()
        g_ap.add_nodes(self.uNum+self.iNum)
        g_ap.add_edges(self.train['aid'].tolist(),self.train['n_pid'].tolist())
        g_ap.add_edges(self.train['n_pid'].tolist(),self.train['aid'].tolist())

        g_app = dgl.DGLGraph()
        g_app.add_nodes(self.uNum+self.iNum)
        g_app.add_edges(df_app['aid'].tolist(),df_app['n_pid'].tolist())
        g_app.add_edges(df_app['n_pid'].tolist(),df_app['aid'].tolist())

        g_aap = dgl.DGLGraph()
        g_aap.add_nodes(self.uNum+self.iNum)
        g_aap.add_edges(df_aap['aid'].tolist(),df_aap['n_pid'].tolist())
        g_aap.add_edges(df_aap['n_pid'].tolist(),df_aap['aid'].tolist())
        
        return [g_ap,g_app,g_aap]
    
    ###############################################################################################################################
    
    def ACM_buildLaplacianMat_u2i(self):

        rt_item = self.train['pid'] + self.uNum
        uiMat = coo_matrix((self.train['rating'], (self.train['aid'], self.train['pid'])))

        uiMat_upperPart = coo_matrix((self.train['rating'], (self.train['aid'], rt_item)))
        uiMat = uiMat.transpose()
        uiMat.resize((self.iNum, self.uNum + self.iNum))

        A = sparse.vstack([uiMat_upperPart,uiMat])
        selfLoop = sparse.eye(self.uNum+self.iNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        diag[np.isinf(diag)] = 0
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i,data)
        return SparseL
    
    def ACM_buildLaplacianMat_u2u(self):
        uuMat = coo_matrix((self.a2a['link'], (self.a2a['aid'], self.a2a['aid_2'])))

        # item的最后一号恰好没有交互，所以这里要补全
        #iiMat.resize((self.iNum, self.uNum + self.iNum))
        
        #uiMat_upperPart = coo_matrix((rt['rating'], (rt['aid'], rt_item)))
        #uiMat = uiMat.transpose()
        #uiMat.resize((self.itemNum, self.userNum + self.itemNum))
        
        A = uuMat
        selfLoop = sparse.eye(self.uNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        diag[np.isinf(diag)] = 0
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        data = L.data
        
        # item的最后一号恰好没有交互，所以这里要补全
        #row = np.append(row,self.uNum + self.iNum-1)
        #col = np.append(col,self.uNum + self.iNum-1)
        #data = np.append(data,0)
        
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(data)
        SparseL = torch.sparse.FloatTensor(i,data)
        
        return SparseL
    
    
    def ACM_buildLaplacianMat_i2i(self):
        iiMat = coo_matrix((self.p2p['link'], (self.p2p['pid'], self.p2p['pid_2'])))

        # item的最后一号恰好没有交互，所以这里要补全
        iiMat.resize((self.iNum, self.iNum))
        
        #uiMat_upperPart = coo_matrix((rt['rating'], (rt['aid'], rt_item)))
        #uiMat = uiMat.transpose()
        #uiMat.resize((self.itemNum, self.userNum + self.itemNum))
        
        A = iiMat
        selfLoop = sparse.eye(self.iNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        diag[np.isinf(diag)] = 0
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        data = L.data
        
        # item的最后一号恰好没有交互，所以这里要补全
        row = np.append(row,self.iNum-1)
        col = np.append(col,self.iNum-1)
        data = np.append(data,0)
        
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(data)
        SparseL = torch.sparse.FloatTensor(i,data)
        
        return SparseL


# In[4]:


class Amazon_Dataset(object):
    def __init__(self,config_dat):
        self.dataset_name = config_dat['dataset_name']
        #self.model_name = model_name
        self.Load_flag = config_dat['isLoad']
        self.Store_flag =config_dat['store']
        self.filepaths = config_dat['filepaths']
        self.uNum,self.iNum = self.Amazon_get_nums()
        self.train,self.dev,self.test = self.data_split(config_dat['neg_num4train'],
                                                   config_dat['neg_num4eval'])
        self.train['rating'] = [1 for i in range(len(self.train))]
        
        self.input_train = self.Amazon_build_input_data(1)
        self.input_dev = self.Amazon_build_input_data(2)
        self.input_test = self.Amazon_build_input_data(3)
        self.eval_dev,self.eval_test = self.Amazon_build_eval_data()
        
        self.i2b = pd.read_csv(self.filepaths[0]+'i2b.csv')
        self.i2v = pd.read_csv(self.filepaths[0]+'i2v.csv')
        self.i2c = pd.read_csv(self.filepaths[0]+'i2c.csv')
        self.i2b['link'] = [1 for _ in range(len(self.i2b))]
        self.i2v['link'] = [1 for _ in range(len(self.i2v))]
        self.i2c['link'] = [1 for _ in range(len(self.i2c))]
        self.bNum = self.i2b['bid'].max()+1
        self.vNum = self.i2v['vid'].max()+1
        self.cNum = self.i2c['cid'].max()+1
        #self.a2a,self.p2p = self.ACM_relation_process()
        self.mp_graphs = self.Amazon_create_mp_neighbor_graph()
        
        self.u2i = [self.Amazon_buildLaplacianMat_u2i(),[self.uNum,self.iNum]]
        self.u2es = []
        self.i2es = [
            [self.Amazon_buildLaplacianMat_i2b(),self.bNum,False],
            [self.Amazon_buildLaplacianMat_i2v(),self.vNum,False],
            [self.Amazon_buildLaplacianMat_i2c(),self.cNum,False]
        ]
        
    def Amazon_get_nums(self):
        df_u2i = pd.read_csv(self.filepaths[1])
        return df_u2i['uid'].max()+1,df_u2i['iid'].max()+1
    
    def Amazon_split(self,df_u2i):
        """ 
        interact>10 ---> [n-10:5:5]
        10>=interact>5 ---> [n-5:0:5]
        5>=interact ---> [n:0:0]
        
        """
        rd_val = []
        for i in range(len(df_u2i)):
            rd_val.append(random.random())
        df_u2i['random_val']=rd_val
        df_u2i['rank'] = df_u2i.groupby(['uid'])['random_val'].rank(method='first', ascending=False)

        grouped = df_u2i.groupby(['uid'])
        test1 = pd.DataFrame([], columns=['uid', 'iid','rating','random_val','rank'])
        dev1 = pd.DataFrame([], columns=['uid', 'iid','rating','random_val','rank'])
        train1 = pd.DataFrame([], columns=['uid', 'iid','rating','random_val','rank'])
        for name,group in tqdm.tqdm(grouped):
            if(5>=len(group)):
                train1 = train1.append(group)
            if(10>=len(group)>5):
                train1 = train1.append(group[group['rank'] >5])
                test1 = test1.append(group[group['rank'] <=5])
            if(len(group)>10):
                train1 = train1.append(group[group['rank'] >10])
                dev1= dev1.append(group[(group['rank']>5) & (group['rank']<=10)])
                test1 = test1.append(group[group['rank'] <=5])

        return train1[['uid', 'iid','rating']],dev1[['uid', 'iid','rating']], test1[['uid', 'iid','rating']]
    
    def get_prob(self,neg_items,counter_dict):
        """ get the prob of each neg_items"""
        neg_occurance = [counter_dict[i] for i in neg_items] 
        neg_num = sum(neg_occurance)
        neg_prob = [i/neg_num for i in neg_occurance]
        return neg_prob
    
    def Amazon_get_negative_items(self,df_u2i):
        paper_pool = set(df_u2i['iid'].unique())
        counter_dict = Counter(df_u2i['iid'])
        interact_status = df_u2i.groupby('uid')['iid'].apply(set).reset_index().rename(columns={'iid': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: paper_pool - x)
        interact_status['negative_probs'] = interact_status['negative_items'].apply(lambda x:self.get_prob(x,counter_dict))
        return interact_status[['uid', 'negative_items', 'negative_probs']]
    
    def Amazon_negative_sample(self,df,neg_num,neg):
        """依照文章出现的频率来采负样本【热门物品优先】"""
        full = pd.merge(df, neg, on=['uid'],how='left')
        full['negatives'] = full.apply(lambda row: np.random.choice(list(row['negative_items']),neg_num,replace=False,
                                                                  p = row['negative_probs']),axis=1)
        return full[['uid','iid','negatives']] 
    
    def Amazon_split_and_sample(self, neg_num4train, neg_num4eval):
        df_u2i = pd.read_csv(self.filepaths[1])        
        #split
        train,dev,test = self.Amazon_split(df_u2i)
        #sample
        neg = self.Amazon_get_negative_items(df_u2i)
        train_w_neg = self.Amazon_negative_sample(train,neg_num4train,neg)
        test_w_neg = self.Amazon_negative_sample(test,neg_num4eval,neg)
        dev_w_neg = self.Amazon_negative_sample(dev,neg_num4eval,neg)
        
        if self.Store_flag:
            train_w_neg.to_csv(self.filepaths[0]+'train.csv')
            test_w_neg.to_csv(self.filepaths[0]+'test.csv')
            dev_w_neg.to_csv(self.filepaths[0]+'dev.csv')
        return train_w_neg,dev_w_neg,test_w_neg
    
    def data_split(self, neg_num4train, neg_num4eval):
        if self.Load_flag==True:
            #raise Exception("Dataset has been loaded.")
            train = pd.read_csv(self.filepaths[0]+'train.csv')
            dev = pd.read_csv(self.filepaths[0]+'dev.csv')
            test = pd.read_csv(self.filepaths[0]+'test.csv')
        else:
            train,dev,test = self.Amazon_split_and_sample(neg_num4train, neg_num4eval)
        print('Dataset has been splited!')
        return train,dev,test
    ###############################################################################################################################
    
    def Amazon_build_input_data(self,dat_type):
        users, pos_items, neg_items = [], [], []
        input_dat= pd.DataFrame()
        
        if self.Load_flag==True:
            if dat_type==1:
                for row in self.train.itertuples():
                    for i in row.negatives[1:-2].split():
                        users.append(int(row.uid))
                        pos_items.append(int(row.iid))
                        neg_items.append(int(i))
            if dat_type==2:
                for row in self.dev.itertuples():
                    for i in row.negatives[1:-2].split():
                        users.append(int(row.uid))
                        pos_items.append(int(row.iid))
                        neg_items.append(int(i))
            if dat_type==3:
                for row in self.test.itertuples():
                    for i in row.negatives[1:-2].split():
                        users.append(int(row.uid))
                        pos_items.append(int(row.iid))
                        neg_items.append(int(i))
        else:
            if dat_type==1:
                for row in self.train.itertuples():
                    for i in row.negatives:
                        users.append(int(row.uid))
                        pos_items.append(int(row.iid))
                        neg_items.append(int(i))
            if dat_type==2:
                for row in self.dev.itertuples():
                    for i in row.negatives:
                        users.append(int(row.uid))
                        pos_items.append(int(row.iid))
                        neg_items.append(int(i))
            if dat_type==3:
                for row in self.test.itertuples():
                    for i in row.negatives:
                        users.append(int(row.uid))
                        pos_items.append(int(row.iid))
                        neg_items.append(int(i))
                        
        input_dat['uid']=users
        input_dat['pos_iid']=pos_items
        input_dat['neg_iid']=neg_items
        return input_dat
    
    def Amazon_build_eval_data(self):
        users,items,ratings = [],[],[]
        eval_dev=pd.DataFrame()
        if self.Load_flag==True:
            for row in self.dev.itertuples():
                for i in row.negatives[1:-2].split():
                    users.append(int(row.uid))
                    items.append(int(i))
                    ratings.append(float(0))  # negative samples get 0 rating
                users.append(int(row.uid))
                items.append(int(row.iid))
                ratings.append(float(1))
        else:
            for row in self.dev.itertuples():
                for i in row.negatives:
                    users.append(int(row.uid))
                    items.append(int(i))
                    ratings.append(float(0))  # negative samples get 0 rating
                users.append(int(row.uid))
                items.append(int(row.iid))
                ratings.append(float(1))
        eval_dev['uid']=users
        eval_dev['iid']=items
        eval_dev['rating']=ratings
        
        users,items,ratings = [],[],[]
        eval_test=pd.DataFrame()
        if self.Load_flag==True:
            for row in self.test.itertuples():
                for i in row.negatives[1:-2].split():
                    users.append(int(row.uid))
                    items.append(int(i))
                    ratings.append(float(0))  # negative samples get 0 rating
                users.append(int(row.uid))
                items.append(int(row.iid))
                ratings.append(float(1))
        else:
            for row in self.test.itertuples():
                for i in row.negatives:
                    users.append(int(row.uid))
                    items.append(int(i))
                    ratings.append(float(0))  # negative samples get 0 rating
                users.append(int(row.uid))
                items.append(int(row.iid))
                ratings.append(float(1))
        eval_test['uid']=users
        eval_test['iid']=items
        eval_test['rating']=ratings
        
        return eval_dev,eval_test
    ###############################################################################################################################
    
    def Amazon_sample_mp_neighbor(self,df):
        sample_df = pd.DataFrame([],columns = ['uid','iid'])
        grouped_df = df.groupby('uid')
        for name,group in tqdm.tqdm(grouped_df):
            if len(group)> 50:
                    sample_df = sample_df.append(group.sample(n=50, replace=False))
            if 50>=len(group)>10:
                    sample_df = sample_df.append(group.sample(frac=0.2, replace=False))
            if len(group)<=10:
                    sample_df = sample_df.append(group)
        return sample_df
                    
    def Amazon_create_mp_neighbor_graph(self):
        """
        creat metapath neighbor: UI, UIBI,UIVI,UICI
        and transform to DGL Graph
        """
        
        if self.Load_flag==True:
            df_uibi = pd.read_csv(self.filepaths[0]+'uibi.csv')
            df_uivi = pd.read_csv(self.filepaths[0]+'uivi.csv')
            df_uici = pd.read_csv(self.filepaths[0]+'uici.csv')
        else:
            temp = pd.merge(self.train[['uid','iid']], self.i2b[['iid','bid']], on=['iid'])
            temp = temp.drop(columns=['iid']).drop_duplicates().reset_index(drop=True)
            df_uibi = pd.merge(temp,self.i2b[['iid','bid']],on=['bid'])
            
            temp = pd.merge(self.train[['uid','iid']], self.i2v[['iid','vid']], on=['iid'])
            temp = temp.drop(columns=['iid']).drop_duplicates().reset_index(drop=True)
            df_uivi = pd.merge(temp,self.i2v[['iid','vid']],on=['vid'])
            
            temp = pd.merge(self.train[['uid','iid']], self.i2c[['iid','cid']], on=['iid'])
            temp = temp.drop(columns=['iid']).drop_duplicates().reset_index(drop=True)
            df_uici = pd.merge(temp,self.i2c[['iid','cid']],on=['cid'])
        
            df_uibi = df_uibi[['uid','iid']].drop_duplicates().reset_index(drop=True)
            df_uivi = df_uivi[['uid','iid']].drop_duplicates().reset_index(drop=True)
            df_uici = df_uici[['uid','iid']].drop_duplicates().reset_index(drop=True)

            #sample (sample standard? sample strategy?)
            
            if len(df_uibi)>200000:
                print('length of df_uibi:',len(df_uibi),'start sampling...') 
                df_uibi = self.Amazon_sample_mp_neighbor(df_uibi)

            if len(df_uivi)>240000:
                print('length of df_uivi:',len(df_uivi),'start sampling...') 
                df_uivi = self.Amazon_sample_mp_neighbor(df_uivi)
                
            if len(df_uici)>200000:
                print('length of df_uici:',len(df_uici),'start sampling...') 
                df_uici = self.Amazon_sample_mp_neighbor(df_uici)
                
            if self.Store_flag==True:
                df_uibi.to_csv('Amazon/uibi.csv')
                df_uivi.to_csv('Amazon/uivi.csv')
                df_uici.to_csv('Amazon/uici.csv')

        df_uibi['n_iid'] = df_uibi['iid'].apply(lambda x: x+self.uNum)        
        df_uivi['n_iid'] = df_uivi['iid'].apply(lambda x: x+self.uNum)
        df_uici['n_iid'] = df_uici['iid'].apply(lambda x: x+self.uNum)
        self.train['n_iid'] = self.train['iid'].apply(lambda x: x+self.uNum)
        
        g_ui = dgl.DGLGraph()
        g_ui.add_nodes(self.uNum+self.iNum)
        g_ui.add_edges(self.train['uid'].tolist(),self.train['n_iid'].tolist())
        g_ui.add_edges(self.train['n_iid'].tolist(),self.train['uid'].tolist())
        
        g_uibi = dgl.DGLGraph()
        g_uibi.add_nodes(self.uNum+self.iNum)
        g_uibi.add_edges(df_uibi['uid'].tolist(),df_uibi['n_iid'].tolist())
        g_uibi.add_edges(df_uibi['n_iid'].tolist(),df_uibi['uid'].tolist())
        
        g_uivi = dgl.DGLGraph()
        g_uivi.add_nodes(self.uNum+self.iNum)
        g_uivi.add_edges(df_uivi['uid'].tolist(),df_uivi['n_iid'].tolist())
        g_uivi.add_edges(df_uivi['n_iid'].tolist(),df_uivi['uid'].tolist())
        
        g_uici = dgl.DGLGraph()
        g_uici.add_nodes(self.uNum+self.iNum)
        g_uici.add_edges(df_uici['uid'].tolist(),df_uici['n_iid'].tolist())
        g_uici.add_edges(df_uici['n_iid'].tolist(),df_uici['uid'].tolist())

        return [g_ui,g_uibi,g_uivi,g_uici]
    
    ###############################################################################################################################
    def Amazon_buildLaplacianMat_u2i(self):
        rt_item = self.train['iid'] + self.uNum
        uiMat = coo_matrix((self.train['rating'], (self.train['uid'], self.train['iid'])))

        uiMat_upperPart = coo_matrix((self.train['rating'], (self.train['uid'], rt_item)))
        uiMat_upperPart.resize((self.uNum, self.uNum + self.iNum))
        
        uiMat = uiMat.transpose()
        uiMat.resize((self.iNum, self.uNum + self.iNum))
        #print(uiMat_upperPart.shape,uiMat.shape)
        A = sparse.vstack([uiMat_upperPart,uiMat])
        selfLoop = sparse.eye(self.uNum+self.iNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        diag[np.isinf(diag)] = 0
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        #print(L.shape)
        row = L.row
        col = L.col
        data = L.data
        
        row =np.append(row,self.uNum + self.iNum-1)
        col =np.append(col,self.uNum + self.iNum-1)
        data =np.append(data,0)
        
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(data)
        SparseL = torch.sparse.FloatTensor(i,data)
        return SparseL
    
    def Amazon_buildLaplacianMat_i2b(self):
        rt_item = self.i2b['bid'] + self.iNum
        uiMat = coo_matrix((self.i2b['link'], (self.i2b['iid'], self.i2b['bid'])))

        uiMat_upperPart = coo_matrix((self.i2b['link'], (self.i2b['iid'], rt_item)))
        uiMat = uiMat.transpose()
        uiMat.resize((self.bNum, self.iNum + self.bNum))

        A = sparse.vstack([uiMat_upperPart,uiMat])
        selfLoop = sparse.eye(self.iNum + self.bNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        diag[np.isinf(diag)] = 0
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i,data)
        return SparseL
    
    def Amazon_buildLaplacianMat_i2v(self):
        rt_item = self.i2v['vid'] + self.iNum
        uiMat = coo_matrix((self.i2v['link'], (self.i2v['iid'], self.i2v['vid'])))

        uiMat_upperPart = coo_matrix((self.i2v['link'], (self.i2v['iid'], rt_item)))
        uiMat = uiMat.transpose()
        uiMat.resize((self.vNum, self.iNum + self.vNum))

        A = sparse.vstack([uiMat_upperPart,uiMat])
        selfLoop = sparse.eye(self.iNum + self.vNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        diag[np.isinf(diag)] = 0
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i,data)
        return SparseL
    
    def Amazon_buildLaplacianMat_i2c(self):
        rt_item = self.i2c['cid'] + self.iNum
        uiMat = coo_matrix((self.i2c['link'], (self.i2c['iid'], self.i2c['cid'])))

        uiMat_upperPart = coo_matrix((self.i2c['link'], (self.i2c['iid'], rt_item)))
        uiMat = uiMat.transpose()
        uiMat.resize((self.cNum, self.iNum + self.cNum))

        A = sparse.vstack([uiMat_upperPart,uiMat])
        selfLoop = sparse.eye(self.iNum + self.cNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        diag[np.isinf(diag)] = 0
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i,data)
        return SparseL


# In[5]:


class Movielens_Dataset(object):
    def __init__(self,config_dat):
        self.dataset_name = config_dat['dataset_name']
        #self.model_name = model_name
        self.Load_flag = config_dat['isLoad']
        self.Store_flag =config_dat['store']
        self.filepaths = config_dat['filepaths']
        self.uNum,self.iNum = self.Movielens_get_nums()
        self.train,self.dev,self.test = self.data_split(config_dat['neg_num4train'],
                                                   config_dat['neg_num4eval'])
        self.train['rating'] = [1 for i in range(len(self.train))]
        
        self.input_train = self.Movielens_build_input_data(1)
        self.input_dev = self.Movielens_build_input_data(2)
        self.input_test = self.Movielens_build_input_data(3)
        self.eval_dev,self.eval_test = self.Movielens_build_eval_data()
        
        self.i2g = pd.read_csv(self.filepaths[0]+'i2g.csv')
        self.i2i = pd.read_csv(self.filepaths[0]+'i2i.csv')
        self.u2a = pd.read_csv(self.filepaths[0]+'u2a.csv')
        self.u2o = pd.read_csv(self.filepaths[0]+'u2o.csv')
        self.u2u = pd.read_csv(self.filepaths[0]+'u2u.csv')
        self.i2g['link'] = [1 for _ in range(len(self.i2g))]
        self.i2i['link'] = [1 for _ in range(len(self.i2i))]
        self.u2a['link'] = [1 for _ in range(len(self.u2a))]
        self.u2o['link'] = [1 for _ in range(len(self.u2o))]
        self.u2u['link'] = [1 for _ in range(len(self.u2u))]
        self.gNum = self.i2g['gid'].max()+1
        self.aNum = self.u2a['aid'].max()+1
        self.oNum = self.u2o['oid'].max()+1
        # L_u2i,L_u2es,L_i2es,nNum_u2i,nNum_u2es,nNum_i2es
        self.mp_graphs = self.Movielens_create_mp_neighbor_graph()
        
        self.u2i = [self.Movielens_buildLaplacianMat_u2i(),[self.uNum,self.iNum]]
        self.u2es = [
            [self.Movielens_buildLaplacianMat_u2u(),0,True],
            [self.Movielens_buildLaplacianMat_u2a(),self.aNum,False],
            [self.Movielens_buildLaplacianMat_u2o(),self.oNum,False]
        ]
        self.i2es = [
            [self.Movielens_buildLaplacianMat_i2i(),0,True],
            [self.Movielens_buildLaplacianMat_i2g(),self.gNum,False]
        ]

    def Movielens_get_nums(self):
        df_u2i = pd.read_csv(self.filepaths[1])
        return df_u2i['uid'].max()+1,df_u2i['iid'].max()+1
    
    def Movielens_split(self,df_u2i):
        """ 
        interact>20 ---> [n-20:10:10]
        20>=interact>10 ---> [n-10:0:10]
        10>=interact ---> [n:0:0]
        
        """
        rd_val = []
        for i in range(len(df_u2i)):
            rd_val.append(random.random())
        df_u2i['random_val']=rd_val
        df_u2i['rank'] = df_u2i.groupby(['uid'])['random_val'].rank(method='first', ascending=False)

        grouped = df_u2i.groupby(['uid'])
        test1 = pd.DataFrame([], columns=['uid', 'iid','rating','random_val','rank'])
        dev1 = pd.DataFrame([], columns=['uid', 'iid','rating','random_val','rank'])
        train1 = pd.DataFrame([], columns=['uid', 'iid','rating','random_val','rank'])
        for name,group in tqdm.tqdm(grouped):
            if(10>=len(group)):
                train1 = train1.append(group)
            if(20>=len(group)>10):
                train1 = train1.append(group[group['rank'] >10])
                test1 = test1.append(group[group['rank'] <=10])
            if(len(group)>20):
                train1 = train1.append(group[group['rank'] >20])
                dev1= dev1.append(group[(group['rank']>10) & (group['rank']<=20)])
                test1 = test1.append(group[group['rank'] <=10])

        return train1[['uid', 'iid','rating']],dev1[['uid', 'iid','rating']], test1[['uid', 'iid','rating']]
    
    def get_prob(self,neg_items,counter_dict):
        """ get the prob of each neg_items"""
        neg_occurance = [counter_dict[i] for i in neg_items] 
        neg_num = sum(neg_occurance)
        neg_prob = [i/neg_num for i in neg_occurance]
        return neg_prob
    
    def Movielens_get_negative_items(self,df_u2i):
        paper_pool = set(df_u2i['iid'].unique())
        counter_dict = Counter(df_u2i['iid'])
        interact_status = df_u2i.groupby('uid')['iid'].apply(set).reset_index().rename(columns={'iid': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: paper_pool - x)
        interact_status['negative_probs'] = interact_status['negative_items'].apply(lambda x:self.get_prob(x,counter_dict))
        return interact_status[['uid', 'negative_items', 'negative_probs']]
    
    def Movielens_negative_sample(self,df,neg_num,neg):
        """依照文章出现的频率来采负样本【热门物品优先】"""
        full = pd.merge(df, neg, on=['uid'],how='left')
        full['negatives'] = full.apply(lambda row: np.random.choice(list(row['negative_items']),neg_num,replace=False,
                                                                  p = row['negative_probs']),axis=1)
        return full[['uid','iid','negatives']] 
    
    def Movielens_split_and_sample(self, neg_num4train, neg_num4eval):
        df_u2i = pd.read_csv(self.filepaths[1])        
        #split
        train,dev,test = self.Movielens_split(df_u2i)
        #sample
        neg = self.Movielens_get_negative_items(df_u2i)
        train_w_neg = self.Movielens_negative_sample(train,neg_num4train,neg)
        test_w_neg = self.Movielens_negative_sample(test,neg_num4eval,neg)
        dev_w_neg = self.Movielens_negative_sample(dev,neg_num4eval,neg)
        
        if self.Store_flag:
            train_w_neg.to_csv(self.filepaths[0]+'train.csv')
            test_w_neg.to_csv(self.filepaths[0]+'test.csv')
            dev_w_neg.to_csv(self.filepaths[0]+'dev.csv')
        return train_w_neg,dev_w_neg,test_w_neg
    
    def data_split(self, neg_num4train, neg_num4eval):
        if self.Load_flag==True:
            #raise Exception("Dataset has been loaded.")
            train = pd.read_csv(self.filepaths[0]+'train.csv')
            dev = pd.read_csv(self.filepaths[0]+'dev.csv')
            test = pd.read_csv(self.filepaths[0]+'test.csv')
        else:
            train,dev,test = self.Movielens_split_and_sample(neg_num4train, neg_num4eval)
        print('Dataset has been splited!')
        return train,dev,test
    ###############################################################################################################################
    
    def Movielens_build_input_data(self,dat_type):
        users, pos_items, neg_items = [], [], []
        input_dat= pd.DataFrame()
        
        if self.Load_flag==True:
            if dat_type==1:
                for row in self.train.itertuples():
                    for i in row.negatives[1:-2].split():
                        users.append(int(row.uid))
                        pos_items.append(int(row.iid))
                        neg_items.append(int(i))
            if dat_type==2:
                for row in self.dev.itertuples():
                    for i in row.negatives[1:-2].split():
                        users.append(int(row.uid))
                        pos_items.append(int(row.iid))
                        neg_items.append(int(i))
            if dat_type==3:
                for row in self.test.itertuples():
                    for i in row.negatives[1:-2].split():
                        users.append(int(row.uid))
                        pos_items.append(int(row.iid))
                        neg_items.append(int(i))
        else:
            if dat_type==1:
                for row in self.train.itertuples():
                    for i in row.negatives:
                        users.append(int(row.uid))
                        pos_items.append(int(row.iid))
                        neg_items.append(int(i))
            if dat_type==2:
                for row in self.dev.itertuples():
                    for i in row.negatives:
                        users.append(int(row.uid))
                        pos_items.append(int(row.iid))
                        neg_items.append(int(i))
            if dat_type==3:
                for row in self.test.itertuples():
                    for i in row.negatives:
                        users.append(int(row.uid))
                        pos_items.append(int(row.iid))
                        neg_items.append(int(i))
                        
        input_dat['uid']=users
        input_dat['pos_iid']=pos_items
        input_dat['neg_iid']=neg_items
        return input_dat
    
    def Movielens_build_eval_data(self):
        users,items,ratings = [],[],[]
        eval_dev=pd.DataFrame()
        if self.Load_flag==True:
            for row in self.dev.itertuples():
                for i in row.negatives[1:-2].split():
                    users.append(int(row.uid))
                    items.append(int(i))
                    ratings.append(float(0))  # negative samples get 0 rating
                users.append(int(row.uid))
                items.append(int(row.iid))
                ratings.append(float(1))
        else:
            for row in self.dev.itertuples():
                for i in row.negatives:
                    users.append(int(row.uid))
                    items.append(int(i))
                    ratings.append(float(0))  # negative samples get 0 rating
                users.append(int(row.uid))
                items.append(int(row.iid))
                ratings.append(float(1))
        eval_dev['uid']=users
        eval_dev['iid']=items
        eval_dev['rating']=ratings
        
        users,items,ratings = [],[],[]
        eval_test=pd.DataFrame()
        if self.Load_flag==True:
            for row in self.test.itertuples():
                for i in row.negatives[1:-2].split():
                    users.append(int(row.uid))
                    items.append(int(i))
                    ratings.append(float(0))  # negative samples get 0 rating
                users.append(int(row.uid))
                items.append(int(row.iid))
                ratings.append(float(1))
        else:
            for row in self.test.itertuples():
                for i in row.negatives:
                    users.append(int(row.uid))
                    items.append(int(i))
                    ratings.append(float(0))  # negative samples get 0 rating
                users.append(int(row.uid))
                items.append(int(row.iid))
                ratings.append(float(1))
        eval_test['uid']=users
        eval_test['iid']=items
        eval_test['rating']=ratings
        
        return eval_dev,eval_test
    ###############################################################################################################################
    def Movielens_sample_mp_neighbor(self,df):
        sample_df = pd.DataFrame([],columns = ['uid','iid'])
        grouped_df = df.groupby('uid')
        for name,group in tqdm.tqdm(grouped_df):
            if len(group)> 100:
                    sample_df = sample_df.append(group.sample(n=50, replace=False))
            if 100>=len(group)>10:
                    sample_df = sample_df.append(group.sample(frac=0.2, replace=False))
            if len(group)<=10:
                    sample_df = sample_df.append(group)
        return sample_df
                    
    def Movielens_create_mp_neighbor_graph(self):
        """
        creat metapath neighbor: UI, UIBI,UIVI,UICI
        and transform to DGL Graph
        """
        
        if self.Load_flag==True:
            df_uigi = pd.read_csv(self.filepaths[0]+'uigi.csv')
            df_uii = pd.read_csv(self.filepaths[0]+'uii.csv')
            df_uaui = pd.read_csv(self.filepaths[0]+'uaui.csv')
            df_uoui = pd.read_csv(self.filepaths[0]+'uoui.csv')
            df_uui = pd.read_csv(self.filepaths[0]+'uui.csv')
        else:
            temp = pd.merge(self.train[['uid','iid']], self.i2g[['iid','gid']], on=['iid'])
            temp = temp.drop(columns=['iid']).drop_duplicates().reset_index(drop=True)
            df_uigi = pd.merge(temp,self.i2g[['iid','gid']],on=['gid'])
            df_uigi = df_uigi[['uid','iid']].drop_duplicates().reset_index(drop=True)
            
            df_uii = pd.merge(self.train[['uid','iid']],self.i2i[['iid','iid_2']],on=['iid'])
            df_uii = df_uii[['uid','iid_2']].drop_duplicates().reset_index(drop=True)
            df_uii = df_uii.rename(columns={'iid_2': 'iid'})
            
            temp = pd.merge(self.train[['uid','iid']], self.u2a[['uid','aid']], on=['uid'])
            temp = temp.drop(columns=['uid']).drop_duplicates().reset_index(drop=True)
            df_uaui = pd.merge(temp,self.u2a[['uid','aid']],on=['aid'])
            df_uaui = df_uaui[['uid','iid']].drop_duplicates().reset_index(drop=True)
            
            temp = pd.merge(self.train[['uid','iid']], self.u2o[['uid','oid']], on=['uid'])
            temp = temp.drop(columns=['uid']).drop_duplicates().reset_index(drop=True)
            df_uoui = pd.merge(temp,self.u2o[['uid','oid']],on=['oid'])
            df_uoui = df_uoui[['uid','iid']].drop_duplicates().reset_index(drop=True)
            
            df_uui = pd.merge(self.train[['uid','iid']],self.u2u[['uid','uid_2']],on=['uid'])
            df_uui = df_uui[['uid_2','iid']].drop_duplicates().reset_index(drop=True)
            df_uui = df_uui.rename(columns={'uid_2': 'uid'})
            #sample (sample standard? sample strategy?)
            
            if len(df_uigi)>200000:
                print('length of df_uigi:',len(df_uigi),'start sampling...') 
                df_uigi = self.Movielens_sample_mp_neighbor(df_uigi)

            if len(df_uii)>200000:
                print('length of df_uii:',len(df_uii),'start sampling...') 
                df_uii = self.Movielens_sample_mp_neighbor(df_uii)
                
            if len(df_uaui)>200000:
                print('length of df_uaui:',len(df_uaui),'start sampling...') 
                df_uaui = self.Movielens_sample_mp_neighbor(df_uaui)
                
            if len(df_uoui)>200000:
                print('length of df_uoui:',len(df_uoui),'start sampling...') 
                df_uoui = self.Movielens_sample_mp_neighbor(df_uoui)
                
            if len(df_uui)>200000:
                print('length of df_uui:',len(df_uui),'start sampling...') 
                df_uui = self.Movielens_sample_mp_neighbor(df_uui)
                
            if self.Store_flag==True:
                df_uigi.to_csv('Movielens/uigi.csv')
                df_uii.to_csv('Movielens/uii.csv')
                df_uaui.to_csv('Movielens/uaui.csv')
                df_uoui.to_csv('Movielens/uoui.csv')
                df_uui.to_csv('Movielens/uui.csv')

        df_uigi['n_iid'] = df_uigi['iid'].apply(lambda x: x+self.uNum)        
        df_uii['n_iid'] = df_uii['iid'].apply(lambda x: x+self.uNum)
        df_uaui['n_iid'] = df_uaui['iid'].apply(lambda x: x+self.uNum)
        df_uoui['n_iid'] = df_uoui['iid'].apply(lambda x: x+self.uNum)
        df_uui['n_iid'] = df_uui['iid'].apply(lambda x: x+self.uNum)
        self.train['n_iid'] = self.train['iid'].apply(lambda x: x+self.uNum)
        
        g_ui = dgl.DGLGraph()
        g_ui.add_nodes(self.uNum+self.iNum)
        g_ui.add_edges(self.train['uid'].tolist(),self.train['n_iid'].tolist())
        g_ui.add_edges(self.train['n_iid'].tolist(),self.train['uid'].tolist())
        
        g_uigi = dgl.DGLGraph()
        g_uigi.add_nodes(self.uNum+self.iNum)
        g_uigi.add_edges(df_uigi['uid'].tolist(),df_uigi['n_iid'].tolist())
        g_uigi.add_edges(df_uigi['n_iid'].tolist(),df_uigi['uid'].tolist())
        
        g_uii = dgl.DGLGraph()
        g_uii.add_nodes(self.uNum+self.iNum)
        g_uii.add_edges(df_uii['uid'].tolist(),df_uii['n_iid'].tolist())
        g_uii.add_edges(df_uii['n_iid'].tolist(),df_uii['uid'].tolist())
        
        g_uaui = dgl.DGLGraph()
        g_uaui.add_nodes(self.uNum+self.iNum)
        g_uaui.add_edges(df_uaui['uid'].tolist(),df_uaui['n_iid'].tolist())
        g_uaui.add_edges(df_uaui['n_iid'].tolist(),df_uaui['uid'].tolist())
        
        g_uoui = dgl.DGLGraph()
        g_uoui.add_nodes(self.uNum+self.iNum)
        g_uoui.add_edges(df_uoui['uid'].tolist(),df_uoui['n_iid'].tolist())
        g_uoui.add_edges(df_uoui['n_iid'].tolist(),df_uoui['uid'].tolist())
        
        g_uui = dgl.DGLGraph()
        g_uui.add_nodes(self.uNum+self.iNum)
        g_uui.add_edges(df_uui['uid'].tolist(),df_uui['n_iid'].tolist())
        g_uui.add_edges(df_uui['n_iid'].tolist(),df_uui['uid'].tolist())
        
        return [g_ui,g_uigi,g_uii,g_uaui,g_uoui,g_uui]
    
    ###############################################################################################################################
    
    def Movielens_buildLaplacianMat_u2u(self):
        uuMat = coo_matrix((self.u2u['link'], (self.u2u['uid'], self.u2u['uid_2'])))
        A = uuMat
        selfLoop = sparse.eye(self.uNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        diag[np.isinf(diag)] = 0
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        data = L.data

        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(data)
        SparseL = torch.sparse.FloatTensor(i,data)
        
        return SparseL
    
    def Movielens_buildLaplacianMat_i2i(self):
        iiMat = coo_matrix((self.i2i['link'], (self.i2i['iid'], self.i2i['iid_2'])))
        A = iiMat
        selfLoop = sparse.eye(self.iNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        diag[np.isinf(diag)] = 0
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        data = L.data

        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(data)
        SparseL = torch.sparse.FloatTensor(i,data)
        
        return SparseL
    
    
    def Movielens_buildLaplacianMat_u2i(self):
        rt_item = self.train['iid'] + self.uNum
        uiMat = coo_matrix((self.train['rating'], (self.train['uid'], self.train['iid'])))

        uiMat_upperPart = coo_matrix((self.train['rating'], (self.train['uid'], rt_item)))
        #uiMat_upperPart.resize((self.uNum, self.uNum + self.iNum))
        
        uiMat = uiMat.transpose()
        uiMat.resize((self.iNum, self.uNum + self.iNum))
        #print(uiMat_upperPart.shape,uiMat.shape)
        A = sparse.vstack([uiMat_upperPart,uiMat])
        selfLoop = sparse.eye(self.uNum+self.iNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        diag[np.isinf(diag)] = 0
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        #print(L.shape)
        row = L.row
        col = L.col
        data = L.data
        
        #row =np.append(row,self.uNum + self.iNum-1)
        #col =np.append(col,self.uNum + self.iNum-1)
        #data =np.append(data,0)
        
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(data)
        SparseL = torch.sparse.FloatTensor(i,data)
        return SparseL
    
    def Movielens_buildLaplacianMat_i2g(self):
        rt_item = self.i2g['gid'] + self.iNum
        uiMat = coo_matrix((self.i2g['link'], (self.i2g['iid'], self.i2g['gid'])))

        uiMat_upperPart = coo_matrix((self.i2g['link'], (self.i2g['iid'], rt_item)))
        uiMat = uiMat.transpose()
        uiMat.resize((self.gNum, self.iNum + self.gNum))

        A = sparse.vstack([uiMat_upperPart,uiMat])
        selfLoop = sparse.eye(self.iNum + self.gNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        diag[np.isinf(diag)] = 0
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i,data)
        return SparseL
    
    def Movielens_buildLaplacianMat_u2a(self):
        rt_item = self.u2a['aid'] + self.uNum
        uiMat = coo_matrix((self.u2a['link'], (self.u2a['uid'], self.u2a['aid'])))

        uiMat_upperPart = coo_matrix((self.u2a['link'], (self.u2a['uid'], rt_item)))
        #uiMat_upperPart.resize((self.uNum, self.uNum + self.iNum))
        uiMat = uiMat.transpose()
        uiMat.resize((self.aNum, self.uNum + self.aNum))
        
        A = sparse.vstack([uiMat_upperPart,uiMat])
        selfLoop = sparse.eye(self.uNum+self.aNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        diag[np.isinf(diag)] = 0
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        data = L.data
        #row =np.append(row,self.uNum + self.iNum-1)
        #col =np.append(col,self.uNum + self.iNum-1)
        #data =np.append(data,0)
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(data)
        SparseL = torch.sparse.FloatTensor(i,data)
        return SparseL
    
    def Movielens_buildLaplacianMat_u2o(self):
        rt_item = self.u2o['oid'] + self.uNum
        uiMat = coo_matrix((self.u2o['link'], (self.u2o['uid'], self.u2o['oid'])))

        uiMat_upperPart = coo_matrix((self.u2o['link'], (self.u2o['uid'], rt_item)))
        #uiMat_upperPart.resize((self.uNum, self.uNum + self.iNum))
        uiMat = uiMat.transpose()
        uiMat.resize((self.oNum, self.uNum + self.oNum))
        
        A = sparse.vstack([uiMat_upperPart,uiMat])
        selfLoop = sparse.eye(self.uNum+self.oNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        diag[np.isinf(diag)] = 0
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        data = L.data
        #row =np.append(row,self.uNum + self.iNum-1)
        #col =np.append(col,self.uNum + self.iNum-1)
        #data =np.append(data,0)
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(data)
        SparseL = torch.sparse.FloatTensor(i,data)
        return SparseL


# In[ ]:




