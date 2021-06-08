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
from torch.nn import init
from torch.optim import Adam,Adadelta,RMSprop
from torch.nn.functional import normalize
from dgl.nn.pytorch import GATConv

from scipy.sparse import coo_matrix
from scipy.sparse import vstack
from scipy import sparse
from math import exp
from collections import Counter


# In[2]:


class MF(Module):
    def __init__(self,config):
        super(MF, self).__init__()
        self.uEmbd = nn.Embedding(config['num_users'],config['embed_dim'])
        self.iEmbd = nn.Embedding(config['num_items'],config['embed_dim'])
        self.uBias = nn.Embedding(config['num_users'],1)
        self.iBias = nn.Embedding(config['num_items'],1)
        self.overAllBias = nn.Parameter(torch.Tensor([0]))
        
    def weight_init(self):
        wts= [self.uEmbd.weight,
            self.iEmbd.weight]
        for wt in wts:
            nn.init.xavier_normal_(wt, gain=1)   

    def forward(self, userIdx,itemIdx):
        uembd = self.uEmbd(userIdx)
        iembd = self.iEmbd(itemIdx)
        ubias = self.uBias(userIdx)
        ibias = self.iBias(itemIdx)

        biases = ubias + ibias + self.overAllBias
        prediction = torch.sum(torch.mul(uembd,iembd),dim=1) + biases.flatten()

        return prediction


# In[ ]:


class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        self.logistic = torch.nn.Sigmoid()
        
    def weight_init(self):
        wts= [self.embedding_user_mlp.weight,
              self.embedding_item_mlp.weight,
              self.embedding_user_mf.weight,
              self.embedding_item_mf.weight,
              self.fc_layers[0].weight,
              self.fc_layers[1].weight,
              self.fc_layers[2].weight,
              self.affine_output.weight,
              ]
        for wt in wts:
            nn.init.xavier_normal_(wt, gain=1)   
            
    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        #rating = self.logistic(logits)
        return logits


# In[ ]:


class GNNLayer(Module):

    def __init__(self,inF,outF,useCuda):
        super(GNNLayer,self).__init__()
        self.inF = inF
        self.outF = outF
        self.useCuda = useCuda
        self.linear = torch.nn.Linear(in_features=inF,out_features=outF)
        self.interActTransform = torch.nn.Linear(in_features=inF,out_features=outF)

    def forward(self, laplacianMat,selfLoop,features):
        L1 = laplacianMat + selfLoop
        L2 = laplacianMat
        if self.useCuda == True:
            L1 = L1.cuda()
            L2 = L2.cuda()
            
        inter_part1 = self.linear(torch.sparse.mm(L1,features))
        inter_feature = torch.mul(torch.sparse.mm(L2,features),features)        
        inter_part2 = self.interActTransform(inter_feature)

        return inter_part1+inter_part2

class GCF(Module):

    def __init__(self,config,laplacianMat):

        super(GCF,self).__init__()
        self.useCuda = config['cuda']
        self.userNum = config['num_users']
        self.itemNum = config['num_items']
        self.uEmbd = nn.Embedding(config['num_users'],config['embed_dim'])
        self.iEmbd = nn.Embedding(config['num_items'],config['embed_dim'])
        self.layers = config['layers']
        self.GNNlayers = torch.nn.ModuleList()
        self.LaplacianMat = laplacianMat # sparse format
        self.leakyRelu = nn.LeakyReLU()
        self.selfLoop = self.getSparseEye(self.LaplacianMat.shape[0])
        #self.logistic = torch.nn.Sigmoid()
        
        #self.transForm1 = nn.Linear(in_features=sum(self.layers)*2,out_features=64)
        #self.transForm2 = nn.Linear(in_features=64,out_features=32)
        #self.transForm3 = nn.Linear(in_features=32,out_features=1)

        for From,To in zip(self.layers[:-1],self.layers[1:]):
            self.GNNlayers.append(GNNLayer(From,To,self.useCuda))
    
     # 参数初始化
    def weight_init(self):
        wts= [self.uEmbd.weight,
              self.iEmbd.weight,
              self.GNNlayers[0].linear.weight,
              self.GNNlayers[0].interActTransform.weight,
              self.GNNlayers[1].linear.weight,
              self.GNNlayers[1].interActTransform.weight,
              self.GNNlayers[2].linear.weight,
              self.GNNlayers[2].interActTransform.weight
              #self.transForm1.weight,
              #self.transForm2.weight,
              #self.transForm3.weight,
              ]
        for wt in wts:
            nn.init.xavier_normal_(wt, gain=1)      
    
    def getSparseEye(self,num):
        i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
        val = torch.FloatTensor([1]*num)
        return torch.sparse.FloatTensor(i,val)

    def getFeatureMat(self):
  
        uidx = torch.LongTensor([i for i in range(self.userNum)])
        iidx = torch.LongTensor([i for i in range(self.itemNum)])
        if self.useCuda == True:
            uidx = uidx.cuda()
            iidx = iidx.cuda()

        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)

        features = torch.cat([userEmbd,itemEmbd],dim=0)
        return features

    def forward(self,userIdx,itemIdx):

        itemIdx = itemIdx + self.userNum
        features = self.getFeatureMat()
        finalEmbd = features.clone()
        for gnn in self.GNNlayers:
            features = gnn(self.LaplacianMat,self.selfLoop,features)
            features = self.leakyRelu(features)
            features = normalize(features, 2, 1)
            finalEmbd = torch.cat([finalEmbd,features.clone()],dim=1)
            
        userEmbd = finalEmbd[userIdx]
        itemEmbd = finalEmbd[itemIdx]

        add  = torch.mul(userEmbd,itemEmbd)
        logits = torch.sum(add,dim=1)

        return logits


# In[ ]:


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        return (beta * z).sum(1)

class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class HAN(nn.Module):
    def __init__(self,config, gs):
        super(HAN, self).__init__()
        self.gs = gs
        self.usecuda = config['cuda']
        self.dropout = config['dropout']
        self.num_heads = config['num_heads']
        self.num_meta_paths = len(gs)
        self.userNum = config['num_users']
        self.itemNum = config['num_items']
        self.uEmbd = nn.Embedding(config['num_users'],config['embed_dim'])
        self.iEmbd = nn.Embedding(config['num_items'],config['embed_dim'])
        self.in_size = config['embed_dim']
        self.hidden_size = config['hidden_dim']
        #self.h = torch.cat((self.uEmbd,self.iEmbd),dim=0)
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(self.num_meta_paths, self.in_size, self.hidden_size, self.num_heads[0], self.dropout))
        for l in range(1, len(self.num_heads)):
            self.layers.append(HANLayer(self.num_meta_paths, self.hidden_size * self.num_heads[l-1],
                                        self.hidden_size,  self.num_heads[l],  self.dropout))
        #self.predict = nn.Linear(hidden_size * num_heads[-1], 1)
    
    def weight_init(self):
        wts = [
            self.uEmbd.weight,
            self.iEmbd.weight,
            self.layers[0].gat_layers[0].fc.weight,
            self.layers[0].gat_layers[1].fc.weight,
            self.layers[0].gat_layers[2].fc.weight,
            self.layers[0].gat_layers[3].fc.weight,
            self.layers[0].gat_layers[4].fc.weight,
            self.layers[0].gat_layers[5].fc.weight,
            self.layers[0].semantic_attention.project[0].weight,
            self.layers[0].semantic_attention.project[2].weight
            #self.layers[1].gat_layers[0].fc.weight,
            #self.layers[1].gat_layers[1].fc.weight,
            #self.layers[1].gat_layers[2].fc.weight,
            #self.layers[1].semantic_attention.project[0].weight,
            #self.layers[1].semantic_attention.project[2].weight
            #self.layers[2].gat_layers[0].fc.weight,
            #self.layers[2].gat_layers[1].fc.weight,
            #self.layers[2].gat_layers[2].fc.weight,
            #self.layers[2].semantic_attention.project[0].weight,
            #self.layers[2].semantic_attention.project[2].weight
        ]
        
        for wt in wts:
            nn.init.xavier_normal_(wt, gain=1)  
    
    def getFeatureMat(self):
        uidx = torch.LongTensor([i for i in range(self.userNum)])
        iidx = torch.LongTensor([i for i in range(self.itemNum)])
        if self.usecuda  == True:
            uidx = uidx.cuda()
            iidx = iidx.cuda()
        #print(type(uidx))
        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)

        features = torch.cat([userEmbd,itemEmbd],dim=0)
        return features
    
    def forward(self, userIdx,itemIdx):
        itemIdx = itemIdx + self.userNum
        features = self.getFeatureMat()
        #finalEmbd = features.clone()
        for gnn in self.layers:
            features = gnn(self.gs, features)
          
        userEmbd = features[userIdx]
        itemEmbd = features[itemIdx]    
        #print(userEmbd.size()) 
        add  = torch.mul(userEmbd,itemEmbd)
        logits = torch.sum(add,dim=1)
        
        return logits


# In[ ]:


class RelationAttention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(RelationAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        return (beta * z).sum(1)

class SelfCell(Module):

    def __init__(self,inF,outF,useCuda):
        super(SelfCell,self).__init__()
        self.inF = inF
        self.outF = outF
        self.useCuda = useCuda
        self.uuTransform = torch.nn.Linear(in_features=inF,out_features=outF)
        self.iiTransform = torch.nn.Linear(in_features=inF,out_features=outF)
        self.uuInterAct = torch.nn.Linear(in_features=inF,out_features=outF)
        self.iiInterAct = torch.nn.Linear(in_features=inF,out_features=outF)

    def forward(self, laplacianMat,selfLoop,features,userNum,itemNum):
        L1 = laplacianMat + selfLoop
        L2 = laplacianMat
        if self.useCuda == True:
            L1 = L1.cuda()
            L2 = L2.cuda()

        adj_features = torch.cat((self.uuTransform(features[:userNum]),self.iiTransform(features[userNum:userNum+itemNum])),dim=0)
        inter_part1 = torch.sparse.mm(L1,adj_features)
        
        inter_features = torch.cat((self.uuInterAct(torch.mul(features[:userNum],features[:userNum])),                                    self.iiInterAct(torch.mul(features[userNum:userNum+itemNum],features[userNum:userNum+itemNum]))),dim=0)
        inter_part2 = torch.sparse.mm(L2,inter_features)

        return inter_part1+inter_part2

    
class CrossCell(Module):

    def __init__(self,inF,outF,useCuda):
        super(CrossCell,self).__init__()
        self.inF = inF
        self.outF = outF
        self.useCuda = useCuda
        self.uiTransform = torch.nn.Linear(in_features=inF,out_features=outF)
        self.uiInterAct = torch.nn.Linear(in_features=inF,out_features=outF)

    def forward(self, laplacianMat,selfLoop,features):
        L1 = laplacianMat + selfLoop
        L2 = laplacianMat
        if self.useCuda == True:
            L1 = L1.cuda()
            L2 = L2.cuda()
            
        inter_part1 = torch.sparse.mm(L1,self.uiTransform(features))     
        inter_part2 = torch.sparse.mm(L2,self.uiInterAct(torch.mul(features,features)))

        return inter_part1+inter_part2

    
class NHGCFLayer(Module):
    
    def __init__(self,inF,outF,useCuda,userNum,itemNum,crossNum):
        super(NHGCFLayer,self).__init__()
        self.inF = inF
        self.outF = outF
        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum
        self.crossNum = crossNum
        
        self.SelfCell = SelfCell(self.inF,self.outF,self.useCuda)
        self.CrossCells = torch.nn.ModuleList()
        for _ in range(self.crossNum):
            self.CrossCells.append(CrossCell(self.inF,self.outF,self.useCuda))
        self.relation_attention = RelationAttention(in_size=self.inF)
        
    def forward(self,L_self,L_cross,selfLoops,features):
        relation_embeddings = []
        for sL,L in zip(selfLoops[:-1],L_cross):
            relation_embeddings.append(self.CrossCell(L,sL,features))
        relation_embeddings.append(self.SelfCell(L_self,selfLoops[-1],features,self.userNum,self.itemNum))
        
        relation_embeddings = torch.stack(relation_embeddings, dim=1)
        return self.relation_attention(relation_embeddings)

    
class NHGCF(Module):

    def __init__(self,config,L_corss,L_self):

        super(NHGCF,self).__init__()
        self.useCuda = config['cuda']
        self.userNum = config['num_users']
        self.itemNum = config['num_items']
        self.uEmbd = nn.Embedding(config['num_users'],config['embed_dim'])
        self.iEmbd = nn.Embedding(config['num_items'],config['embed_dim'])
        self.layers = config['layers']
        self.NHGCFLayers = torch.nn.ModuleList()
        self.L_corss = L_corss
        self.L_self = L_self
        self.leakyRelu = nn.LeakyReLU()
        
        self.Ls = self.L_corss
        self.Ls.append(L_self)
        self.selfLoops = [self.getSparseEye(L.shape[0]+L.shape[1]) for L in self.Ls ]
        
        for From,To in zip(self.layers[:-1],self.layers[1:]):
            self.NHGCFLayers.append(NHGCFLayer(From,To,self.useCuda,self.userNum,self.itemNum,len(self.L_corss)))
    
     # 参数初始化
    def weight_init(self):
        wts= [self.uEmbd.weight,
                self.iEmbd.weight,
                self.NHGCFLayers[0].SelfCell.uuTransform.weight,
                self.NHGCFLayers[0].SelfCell.iiTransform.weight,
                self.NHGCFLayers[0].SelfCell.uuInterAct.weight,
                self.NHGCFLayers[0].SelfCell.iiInterAct.weight,
                self.NHGCFLayers[0].InterCell.uiTransform.weight,
                self.NHGCFLayers[0].InterCell.uiInterAct.weight,
                self.NHGCFLayers[0].relation_attention.project[0].weight,
                self.NHGCFLayers[0].relation_attention.project[2].weight,
                self.NHGCFLayers[1].SelfCell.uuTransform.weight,
                self.NHGCFLayers[1].SelfCell.iiTransform.weight,
                self.NHGCFLayers[1].SelfCell.uuInterAct.weight,
                self.NHGCFLayers[1].SelfCell.iiInterAct.weight,
                self.NHGCFLayers[1].InterCell.uiTransform.weight,
                self.NHGCFLayers[1].InterCell.uiInterAct.weight,
                self.NHGCFLayers[1].relation_attention.project[0].weight,
                self.NHGCFLayers[1].relation_attention.project[2].weight,
                self.NHGCFLayers[2].SelfCell.uuTransform.weight,
                self.NHGCFLayers[2].SelfCell.iiTransform.weight,
                self.NHGCFLayers[2].SelfCell.uuInterAct.weight,
                self.NHGCFLayers[2].SelfCell.iiInterAct.weight,
                self.NHGCFLayers[2].InterCell.uiTransform.weight,
                self.NHGCFLayers[2].InterCell.uiInterAct.weight,
                self.NHGCFLayers[2].relation_attention.project[0].weight,
                self.NHGCFLayers[2].relation_attention.project[2].weight
              ]
        for wt in wts:
            nn.init.xavier_normal_(wt, gain=1)      
    
    def getSparseEye(self,num):
        i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
        val = torch.FloatTensor([1]*num)
        return torch.sparse.FloatTensor(i,val)


    def getFeatureMat(self):
  
        uidx = torch.LongTensor([i for i in range(self.userNum)])
        iidx = torch.LongTensor([i for i in range(self.itemNum)])
        if self.useCuda == True:
            uidx = uidx.cuda()
            iidx = iidx.cuda()

        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)

        features = torch.cat([userEmbd,itemEmbd],dim=0)
        return features

    def forward(self,userIdx,itemIdx):

        itemIdx = itemIdx + self.userNum
        features = self.getFeatureMat()
        finalEmbd = features.clone()
        for gnn in self.NHGCFLayers:
            features = gnn(self.L_self,self.L_corss,self.selfLoops,features)
            features = self.leakyRelu(features)
            finalEmbd = torch.cat([finalEmbd,features.clone()],dim=1)

        userEmbd = finalEmbd[userIdx]
        itemEmbd = finalEmbd[itemIdx]

        add  = torch.mul(userEmbd,itemEmbd)
        logits = torch.sum(add,dim=1)

        return logits


# In[ ]:


class RelationAttention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(RelationAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False))

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)
    
class GCNCell(Module):
    def __init__(self,inF,outF,useCuda):
        super(GCNCell,self).__init__()
        self.inF = inF
        self.outF = outF
        self.useCuda = useCuda
        self.Transform = torch.nn.Linear(in_features=inF,out_features=outF)
        self.InterAct = torch.nn.Linear(in_features=inF,out_features=outF)

    def forward(self, laplacianMat,selfLoop,features):
        L1 = laplacianMat + selfLoop
        L2 = laplacianMat
        if self.useCuda == True:
            L1 = L1.cuda()
            L2 = L2.cuda()
        inter_part1 = torch.sparse.mm(L1,self.Transform(features))     
        inter_part2 = torch.sparse.mm(L2,self.InterAct(torch.mul(features,features)))
        return inter_part1+inter_part2
    
    
class NHGCFLayer(Module):
    def __init__(self,inF,outF,useCuda,model_type,n_u2e,n_i2e):
        super(NHGCFLayer,self).__init__()
        self.inF = inF
        self.outF = outF
        self.useCuda = useCuda

        self.u2i_Cell = GCNCell(inF,outF,useCuda)
        
        self.u2e_Cells = torch.nn.ModuleList()
        for i in range(n_u2e):
            self.u2e_Cells.append(GCNCell(inF,outF,useCuda))
            
        self.i2e_Cells = torch.nn.ModuleList()
        for i in range(n_i2e):
            self.i2e_Cells.append(GCNCell(inF,outF,useCuda))
         
        self.u_relation_attention = RelationAttention(in_size=self.inF)
        self.i_relation_attention = RelationAttention(in_size=self.inF)
        
    def forward(self, u2i_pack, u2e_pack, i2e_pack, u_feature, i_feature, u2e_features, i2e_features):
        u_embeddings = []
        i_embeddings = []
        u2e_embeddings =[]
        i2e_embeddings =[]
        u_num = u2i_pack[2][0]
        #i_num = u2i_pack[2][1]
        
        for (L,sL),u2e_feature,u2e_Cell in zip(u2e_pack,u2e_features,self.u2e_Cells):
            if u2e_feature is not None:
                temp = u2e_Cell(L,sL,torch.cat([u_feature,u2e_feature],dim=0))
                u2e_embeddings.append(temp[u_num:])
                u_embeddings.append(temp[:u_num])
            else:
                # self interact
                temp = u2e_Cell(L,sL,u_feature)
                u2e_embeddings.append(None)
                u_embeddings.append(temp)
            
            
        for (L,sL),i2e_feature,i2e_Cell in zip(i2e_pack,i2e_features,self.i2e_Cells):
            if i2e_feature is not None:
                temp = i2e_Cell(L,sL,torch.cat([i_feature,i2e_feature],dim=0))
                i2e_embeddings.append(temp[i_num:])
                i_embeddings.append(temp[:i_num]) 
            else:
                # self interact
                temp = i2e_Cell(L,sL,i_feature)
                i2e_embeddings.append(None)
                i_embeddings.append(temp)
                
        temp = self.u2i_Cell(u2i_pack[0],u2i_pack[1],torch.cat([u_feature,i_feature],dim=0))
        i_embeddings.append(temp[u_num:]) 
        u_embeddings.append(temp[:u_num])
        
        i_embeddings = torch.stack(i_embeddings, dim=1)
        u_embeddings = torch.stack(u_embeddings, dim=1)
        return self.u_relation_attention(u_embeddings),self.i_relation_attention(i_embeddings),u2e_embeddings,i2e_embeddings 
    
    

class NHGCF(Module):
    def __init__(self,config,u2i,u2es,i2es):
        """
        u2es[x][0]: laplacian
        u2es[x][1]: node num
        u2es[x][2]: self intercat tag
        """
        super(NHGCF,self).__init__()
        self.useCuda = config['cuda']
        
        self.userNum = u2i[1][0]
        self.itemNum = u2i[1][1]
        
        self.uEmbd = nn.Embedding(self.userNum,config['embed_dim'])
        self.iEmbd = nn.Embedding(self.itemNum,config['embed_dim'])
        self.L_u2i = u2i[0]
        self.selfLoop_u2i = self.getSparseEye(self.L_u2i.shape[0])
        self.u2i_pack = (self.L_u2i,self.selfLoop_u2i,[self.userNum,self.itemNum])
        
        self.model_type_flag = 0
        
        self.u2eEmbds=torch.nn.ModuleList()
        if u2es is not None:
            self.L_u2es = []
            self.u2eNums =[]
            self.selfLoop_u2es = []
            for u2e in u2es:
                self.L_u2es.append(u2e[0])
                self.u2eNums.append(u2e[1])
                if u2e[2]==False:
                    self.u2eEmbds.append(nn.Embedding(u2e[1],config['embed_dim']))
                    #self.selfLoop_u2es.append(self.getSparseEye(u2e[0].shape[0]+u2e[0].shape[1]))
                else:
                    # self interact
                    self.u2eEmbds.append(None)
                self.selfLoop_u2es.append(self.getSparseEye(u2e[0].shape[0]))
            self.u2e_pack = zip(self.L_u2es,self.selfLoop_u2es)
            self.n_u2e = len(self.u2eNums)
            self.model_type_flag += 1.5
            
        self.i2eEmbds=torch.nn.ModuleList()  
        if i2es is not None:
            self.L_i2es = []
            self.i2eNums =[]
            self.selfLoop_i2es = []
            for i2e in i2es:
                self.L_i2es.append(i2e[0])
                self.i2eNums.append(i2e[1])
                if i2e[2]==False:
                    self.i2eEmbds.append(nn.Embedding(i2e[1],config['embed_dim']))
                    #self.selfLoop_i2es.append(self.getSparseEye(i2e[0].shape[0]+i2e[0].shape[1]))
                else:
                    # self interact
                    self.i2eEmbds.append(None)
                self.selfLoop_i2es.append(self.getSparseEye(i2e[0].shape[0]))
            self.i2e_pack = zip(self.L_i2es,self.selfLoop_i2es)
            self.n_i2e = len(self.i2eNums)
            self.model_type_flag += 0.5
            
        if self.model_type_flag==0:
            self.model_type = 'naive'
        elif self.model_type_flag==2:
            self.model_type = 'full'
        elif self.model_type_flag==1.5:
            self.model_type = 'u-side'
        elif self.model_type_flag==0.5:
            self.model_type = 'i-side'
        
        self.layers = config['layers']
        self.NHGCFLayers = torch.nn.ModuleList()
        self.leakyRelu = nn.LeakyReLU()

        for From,To in zip(self.layers[:-1],self.layers[1:]):
            self.NHGCFLayers.append(NHGCFLayer(From,To,self.useCuda,self.model_type,self.n_u2e,self.n_i2e))  
            
    def getSparseEye(self,num):
        i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
        val = torch.FloatTensor([1]*num)
        return torch.sparse.FloatTensor(i,val)
    
    def getFeatureMat(self,num,embd):
        idx = torch.LongTensor([i for i in range(num)])
        if self.useCuda == True:
            #print('in')
            idx = idx.cuda()
        #print(idx)
        #idx = idx.cuda()
        fullEmbd = embd(idx)
        #print('out')
        return fullEmbd

    def forward(self,userIdx,itemIdx):

        #itemIdx = itemIdx + self.userNum
        u_feature = self.getFeatureMat(self.userNum,self.uEmbd)
        i_feature = self.getFeatureMat(self.itemNum,self.iEmbd)
        
        u2e_features = []
        for num,embd in zip(self.u2eNums,self.u2eEmbds):
            if embd is not None:
                u2e_features.append(self.getFeatureMat(num,embd))
            else:
                # self interact
                u2e_features.append(None)
            
        i2e_features = []
        for num,embd in zip(self.i2eNums,self.i2eEmbds):
            if embd is not None:
                u2e_features.append(self.getFeatureMat(num,embd))
            else:
                # self interact
                u2e_features.append(None)
        
        u_finalEmbd = u_feature.clone()
        i_finalEmbd = i_feature.clone()
        for gnn in self.NHGCFLayers:
            u_feature, i_feature, u2e_features, i2e_features = gnn(self.u2i_pack, self.u2e_pack, self.i2e_pack, 
                                                                   u_feature, i_feature, u2e_features, i2e_features)
            u_feature = self.leakyRelu(u_feature)
            i_feature = self.leakyRelu(i_feature)
  
            u2e_features = [self.leakyRelu(f) if (f is not None) else None for f in u2e_features]
            i2e_features = [self.leakyRelu(f) if (f is not None) else None for f in i2e_features]
            u_finalEmbd = torch.cat([u_finalEmbd,u_feature.clone()],dim=1)
            i_finalEmbd = torch.cat([i_finalEmbd,i_feature.clone()],dim=1)

        userEmbd = u_finalEmbd[userIdx]
        itemEmbd = i_finalEmbd[itemIdx]

        add  = torch.mul(userEmbd,itemEmbd)
        logits = torch.sum(add,dim=1)

        return logits

