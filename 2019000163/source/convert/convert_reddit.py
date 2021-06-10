import numpy as np
import pickle as pkl
import sys
import os
import json
import time
import scipy.sparse
import struct
import networkx as nx
import numpy as np
import scipy.sparse as sp
import sklearn.preprocessing
from networkx.readwrite import json_graph
from sklearn.preprocessing import StandardScaler

def split_random(seed, n, n_train, n_val):
	np.random.seed(seed)
	rnd = np.random.permutation(n)
	train_idx = np.sort(rnd[:n_train])
	val_idx = np.sort(rnd[n_train:n_train + n_val])
	train_val_idx = np.concatenate((train_idx, val_idx))
	test_idx = np.sort(np.setdiff1d(np.arange(n), train_val_idx))
	return train_idx, val_idx, test_idx

def Reddit(datastr):

	red=np.load(datastr)

	#save features
	print("For features...")
	feats=red['attr_matrix']
	scaler = sklearn.preprocessing.StandardScaler()
	scaler.fit(feats)
	feats = scaler.transform(feats)
	feats=np.array(feats,dtype=np.float)
	np.save('../data/reddit_feat.npy',feats)

	#save graph
	print("For graph...")
	el=red['adj_indices']
	pl=red['adj_indptr']
	el=np.array(el,dtype=np.uint32)
	pl=np.array(pl,dtype=np.uint32)

	EL_re=[]
	for i in range(1,pl.shape[0]):
		EL_re+=sorted(el[pl[i-1]:pl[i]],key=lambda x:pl[x+1]-pl[x])
	EL_re=np.asarray(EL_re,dtype=np.uint32)

	f1=open('../data/reddit_adj_el.txt','wb')
	for i in EL_re:
		m=struct.pack('I',i)
		f1.write(m)
	f1.close()

	f2=open('../data/reddit_adj_pl.txt','wb')
	for i in pl:
		m=struct.pack('I',i)
		f2.write(m)
	f2.close()

	#save labels
	print("For labels...")
	labels=red['labels']
	n=labels.shape[0]
	num_classes =labels.max() + 1
	n_train = num_classes * 20
	n_val = n_train * 10
	idx_train, idx_val, idx_test = split_random(0, n, n_train, n_val)
	np.savez('../data/reddit_labels.npz',labels=labels,idx_train=idx_train,idx_val=idx_val,idx_test=idx_test)

if __name__ == "__main__":

	#Your file storage path. For example, this is shown below.
	datastr="/XXX/reddit.npz"
	Reddit(datastr)