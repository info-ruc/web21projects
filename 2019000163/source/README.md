## Requirements
- CUDA 10.1
- python 3.7.7
- pytorch 1.5.1
- GCC 5.4.0
- cython 0.29.21
- eigency 1.77
- numpy 1.18.1
- torch-geometric 1.4.3 (https://github.com/rusty1s/pytorch_geometric/blob/master/docs/source/notes/installation.rst)
- tqdm 4.46.0
- ogb 1.2.3
- [eigen 3.3.8](https://gitlab.com/libeigen/eigen.git)


## Datasets
We provide the dataset Reddit in the folder "data". To accelerate the speed to download the codes, we take out the data file of Reddit from the master directory, and only set an empty folder named "data" to show the accurate file structures("AGP-master/NodeClassication-GNN/data/"). After you download the folder "data" from our dropbox address, please put it in the directory "AGP-master/NodeClassication-GNN/" to substitute the empty folder "data".

The other three datasets(Yelp, Amazon2M, papers100M) can be downloaded from:
(1)Yelp: https://github.com/GraphSAINT/GraphSAINT
(2)Amazon2M: https://github.com/google-research/google-research/tree/master/cluster_gcn
(3)Papers100M: https://ogb.stanford.edu

For Yelp and Amazon2m, after downloading the raw data from the above links, the two datasets need to be converted to the CSR format. In the directory "AGP-master/NodeClassication-GNN/convert/", we provide the codes to convert the two datasets. Please run 'python convert_yelp.py' or 'python convert_amazon.py' correspondingly.

For papers100M, the website "Open Graph Benchmark" provides an automatic method to download and convert the dataset. So you can straightly run 'python convert_papers100M.py' instead of downloading the dataset papers100M manually. 

Note that the provided dataset Reddit has been converted to CSR format. So you don't need to convert the format of Reddit. 



## Compilation

Cython needs to be compiled before running, run this command:
```
python setup.py build_ext --inplace
```


## Run the code

Reddit
```
sh reddit.sh
```
Yelp
```
sh yelp.sh
```
Amazon2M
```
sh amazon2M.sh
```
Papers100M
```
sh papers100M.sh
```

