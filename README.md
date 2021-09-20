# HSTGNN

## Inroduction 

This repository is a pytorch version implementation of DEXA 2021 conference paper [paper link](https://link.springer.com/chapter/10.1007/978-3-030-86472-9_29)
"Traffic Flow Prediciton through the Fusion of Spatial Temporal Data and Points of Interest".  

## Run program

### 1.pakage dependencies  
   To run this code, you need to install the following packages:  
   torch==1.6.1    
   numpy==1.16.3  
   pandas==1.1.5  
   scipy==1.2.1  
   h5py==3.1.0  

### 2.datasets  
   download the dataset from [repository link](https://github.com/panzheyi/ST-MetaNet), 
   ZheYi Pan,KDD2019,"Urban traffic prediction from spatio-temporal data using deep meta learning"  

copy the BJ_FLOW.h5,BJ_POI.h5 to data folder    
-- bj_tfidf_poi.h5 is based on BJ_POI.h5 and  has been processed by TF-IDF algorithm to calculate the importance of poi in each region.  
-- cossimi_graph.npz is obtained from BJ_FLOW.h5 using cossine similarity to caculate the flow similarity of region pairs with a threshold to determine
whether there is an edge between two regions, datails see /utils/generate_time_embedding.py, cossimi_graph serves as initial adjacent matrix to initialize the parameter of 
adaptive adjacent matrix.

### 3.Run  
nohup python -u train.py > file.log 2>&1 &  

If you find this repository is helpful to you, please cite our paper, thanks for your attention.
