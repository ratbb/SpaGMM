#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 20:51:02 2022

@author: rwang
"""
import pickle
import community as community_louvain
import networkx as nx
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from unit import *
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
import pandas as pd 

def normallize_matrix(matrix):

    for times , i in enumerate(matrix):
        matrix[times,:] = i / (np.sum(i)-1)

    return matrix

def get_prob(bilinear_adata,image_factor=0.05,classify=5):
        
    smooth_prob=np.zeros((bilinear_adata.to_df().shape[0],classify))
    
    for i in range(classify): #这里5要更改为class数目
    
        smooth_prob[:,i]=bilinear_adata.obs["smooth_prob_"+str(i)]
    
    img_prob=np.dot(smooth_prob,smooth_prob.T)
    
    img_prob= np.where(img_prob<image_factor,image_factor,img_prob)
    
    normallize_matrix(img_prob)
    
    return img_prob

def get_gene(bilinear_adata,X_pca):
    
    data=bilinear_adata.obsm["X_pca"][:,:15]
    
    gene_matrix_prob=np.reshape(np.sum(data**2,axis=1),(data.shape[0],1))+ np.sum(data**2,axis=1)-2*data.dot(data.T)
    
    # gene_matrix_prob = find_perplexity(gene_matrix_prob,perplexity=1000)
    
    # gene_matrix_prob = find_perplexity(gene_matrix_prob, perplexity=perplexity)
    
    gene_matrix_prob = 1/(1+gene_matrix_prob)
    
    normallize_matrix(gene_matrix_prob)

    # gene_matrix_prob=(gene_matrix_prob+gene_matrix_prob.T)/2
    
    #mm = MinMaxScaler()
    
    #gene_matrix_prob = mm.fit_transform(gene_matrix_prob)
    
    return gene_matrix_prob

def get_gene_neighbour(bilinear_adata,neighbour=400):
    
    data=bilinear_adata.obsm["X_pca"][:,:15]
    
    nbrs = NearestNeighbors(n_neighbors=neighbour, algorithm='kd_tree').fit(data)
    
    gene_distance , gene_neighbour = nbrs.kneighbors(data)
    
    return gene_neighbour
    


def louvain_based_gene_negighobur(mixture_matrix,bilinear_adata,gene_neighbour,neighbours=200,resolution=3,random_state=1):
    
    gene_neighbour = gene_neighbour[:,1:]
    
    result= np.ones(gene_neighbour.shape)
    
    for times,i in enumerate(gene_neighbour):
        
        for space,j in enumerate(i):
            
            result[times,space] = mixture_matrix[times,j]
            
    neighbour2 = np.argsort(-result,axis=1)
    
    neighbour2=neighbour2[:,1:neighbours]
    
    G = nx.Graph()
    
    slic_name=list(bilinear_adata.obs.index)
    
    G.add_nodes_from(slic_name)
    
    elist=[]
    
    for times,i in enumerate(neighbour2):
        
        for j in i:
    
            elist.append((slic_name[times], slic_name[gene_neighbour[times,j]], result[times,j]))
            
        if times % 10000 == 0:
            
            print("已经添加了{}node的边".format(times))
            
    G.add_weighted_edges_from(elist)
            
    partition = community_louvain.best_partition(G,resolution=resolution,random_state=random_state,randomize=False)
    
    return partition
    
    

def get_cluster(output_path,X_pca=15,neighbours=200,resolution=1.1,classify=10):

    bilinear_adata = sc.read_h5ad(output_path+"/bilinear_data.h5ad")
    
    sc.pp.scale(bilinear_adata, max_value=10)
    
    sc.tl.pca(bilinear_adata, svd_solver='arpack',random_state =1)
    
    # sc.pp.neighbors(bilinear_adata, n_neighbors=neighbours, n_pcs=X_pca)
    
    # sc.tl.leiden(bilinear_adata)


    img_prob = get_prob(bilinear_adata , image_factor=1e-3,classify =classify)
    
    gene_matrix_prob = get_gene(bilinear_adata, X_pca)
    
    mixture_matrix = gene_matrix_prob*img_prob
    
    #normallize_matrix(mixture_matrix)
    
    del img_prob
    
    del gene_matrix_prob
    
    gene_neighbour = get_gene_neighbour(bilinear_adata,neighbour=neighbours)
    
    partition = louvain_based_gene_negighobur(mixture_matrix, bilinear_adata, gene_neighbour ,neighbours= neighbours , resolution= resolution)
    
    print("louvain 分类为 {} 组".format(max(partition.values())))
    
    bilinear_adata.obs["my_classify_result"]=pd.Series([partition[i] for i in partition],index=bilinear_adata.to_df().index , dtype="category")
    
    #bilinear_adata.write_h5ad(output_path+"/bilinear_data_result.h5ad")
    
    

    # partition=louvain_wr(mixture_matrix,bilinear_adata ,neighbours=200,resolution=2)
    
    # partition =partition_new
    
    # result_max = max(partition,key=lambda x:partition[x])
    #
    # result_max = partition[result_max]
    #
    # temp_sum = np.zeros([len(partition),int(result_max)+1])
    #
    # spatial_position = np.zeros((bilinear_adata.to_df().shape[0], 2))
    #
    # for times, i in enumerate(["pos_x", "pos_y"]):
    #
    #     spatial_position[:, times] = bilinear_adata.obs[i]
    #
    # nbrs = NearestNeighbors(n_neighbors=6, algorithm='auto').fit(spatial_position)
    #
    # distances, indices = nbrs.kneighbors(spatial_position)
    #
    # bili_slic = bilinear_adata.to_df().index
    #
    # for index,i in enumerate(indices):
    #
    #     for j in i:
    #
    #         temp_sum[index,partition[bili_slic[j]]]+=1
    #
    # gamma=2
    #
    # prior_prob = np.exp(temp_sum*gamma/4)
    #
    # result_prob = np.zeros(prior_prob.shape)
    #
    # normallize_matrix(prior_prob)
    #
    # gene_means= np.zeros([int(result_max)+1,15])
    #
    with open(output_path+"/slic.pickle","rb") as f:

        slics=pickle.load(f)
    #
    # data=bilinear_adata.obsm["X_pca"][:,:15]
    #
    # count = {}
    #
    # for i in partition:
    #
    #     if partition[i] in count:
    #
    #         count[partition[i]] += 1
    #
    #     else:
    #
    #         count[partition[i]] = 1
    #
    # for times , i in enumerate(bili_slic):
    #
    #     gene_means[partition[i],:] += data[times,:]
    #
    # for i in count:
    #
    #     gene_means[i,:] /= count[i]
    #
    # for times , i in enumerate(data):
    #
    #     temp=np.sum((i-gene_means)**2,axis=1)
    #
    #     result_prob[times,:] = 1/(1+temp)
    #
    # normallize_matrix(result_prob)
    #
    # result_prob *= prior_prob
    #
    # partition_new = deepcopy(partition)
    #
    # for times , i in enumerate(result_prob):
    #
    #     partition_new[bili_slic[times]] = np.argmax(i)
    


    # with open("/Users/rwang/Desktop/23/work/AM559-ST-010001/louvain_result.txt","w") as k:
    #     for i in range(data.shape[0]):
    #         k.write(str(slics["slic"+str(i)].mean_pos[0])+"\t"+str(slics["slic"+str(i)].mean_pos[1])+"\t"+str(partition[i])+"\n")
    
    
    with open(output_path+"/louvain_result_gauss.txt","w") as k:
        
        for i in partition:
            
            # try:
                
            k.write(i+"\t"+str(slics[i].mean_pos[0])+"\t"+str(slics[i].mean_pos[1])+"\t"+str(partition[i])+"\n")
            
    
            
    # with open(output_path+"/louvain_result_gauss.txt","w") as k:
    #
    #     for times,i in enumerate(bilinear_adata.obs.index):
    #
    #         # try:
    #
    #         k.write(str(slics[i].mean_pos[0])+"\t"+str(slics[i].mean_pos[1])+"\t"+str(bilinear_adata.obs["leiden"][i])+"\n")
            
            # except:
            #     continue
        
if __name__ =="__main__":

    output_path="/Users/rwang/Desktop/mydata/cancer/my_result/"
    
    # output_path="/Users/rwang/Desktop/23/work/osfstorage-archive/Slide_2/bilinear_interpolation_result/"
    
    get_cluster(output_path)
