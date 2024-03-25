#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:30:09 2022

@author: ruiwang
"""
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
import cv2 as cv
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import argparse
import time

def read_gene(path):
    adata=sc.read_visium(path)    #读取单细胞测序文件
    return adata

# def anylasis(adata):
#     pos_beed = np.array(adata.values.tolist())  # 读入处理后的坐标文件
#     sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
#     sc.logging.print_header()
#     sc.settings.set_figure_params(dpi=80, facecolor='white')
#     results_file = 'write/pbmc3k.h5ad'  # the file that will store the analysis results
#     adata=sc.read_visium('/Users/rwang/Desktop/23/work/AM559-ST-010001/')    #读取单细胞测序文件
#     adata.obsm["spatial_hires"]=np.array(pos_beed)
#     #sc.pp.filter_cells(adata, min_genes=200)#过滤至每个cell至少有200个gene表达
#     #sc.pp.filter_genes(adata, min_cells=3)#每个基因至少在3个细胞中表达
#     adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
#     sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
#     # sc.pl.violin(adata, ['n_genes_by_counts','total_counts'],jitter=0.4, multi_panel=True)                  #可视化
#     # sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')#可视
#     #adata = adata[adata.obs.n_genes_by_counts < 9000, :]
#     #adata = adata[adata.obs.total_counts < 60000, :]
#     #adata = adata[adata.obs.pct_counts_mt < 5, :]
#     sc.pp.normalize_total(adata, target_sum=1e4)
#     sc.pp.log1p(adata)
#     sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5,n_top_genes=2000)
#     # sc.pl.highly_variable_genes(adata)
#     adata.raw = adata
#     adata = adata[:, adata.var.highly_variable]
#     sc.pp.regress_out(adata, ['total_counts'])
#     sc.pp.scale(adata, max_value=10)
#     sc.tl.pca(adata, svd_solver='arpack',n_comps=15)
#     # sc.pl.pca_variance_ratio(adata, log=True)
#     return adata
# #calculate distance

# def distance():
#     adata=anylasis()
#     distance_matrix=np.zeros((adata.obsm["X_pca"].shape[0],adata.obsm["X_pca"].shape[0]))

#     for times1,i in enumerate(adata.obsm["X_pca"]):
#         sum=0
#         for times2,j in enumerate(adata.obsm["X_pca"]):
#             temp=np.exp(-np.sum((i-j)**2)/(2*(35)**2))
#             distance_matrix[times1][times2]=temp
#             sum+=temp
#             print(times1,times2)
#         distance_matrix[times1:]/=sum
#     # np.savetxt('distance_matrix.txt',distance_matrix,fmt='%.6f',delimiter=',')
#     return distance_matrix

def adjusted_distance():
    distance_matrix = np.loadtxt("/Users/rwang/Desktop/23/work/distance_matrix.txt", dtype=float, delimiter=",")
    # distance_matrix=distance()
    distance_matrix_new=np.zeros(distance_matrix.shape)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[0]):
            distance_matrix_new[i,j]=(distance_matrix[i,j]+distance_matrix[j,i])/(2*distance_matrix.shape[0])

    for i,j in enumerate(distance_matrix_new):
       distance_matrix_new[i]=distance_matrix_new[i]/np.sum(j)
    return distance_matrix_new


    
# def calculate_label(pos_beed, pos_slic):
#     beeds = []  # 一个存入beed的列表
#     # 计算每个beed的labels
#     for i in pos_beed:
#         temp = beed(i[0], i[1])
#         sum = 0
#         for j in pos_slic:
#             if j[1] < i[0] + 12 and j[1] > i[0] - 12 and j[2] < i[1] + 12 and j[2] > i[1] - 12:
#                 if j[0] in temp.labels_dict:
#                     temp.labels_dict[j[0]] += 1
#                     sum += 1
# #        if sum:
#         beeds.append(temp)
#     return beeds


# def similarity(beed1, beed2 ,C):
#     union = 0
#     intersection = 0
#     for i in range(5):
#         union += max(beed1.labels_dict[i], beed2.labels_dict[i])
#         intersection += min(beed1.labels_dict[i], beed2.labels_dict[i])
#     # if union == 0:
#     #     return 0
#     return (intersection+C) / (union+C)


# def calculate_similarity(beeds,C):
#     similarity_matrix = np.ones([len(beeds), len(beeds)])
#     for times_1 in range(len(beeds)):
#         for times_2 in range(times_1 + 1, len(beeds)):
#             similarity_matrix[times_1, times_2] = similarity(beeds[times_1], beeds[times_2],C)
#             similarity_matrix[times_2, times_1] = similarity(beeds[times_1], beeds[times_2],C)
#     return similarity_matrix


def normallize(similarity_matrix):
    for i in range(similarity_matrix.shape[0]):
        similarity_matrix[i, :] /= np.sum(similarity_matrix[i, :])
    return similarity_matrix
class unit:
    def __init__(self,label,pos):
        self.label=label  #该小单元的标签
        self.pos=pos #该小单元包含哪些像素
        self.bool=0 #是否包含beeds的坐标
        self.gene=np.array(0) #gene的值
        self.name=0 #名字

    def calculate_color(self,img):
        color_list=[]
        for i in self.pos:
            color_list+=[img[i]]
        color=np.array(color_list)
        self.color = np.mean(color.reshape(int(color.size / 3), 3), axis=0)

    def mean_pos(self):
        self.mean_pos=np.mean(np.array(self.pos).reshape(int(np.array(self.pos).size/2),2),axis=0)


def creat_dict(unit_devide,image_label,img):
    for i in range(np.max(image_label)+1):
        unit_devide[i]=[]
    for i in range(image_label.shape[0]):
        for j in range(image_label.shape[1]):
            unit_devide[image_label[i][j]]+=[(i,j)]
    return unit_devide

class point:
    def __init__(self,x_pos,y_pos,value):
        self.x_pos=x_pos
        self.y_pos=y_pos
        self.value=value

def interpolation_x(point_x,point_y,x):
    condition=(point_x.x_pos-x)*(point_y.x_pos-x)
    point_temp=point(x,0,0)
    x_distance=abs(point_x.x_pos-point_y.x_pos)
    if condition<0:
        a2=1-abs(point_x.x_pos-x)/x_distance
        a1=1-a2
        point_temp.y_pos=a2*point_x.y_pos+a1*point_y.y_pos
        point_temp.value=a2*point_x.value+a1*point_y.value
    elif condition>0:
        if point_x.x_pos< point_y.x_pos:
            temp=point_x
            point_x=point_y
            point_y=temp
        if (point_x.x_pos-x)<0:
            a2=1-x_distance/abs((point_y.x_pos)-x)
            a1=1-a2
            point_temp.y_pos=(1/a1)*point_x.y_pos-(a2/a1)*point_y.y_pos
            point_temp.value=(1/a1)*point_x.value-(a2/a1)*point_y.value
        if (point_y.x_pos-x)>0:
            a2=1-x_distance/abs((point_x.x_pos)-x)
            a1=1-a2
            point_temp.y_pos=(1/a1)*point_y.y_pos-(a2/a1)*point_x.y_pos
            point_temp.value=(1/a1)*point_y.value-(a2/a1)*point_x.value
    # elif condition==0:
    #     if int(point_x.x_pos-x)==0:
    #         return point_x
    #     elif int(point_y.x_pos-x)==0:
    #         return point_y
    #     else:
    #         point_temp.y_pos=(point_y.y_pos+point_x.y_pos)/2
    #         point_temp.value=(point_y.value+point_x.value)/2
    return point_temp
    
def interpolation_y(point_x,point_y,y):
    condition=(point_x.y_pos-y)*(point_y.y_pos-y)
    point_temp=point(0,y,0)
    y_distance=abs(point_x.y_pos-point_y.y_pos)
    if condition<0:
        point_temp.x_pos=abs(point_x.y_pos-y)/y_distance*point_y.x_pos+abs(point_y.y_pos-y)/y_distance*point_x.x_pos
        point_temp.value=abs(point_x.y_pos-y)/y_distance*point_y.value+abs(point_y.y_pos-y)/y_distance*point_x.value
    elif condition>0:
        if point_x.y_pos< point_y.y_pos:
            temp=point_x
            point_x=point_y
            point_y=temp
        if (point_x.y_pos-y)<0:
            a2=1-y_distance/abs((point_y.y_pos)-y)
            a1=1-a2
            point_temp.x_pos=(1/a1)*point_x.x_pos-(a2/a1)*point_y.x_pos
            point_temp.value=(1/a1)*point_x.value-(a2/a1)*point_y.value
        if (point_y.y_pos-y)>0:
            a2=1-y_distance/abs((point_x.y_pos)-y)
            a1=1-a2
            point_temp.x_pos=(1/a1)*point_y.x_pos-(a2/a1)*point_x.x_pos
            point_temp.value=(1/a1)*point_y.value-(a2/a1)*point_x.value
    # elif condition==0:
    #     if int(point_x.y_pos-y)==0:
    #         return point_x
    #     elif int(point_y.y_pos-y)==0:
    #         return point_y
    #     else:
    #         point_temp.x_pos=(point_y.x_pos+point_x.y_pos)/2
    #         point_temp.value=(point_y.value+point_x.value)/2
    return point_temp
    
def bilinear_interpolation(point_a,point_b,point_c,point_d,point_p):
    temp_1=interpolation_x(point_a,point_b,point_p.x_pos)
    temp_2=interpolation_x(point_c,point_d,point_p.x_pos)
    temp_final=interpolation_y(temp_1,temp_2,point_p.y_pos)
    return temp_final





