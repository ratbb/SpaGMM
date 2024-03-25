#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:35:03 2022

@author: rwang
"""
import pickle
import seaborn as sns
import numpy as np
import cv2 as cv
import scanpy as sc
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from bilinear_interpolation import *
from unit import unit

class beed:  # 定义一个类beed，存入label和坐标信息
    def __init__(self, name, pos_x, pos_y , gene):
        self.pos = [pos_x, pos_y]
        #self.labels_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.gene=np.array(gene)
        self.name=name
def init__beed(adata,pos_beed):
    beeds = []
    pos_x=np.array(pos_beed)[:,0]
    pos_y=np.array(pos_beed)[:,1]
    gene=adata.to_df()
    for j,i in enumerate(adata.to_df().index.tolist()):
        beeds.append(beed(i, pos_x[j], pos_y[j], gene.loc[i].values))
    return beeds

def read_gene(path):
    adata=sc.read_visium(path)   #读取单细胞测序文件
    return adata


def get_slic_gene(input_path,output_path):
    # parser = argparse.ArgumentParser(description='i m lazy,i ll descirbe it latter，author——ray')
    # parser.add_argument('--gene', help='Path to input gene.', default='/Users/rwang/Desktop/23/work/AM559-ST-010001')
    # parser.add_argument('--output', help='Path to output filename.', default="/Users/rwang/Desktop/23/work/AM559-ST-010001/bilinear_interpolation_result")
    # parser.add_argument('--position', help='Path to input position.', default='/Users/rwang/Desktop/23/work/position/position.txt')
    # args = parser.parse_args()
    adata=read_gene(input_path) #获得原数据集
    adata.var_names_make_unique()
    #sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=5)
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    # sc.pp.highly_variable_genes(adata, min_mean=0.015, max_mean=4, min_disp=0.5 ,n_top_genes=2000)
    # # sc.pl.highly_variable_genes(adata)
    # # .raw.to_adata() get raw mutation
    # adata.raw = adata
    # adata = adata[:, adata.var.highly_variable]
    # sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    pos_beed = pd.read_csv(output_path+"/position.txt", sep="\t",header=None).values.tolist()
    beeds=init__beed(adata,pos_beed)
    smooth_prob=np.load(output_path+"/smooth_prob.npy")
    
    
    # pos = [i.pos for i in beeds]
    # #记录每个beed的位置，计算 nearest neighbour
    # nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(pos)
    # distances, indices = nbrs.kneighbors(pos)

    # for i in range(len(beeds)):
    #     beeds[i].gene = np.median(np.array([beeds[j].gene for j in indices[i][1:]]),axis=0)
        
        
    
    beeds_dict=dict() 
    for i in beeds:
        beeds_dict[i.name]=i

    # image_label=np.load(args.output+'/img_label.npy')
    with open(output_path+"/slic.pickle","rb") as f:
        slics=pickle.load(f)

    print("slics结果读入成功")
    for i in slics:#slics是字典
        temp=dict()
        for j in beeds:#beeds是列表
            # if tuple(j.pos) in slics[i].pos:
            #     slics[i].bool=1
            #     slics[i].gene=j.gene
            #     slics[i].name=j.name
            #     break
            temp[str(j.name)]=np.linalg.norm(slics[i].mean_pos - np.array(j.pos), ord=2)
        # if slics[i].bool==0:
        temp=sorted(temp.items(),key=lambda item:item[1])
        temp=[j for i,j in enumerate(temp) if j[1]<100 ]

        condition_a=condition_b=condition_c=condition_d=1
        for k in temp:
            if condition_a and beeds_dict[k[0]].pos[0] > slics[i].mean_pos[0] and beeds_dict[k[0]].pos[1] > slics[i].mean_pos[1]:
                point_a=point(beeds_dict[k[0]].pos[0],beeds_dict[k[0]].pos[1],beeds_dict[k[0]].gene)
                condition_a=0
            if condition_b and beeds_dict[k[0]].pos[0] < slics[i].mean_pos[0] and beeds_dict[k[0]].pos[1] > slics[i].mean_pos[1]:
                point_b=point(beeds_dict[k[0]].pos[0],beeds_dict[k[0]].pos[1],beeds_dict[k[0]].gene)
                condition_b=0
            if condition_c and beeds_dict[k[0]].pos[0] > slics[i].mean_pos[0] and beeds_dict[k[0]].pos[1] < slics[i].mean_pos[1]:
                point_c=point(beeds_dict[k[0]].pos[0],beeds_dict[k[0]].pos[1],beeds_dict[k[0]].gene)
                condition_c=0
            if condition_d and beeds_dict[k[0]].pos[0] < slics[i].mean_pos[0] and beeds_dict[k[0]].pos[1] < slics[i].mean_pos[1]:
                point_d=point(beeds_dict[k[0]].pos[0],beeds_dict[k[0]].pos[1],beeds_dict[k[0]].gene)
                condition_d=0
        if condition_a:
            point_a=point(slics[i].mean_pos[0]+10,slics[i].mean_pos[1]+10,0)
        if condition_b:
            point_b=point(slics[i].mean_pos[0]-10,slics[i].mean_pos[1]+10,0)
        if condition_c:
            point_c=point(slics[i].mean_pos[0]+10,slics[i].mean_pos[1]-10,0)
        if condition_d:
            point_d=point(slics[i].mean_pos[0]-10,slics[i].mean_pos[1]-10,0)
        point_p=point(slics[i].mean_pos[0],slics[i].mean_pos[1],0)
        slics[i].gene=np.array(bilinear_interpolation(point_a,point_b,point_c,point_d,point_p).value,dtype = "int")
        slics[i].bool=1
        slics[i].name=i

    print("slic基因获得完成")
    slics_name=[]
    print("开始写入文件")
    for time,i in enumerate(slics):
        slics_name.append(slics[i].name)
    gene_name=adata.to_df().columns
    data=pd.DataFrame(np.zeros([len(slics_name),len(gene_name)]),columns=gene_name,index=slics_name)
    for indexs,j in zip(slics_name,slics.keys()):
        data.loc[indexs].values[:]=np.array(slics[j].gene)
    data_array=np.array(data)
    np.save(output_path+'/gene_expression.npy', data_array)
    
    
    
    with open(output_path+"/barcodes.tsv",'w') as f:
        for time,i in enumerate(slics):
            f.write("slic"+str(time)+"\n")
            # slics_name.append(slics[i].name)
        f.close()
    print("写入barcodes文件")
    with open(output_path+"/features.tsv",'w') as f:
        for i in gene_name:
            f.write(i+"\t"+i+"\t"+"Gene Expression"+"\n")
        f.close()
    print("写入features文件")
    # data=pd.DataFrame(np.zeros([len(slics_name),len(gene_name)]),columns=gene_name,index=slics_name)
    # for indexs,j in zip(slics_name,slics.keys()):
    #     data.loc[indexs].values[:]=np.array(slics[j].gene)
    # data_array=np.array(data)
    
    # np.save(output_path+'/gene_expression.npy', data_array)
    # data_array=np.around(data_array)
    X=data_array.nonzero()
    
    print("写入mtx文件")
    with open(output_path+"/matrix.mtx",'w') as f:
        f.write("%%MatrixMarket matrix coordinate integer general\n"+'%metadata_json: {"software_version": "spaceranger-1.3.0", "format_version": 2}'+'\n')
        f.write(str(len(gene_name))+"\t"+str(len(slics_name))+"\t"+str(np.count_nonzero(data))+"\n")
        for i in range(len(X[0])):
            row=X[0][i]
            col=X[1][i]
            f.write(str(col+1)+" "+str(row+1)+" "+str(data_array[row,col])+"\n")
        f.close()
    print("写入position文件")
    with open(output_path+"/tissue_positions_slics_list.csv",'w') as f:
        for times,i in enumerate(slics):
            f.write("slic"+str(times)+","+str(1)+","+str(slics[i].mean_pos[0])+","+str(slics[i].mean_pos[1])+","+str(slics[i].mean_pos[0])+","+str(slics[i].mean_pos[1])+"\n")
        f.close()
    return adata
    # im=np.zeros([2000,2000])
    # for indexs in slics_name:
    #     value=data.loc[indexs].loc["PAXXG182380"]
    #     for k in slics[indexs].pos:
    #         im[int(k[0]),int(k[1])]=value
    # sns.heatmap(im)
    # with open(output_path+"/PAXXG070630_gene.txt",'w') as f:
    #     for indexs in slics_name:
    #         value=data.loc[indexs].loc["PAXXG070630"]
    #         f.write(str(indexs)+"\t"+str(slics[indexs].mean_pos[0])+"\t"+str(slics[indexs].mean_pos[1])+"\t"+str(value)+"\n")
        
        
    