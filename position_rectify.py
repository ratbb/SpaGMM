#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 16:51:51 2021

@author: ruiwang
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import argparse
import json
import scanpy as sc
import pandas as pd 

def rect_score(x,y,img,image_width):#计算子方框的得分
    length = int(image_width/2)
    return  np.sum(img[x-length:x+length+1,y-length:y+length+1])
def data_compute(x,y,img,image_width):
    temp=0
    for i in range(len(x)):
        temp+=rect_score(x[i],y[i],img,image_width)
    return temp
def position_rectify(input_path,output_path,times='100',mode="2",image_width=20):
    adata=sc.read_visium(input_path)    #读取单细胞测序文件
    input_image=input_path+"/spatial/tissue_hires_image.png"
    transform=input_path+"/spatial/scalefactors_json.json"
    times=eval(times)
    img=cv.imread(input_image,cv.IMREAD_GRAYSCALE)#读入图片
    file = open(transform, 'r')
    js = file.read()
    dic = json.loads(js)
    if mode=="0":
        trans=1
    elif mode=="1":
        trans= dic['tissue_lowres_scalef']
    elif mode=="2":
        trans= dic['tissue_hires_scalef']
    #像素坐标转换
    
    pos = adata.obsm["spatial"]
    x_pos=np.array([float(i[1]) for i in pos])*trans
    y_pos=np.array([float(i[0]) for i in pos])*trans

    x_pos=np.array(list(map(int,x_pos)))
    y_pos=np.array(list(map(int,y_pos)))
    



    score=0
    scores=[0,0,0,0,0]
    score_set=[]
    for i in range(times):

        scores[0]=data_compute(x_pos+1,y_pos,img,image_width)
        scores[1]=data_compute(x_pos,y_pos+1,img,image_width)
        scores[2]=data_compute(x_pos-1,y_pos,img,image_width)
        scores[3]=data_compute(x_pos,y_pos-1,img,image_width)
        scores[4]=data_compute(x_pos,y_pos,img,image_width)

        score=np.min(scores)
        score_set.append(score)
        if scores.index(score)==0:
            x_pos,y_pos=x_pos+1,y_pos
        if scores.index(score)==1:
            x_pos,y_pos=x_pos,y_pos+1
        if scores.index(score)==2:
            x_pos,y_pos=x_pos-1,y_pos
        if scores.index(score)==3:
            x_pos,y_pos=x_pos,y_pos-1
        if scores.index(score)==4:
            x_pos,y_pos=x_pos,y_pos
        print(i/times)
    with open(output_path+"/position.txt","w") as f :
        for i,j in zip(x_pos,y_pos):
            f.write(str(i)+"\t")
            f.write(str(j)+"\n")
        f.close()

    img=cv.imread(input_image)
    for i in zip(x_pos,y_pos):
        for m in range(-4,3):
            for n in range(-4,3):
                img[i[0]+m,i[1]+n]=0,0,0
    # cv.imshow("t",img)
    cv.imwrite(output_path+"/barcode_position.png",img)
    # cv.waitKey()
    # plt.plot(score_set)
    # plt.show()
    print("图像校正完成")