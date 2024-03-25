#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 21:59:28 2022

@author: rwang
"""
import pickle
import cv2 as cv
import image_clear
import unit
import numpy as np
import scanpy as sc

        


def creat_dict(unit_devide,image_label,img):
    
    for i in range(np.max(image_label)+1):
        unit_devide[i]=[]
    for i in range(image_label.shape[0]):
        for j in range(image_label.shape[1]):
            unit_devide[image_label[i][j]]+=[(i,j)]
    return unit_devide

def slics(input_path,output_path,region_size,gauss_kernel=5,noise=200):
    
    image=input_path+"/spatial/tissue_hires_image.png"
    position=output_path + "/position.txt"
    img = cv.imread(image)#读入图片
    position_file=open(position,"r")#读入调整后的坐标
    img=image_clear.img_clear(img,position_file,windows=30)#去掉图片背景
    img = cv.GaussianBlur(img, (gauss_kernel, gauss_kernel), 0)
    cv.imwrite(output_path+"/hires_image_clear.png",img)#输出调整好的图片
    slic = cv.ximgproc.createSuperpixelSLIC(img, region_size=eval(region_size), ruler=20.0)#进行slic
    slic.iterate(10)  # 设置迭代次数，迭代次数相对来说越大越好
    image_label=slic.getLabels()#图像分割
    # np.save(output+'/img_label.npy', image_label) #存入备用

    mask_slic = slic.getLabelContourMask()
    mask_inv_slic = cv.bitwise_not(mask_slic)
    img_slic = cv.bitwise_and(img, img, mask=mask_inv_slic)
    cv.imwrite(output_path+"/hires_image_clear_slic.png",img_slic)
    unit_devide=dict()
    unit_devide=creat_dict(unit_devide,image_label,img)
    slics=dict()
    label_temp=0
    for i in range(np.max(image_label)+1):
        t=unit.unit(label=i,pos=unit_devide[i])
        t.calculate_color(img)
        t.mean_pos()
        if np.mean(t.color) < noise:
            slics["slic"+str(label_temp)]=t
            t.name="slic"+str(label_temp)
            label_temp+=1
    with open(output_path+"/slic.pickle","wb") as f:
        pickle.dump(slics,f)
        f.close()


# def slics_h5ad(input_path, output_path, region_size, gauss_kernel=5, noise=200):
#     image = input_path + "histology.tif"
#     adata = sc.read_h5ad(input_path+"sample.h5ad")
#     img = cv.imread(image)  # 读入图片
#     img = image_clear.img_clear(img, position_file, windows=30)  # 去掉图片背景
#     img = cv.GaussianBlur(img, (gauss_kernel, gauss_kernel), 0)
#     cv.imwrite(output_path + "/hires_image_clear.png", img)  # 输出调整好的图片
#     slic = cv.ximgproc.createSuperpixelSLIC(img, region_size=eval(region_size), ruler=20.0)  # 进行slic
#     slic.iterate(10)  # 设置迭代次数，迭代次数相对来说越大越好
#     image_label = slic.getLabels()  # 图像分割
#     # np.save(output+'/img_label.npy', image_label) #存入备用

#     mask_slic = slic.getLabelContourMask()
#     mask_inv_slic = cv.bitwise_not(mask_slic)
#     img_slic = cv.bitwise_and(img, img, mask=mask_inv_slic)
#     cv.imwrite(output_path + "/hires_image_clear_slic.png", img_slic)
#     unit_devide = dict()
#     unit_devide = creat_dict(unit_devide, image_label, img)
#     slics = dict()
#     label_temp = 0
#     for i in range(np.max(image_label) + 1):
#         t = unit.unit(label=i, pos=unit_devide[i])
#         t.calculate_color(img)
#         t.mean_pos()
#         if np.mean(t.color) < noise:
#             slics["slic" + str(label_temp)] = t
#             t.name = "slic" + str(label_temp)
#             label_temp += 1
#     with open(output_path + "/slic.pickle", "wb") as f:
#         pickle.dump(slics, f)
#         f.close()