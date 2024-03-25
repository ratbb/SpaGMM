#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:10:11 2022

@author: rwang
"""
import numpy as np
def img_clear(img, position_file,windows=20):
    position_file = position_file.readlines()
    img_zeros = np.zeros([img.shape[0], img.shape[1],3])
    for i in position_file:
        img_zeros[eval(i.split()[0]) - windows:eval(i.split()[0]) + windows, eval(i.split()[1]) - windows:eval(i.split()[1]) + windows,:] = (1,1,1)
    img=img_zeros * img
    img=np.where(img==(0,0,0),255-img,img)
    return np.array(img,dtype=np.uint8)