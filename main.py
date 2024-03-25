#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 16:02:53 2022

@author: rwang
"""
import time

import slics

import beeds

import gauss_mixture_rgb_slic

import position_rectify

import sc_analysis

import argparse

import mytest__louvain

def main():
    parser = argparse.ArgumentParser(description='no description ，author——ray')
    
    parser.add_argument('--input', help='Path to input gene.', default= "/Users/rwang/Desktop/mydata/Mouse_brain_10X/")
    
    parser.add_argument('--output', help='Path to output filename.', default="/Users/rwang/Desktop/mydata/Mouse_brain_10X/my_result")
    
    parser.add_argument('--region_size', help='region_size', default='10')
    
    parser.add_argument('--classify', help='classify', default='8')
    
    parser.add_argument('--mode', help='picture mode', default="2")
    
    parser.add_argument('--times', help='times', default="100")

    parser.add_argument('--picture', help='待输入的图片', default="Error")
    
    args = parser.parse_args()#获得参数的输入


    if args.input.split(".")[-1] != "h5ad" :
    
        position_rectify.position_rectify(args.input, args.output,times=args.times)
    
        slics.slics(args.input, args.output, args.region_size, gauss_kernel=11, noise=170)
    
        gauss_mixture_rgb_slic.gauss_mixture(args.input, args.output,args.classify)
    
        adata = beeds.get_slic_gene(args.input, args.output)
    
        bilinear_adata = sc_analysis.get_h5adfile(adata,args.output,classify=args.classify)
        
        mytest__louvain.get_cluster(args.output,resolution=3,classify=int(args.classify))
        
    # if args.input.split(".")[-1] == "h5ad" and args.picture != "Error":#如果是一个h5ad格式的文件

    #     position_rectify.position_rectify(args.input, args.output, times=args.times)
    

    #     slics.slics(args.input, args.output, args.region_size, gauss_kernel=5, noise=200)

    # else:

    #     print("格式有误，请检查是否输入了正确的h5ad文件和图片")

if __name__ =="__main__":
    
    start=time.time()
    
    main()
    
    print("总共运行了{}".format(time.time()-start))

