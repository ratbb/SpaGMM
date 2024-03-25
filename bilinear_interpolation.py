#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:18:02 2022

@author: rwang
"""
import numpy as np
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
        return point_temp,a2,a1
    elif condition>0:
        if point_x.x_pos< point_y.x_pos:
            temp=point_x
            point_x=point_y
            point_y=temp
            change =1 #交换过位置
        if (point_x.x_pos-x)<0:
            a2=1-x_distance/abs((point_y.x_pos)-x)
            a1=1-a2
            point_temp.y_pos=(1/a1)*point_x.y_pos-(a2/a1)*point_y.y_pos
            point_temp.value=(1/a1)*point_x.value-(a2/a1)*point_y.value
            if change:
                return point_temp,-(a2/a1),(1/a1)
            else:
                return point_temp,(1/a1),-(a2/a1)
        if (point_y.x_pos-x)>0:
            a2=1-x_distance/abs((point_x.x_pos)-x)
            a1=1-a2
            point_temp.y_pos=(1/a1)*point_y.y_pos-(a2/a1)*point_x.y_pos
            point_temp.value=(1/a1)*point_y.value-(a2/a1)*point_x.value
            if not change:
                return point_temp,-(a2/a1),(1/a1)
            else:
                return point_temp,(1/a1),-(a2/a1)
    # elif condition==0:
    #     if int(point_x.x_pos-x)==0:
    #         return point_x
    #     elif int(point_y.x_pos-x)==0:
    #         return point_y
    #     else:
    #         point_temp.y_pos=(point_y.y_pos+point_x.y_pos)/2
    #         point_temp.value=(point_y.value+point_x.value)/2
    #return point_temp
    
def interpolation_y(point_x,point_y,y):
    condition=(point_x.y_pos-y)*(point_y.y_pos-y)
    point_temp=point(0,y,0)
    y_distance=abs(point_x.y_pos-point_y.y_pos)
    if condition<0:
        point_temp.x_pos=abs(point_x.y_pos-y)/y_distance*point_y.x_pos+abs(point_y.y_pos-y)/y_distance*point_x.x_pos
        point_temp.value=abs(point_x.y_pos-y)/y_distance*point_y.value+abs(point_y.y_pos-y)/y_distance*point_x.value
        return point_temp,abs(point_y.y_pos-y)/y_distance,abs(point_x.y_pos-y)/y_distance
    elif condition>0:
        if point_x.y_pos< point_y.y_pos:
            temp=point_x
            point_x=point_y
            point_y=temp
            change =1 
        if (point_x.y_pos-y)<0:
            a2=1-y_distance/abs((point_y.y_pos)-y)
            a1=1-a2
            point_temp.x_pos=(1/a1)*point_x.x_pos-(a2/a1)*point_y.x_pos
            point_temp.value=(1/a1)*point_x.value-(a2/a1)*point_y.value
            if change:
                return point_temp,-(a2/a1),(1/a1)
            else:
                return point_temp,(1/a1),-(a2/a1)
        if (point_y.y_pos-y)>0:
            a2=1-y_distance/abs((point_x.y_pos)-y)
            a1=1-a2
            point_temp.x_pos=(1/a1)*point_y.x_pos-(a2/a1)*point_x.x_pos
            point_temp.value=(1/a1)*point_y.value-(a2/a1)*point_x.value
            if not change:
                return point_temp,-(a2/a1),(1/a1)
            else:
                return point_temp,(1/a1),-(a2/a1)
    # elif condition==0:
    #     if int(point_x.y_pos-y)==0:
    #         return point_x
    #     elif int(point_y.y_pos-y)==0:
    #         return point_y
    #     else:
    #         point_temp.x_pos=(point_y.x_pos+point_x.y_pos)/2
    #         point_temp.value=(point_y.value+point_x.value)/2
    #return point_temp
    
def bilinear_interpolation(point_a,point_b,point_c,point_d,point_p):
    temp_1,w1,w2=interpolation_x(point_a,point_b,point_p.x_pos)
    #print((temp_1.value == w1*point_a.value+w2*point_b.value).all())
    temp_2,w3,w4=interpolation_x(point_c,point_d,point_p.x_pos)
    #print((temp_2.value == w3*point_c.value+w4*point_d.value).all())
    temp_final,w5,w6=interpolation_y(temp_1,temp_2,point_p.y_pos)
    #print((temp_final.value == w5*temp_1.value+w6*temp_2.value).all())
    #print(w1*w5*point_a.value+w2*w5*point_b.value+w3*w6*point_c.value+w4*w6*point_d.value)
    if int(int(np.linalg.norm((w5*w1*point_a.value+w5*w2*point_b.value)+(w6*w3*point_c.value+w6*w4*point_d.value)-temp_final.value,ord=1))):
        print("error")
        
    wa ,wb,wc,wd = w5*w1,w5*w2,w6*w3,w6*w4
    
    
    # print((w5*(w1*point_a.value+w2*point_b.value)+w6*(w3*point_c.value+w4*point_d.value)==temp_final.value).all())
    #print((np.array((w5*w1*point_a.value+w5*w2*point_b.value)+(w6*w3*point_c.value+w6*w4*point_d.value),dtype =np.float64)==np.array(temp_final.value,dtype=np.float64)).all())
    # print((w5*w1*point_a.value+w5*w2*point_b.value)+(w6*w3*point_c.value+w6*w4*point_d.value))
        

    return temp_final
