# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:45:12 2022

@author: Administrator
"""
import cv2 as cv
from sklearn.mixture import GaussianMixture
from image_clear import img_clear
import pickle
from unit import *
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns


def get_spatial_prob(temp_sum,beta=1,neighbour=15):
    prob=(temp_sum*beta+(temp_sum-neighbour)*beta)
    prob_nom=np.log(np.sum(np.exp(prob),axis=1))
    return np.exp(prob-prob_nom[:,np.newaxis])

def gauss_mixture(input_path,output_path,classify="5"):
    input_image = input_path + "/spatial/tissue_hires_image.png"
    c = eval(classify)
    img = cv.imread(input_image)     #读入图片
    # position_file=open(args.position,"r")
    # img=img_clear(img,position_file,windows=20)
    # slic = cv.ximgproc.createSuperpixelSLIC(img, region_size=eval(args.region_size), ruler=20.0)    # 初始化slic项，region_size设置分割图片尺寸大小 ruler设置平滑因子
    # slic.iterate(20)  # 设置迭代次数，迭代次数相对来说越大越
    # image_label=slic.getLabels()
    # mask_slic = slic.getLabelContourMask()
    # mask_inv_slic = cv.bitwise_not(mask_slic)
    # img_slic = cv.bitwise_and(img, img, mask=mask_inv_slic)
    # cv.imshow("temp", img_slic)
    # cv.waitKey()
    # print("slic完成，所花时间{}s".format(time.time()-start))
    # img_lab=cv.cvtColor(img, cv.COLOR_BGR2Lab)
    # unit_devide=dict()
    # unit_devide=creat_dict(unit_devide,image_label,img)#存入每个label所对应的pos坐标,每个value值为一个列表
    temp=[]
    color=[]
    # for i in range(np.max(image_label)+1):
    #     t=unit(label=i,pos=unit_devide[i])
    #     t.calculate_color(img)
    #     t.mean_pos()
    #     if np.mean(t.color)<220:
    #         temp.append(t)
    with open(output_path+"/slic.pickle","rb") as f:
        slics=pickle.load(f)
        
    x = []
    y = []
    for i in slics:
        temp.append(slics[i])
        pos_temp =np.mean(np.array(slics[i].pos),axis = 0)
        x.append(pos_temp[0])
        y.append(pos_temp[1])
        #color.append(slics[i].color)
        color.append(np.concatenate([slics[i].color,pos_temp]))
    im_X_=np.array(color).reshape(int(np.array(color).size/5) ,5)
    #print(im_X)
    im_X=preprocessing.scale(im_X_)
    GM = GaussianMixture(n_components=c,weights_init=np.array([1/c]*c),n_init=10,random_state=5)#进行gauss混合模型
    GM = GM.fit(im_X)
    im_label=GM.predict(im_X)
    proba=GM.predict_proba(im_X)#得到后验概率

    # im_pos=np.zeros((im_X.shape[0],2))
    # for times,i in enumerate(slics):
    #     im_pos[times,:]=slics[i].mean_pos
    # nbrs = NearestNeighbors(n_neighbors=neighbour, algorithm='auto').fit(im_pos)
    # distances, indices = nbrs.kneighbors(im_pos)
    # temp_sum=np.zeros([im_X.shape[0],c])
    # for index,i in enumerate(indices):
    #     for j in i:
    #         temp_sum[index,im_label[j]]+=1
    # spatial_prob=get_spatial_prob(temp_sum,beta= beta,neighbour= neighbour)#beta是平滑因子
    # smooth_prob=proba*spatial_prob
    
    # for times,i in enumerate(smooth_prob):
    #     smooth_prob[times,:]=i/np.sum(i)
    # im_label2=(proba*spatial_prob).argmax(axis=1)
    np.save(output_path+"/smooth_prob.npy",proba)
    print("分类完成")
    color=['#7A57D1','#FF731D','#004d61','#bc8420','#CF0A0A','#83FFE6','#0000A1','#fff568','#0080ff','#81C6E8','#385098','#ffb5ba','#EA047E','#B1AFFF','#425F57','#CFFF8D','#100720','#18978F','#F9CEEE','#7882A4', '#E900FF','#84DFFF','#B2EA70','#FED2AA','#49FF00','#14279B','#911F27','#00dffc']
    #sns.scatterplot(x=im_X[:,-1],y=im_X[:,-2],palette=[color[i] for i in im_label2],size =0.5)
    #dt = pd.DataFrame(im_X)
    plt.scatter( x=x,y=y,  c=[color[i] for i in im_label],s=0.3)
    plt.savefig(output_path+'/gmm_plot.png', dpi=600)
    plt.show()
    # img_zeros = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # # 看一下效果
    # m = [(255, 255, 0), (0, 255, 0), (0, 0, 255), (255,0, 0), (255, 255, 255),(255,0,255)]
    # for i in range(c):
    #     x_pos = []
    #     y_pos = []
    #     for j, k in enumerate(im_X):
    #         if im_label2[j] == i:
    #             for n in temp[j].pos:
    #                 x_pos.append(n[0])
    #                 y_pos.append(n[1])
    #     for n in range(len(x_pos)):
    #         img_zeros[x_pos[n]][y_pos[n]] = m[i]
    
    #     # plt.subplot(3, 3, i + 1)
    #     # plt.scatter(np.array(x_pos), np.array(y_pos),c=color[i] ,marker=".", s=0.1)
    

    # # plt.show()
    # cv.imshow("temp", img_zeros)
    # cv.waitKey()
    # cv.imwrite(output_path+"/gauss_mixture_"+classify+".png", img_zeros)
    with open(output_path+"/result_gaussian_"+classify+".txt","w") as f:
       for times,i in enumerate(temp):
           f.write(str(i.label)+"\t"+str(im_label[times]+1)+"\t"+str(i.mean_pos[0])+"\t"+str(i.mean_pos[1])+"\t")
           #str(proba[times][0])+"\t"+str(proba[times][1])+"\t",str(proba[times][2])+"\t",str(proba[times][3])+"\t",str(proba[times][4])+"\n"
           for j in range(int(classify)):
               f.write(str(proba[times][j])+"\t")
           f.write("\n")

    #im_label_new=[]
    # with open(args.output_file,"w") as f:
    #     for times,i in enumerate(temp):
    #         sum=0
    #         l_t=[]
    #         for j in GM.means_:
    #             a=(1+np.linalg.norm(j-i.color)**2)**-1
    #             l_t.append(a)
    #             sum+=a
    #         f.writelines([str(i.label)+"\t",str(im_label[times]+1)+"\t",str(i.mean_pos[0])+"\t",str(i.mean_pos[1])+"\t",
    #                       str(l_t[0]/sum)+"\t",
    #                       str(l_t[1]/sum)+"\t",
    #                       str(l_t[2]/sum)+"\t",
    #                       str(l_t[3]/sum)+"\t",
    #                       str(l_t[4]/sum)+"\n"])
    #         im_label_new.append(np.argmax(np.array(l_t)))







