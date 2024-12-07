import numpy as np
import random
import os
from pathlib import Path
import json
import seaborn as sns
import matplotlib.pyplot as plt
import math
import torch

json_data_path = Path(r"E:\实验室个人代码备份20230417\graduation\npy_data_with_skeleton")
# 提取所有雷达的标签
data_path=os.path.join(json_data_path,'total_all_labels.npy')
add_obliquity=False
data=[]

print('Reading already exist npy files')
all_labels=np.load(data_path, allow_pickle='TRUE')
all_images=np.load(data_path.replace('_all_labels.npy','_all_images.npy'), allow_pickle='TRUE')
print('add obliquity')
for i in range(len(all_labels)): 
    all_labels[i]['kpts']=np.append(all_labels[i]['kpts'],all_labels[i]['obliquity'])
    del all_labels[i]['obliquity']
print('Done with reading npy files')

import pandas as pd
print(len(all_labels))
# self.f_dict={'静止':[0],'走路':[1],'小跑':[2],'坐下':[3],'站起':[4],'随机动作':[5]}
plt.style.use('ggplot')
#处理中文乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
#坐标轴负号的处理
plt.rcParams['axes.unicode_minus']=False
for i in range(len(all_labels)): 
    if all_labels[i]['movement'] in [0,3,4,5]:
        data.append(list(all_labels[i]['kpts']))
data=np.array(data)
for i in range(len(data)):
    data[i][3]*=2
    data[i][4]*=2
    data[i][5]*=2
i=0
df = pd.DataFrame(data, columns=['X_center','Y_center','Z_center','X_length','Y_length','Z_length','obliquity'])
fig,ax=plt.subplots()
print(df['obliquity'].describe())
# df1=pd.DataFrame(data=None,columns=['data','type'])
# for i in range(len(data)):
#     df1.loc[len(df1)]=[data[i][0],'X_center']
#     df1.loc[len(df1)]=[data[i][1],'Y_center']
#     df1.loc[len(df1)]=[data[i][2],'Z_center']
#     df1.loc[len(df1)]=[data[i][3],'X_length']
#     df1.loc[len(df1)]=[data[i][4],'Y_length']
#     df1.loc[len(df1)]=[data[i][5],'Z_length']
#     df1.loc[len(df1)]=[data[i][6],'obliquity']
# print(df1)
# h12=ax.violinplot([data[:,0],data[:,1],data[:,2]],[1,2,3],showmeans=True,showmedians=True)
# h3=ax.violinplot([data[:,3],data[:,4],data[:,5]],positions=[4,5,6],showmeans=True,showmedians=True)
# h4=ax.violinplot(data[:,6],positions=[7],showmeans=True,showmedians=True)
h12=ax.violinplot(df[['X_center','Y_center','Z_center']],positions=[1,2,3],showmeans=True,showextrema=True,showmedians=False,quantiles=[[0.25,0.75],[0.25,0.75],[0.25,0.75]])
h3=ax.violinplot(df[['X_length','Y_length','Z_length']],positions=[4,5,6],showmeans=True,showextrema=True,showmedians=False,quantiles=[[0.25,0.75],[0.25,0.75],[0.25,0.75]])
h4=ax.violinplot(df[['obliquity']],positions=[7],showmeans=True,showextrema=True,showmedians=False,quantiles=[0.25,0.75])

h12_color=h12['bodies'][0].get_facecolor().flatten()
h3_color=h3['bodies'][0].get_facecolor().flatten()
h4_color=h4['bodies'][0].get_facecolor().flatten()


import matplotlib.patches as mpatches
h12_patch = mpatches.Patch(color=h12_color, label='3-D Coordinates of the Bbox Center')
h3_patch = mpatches.Patch(color=h3_color, label='3-D Size of the Bbox')
h4_patch = mpatches.Patch(color=h4_color, label='Obliquity of the Bbox(in range [0,π])')


ax.legend(handles=[h12_patch,h3_patch,h4_patch],loc=4)
plt.xticks(range(1,8,1), labels=['X_Center','Y_Center','Z_Center','X_Size','Y_Size','Z_Size','Obliquity'])
plt.ylabel('Value')
plt.xlabel('Data Type')
plt.savefig('除去走路、小跑BBOX数据分布图.jpg',dpi=600, bbox_inches='tight')
