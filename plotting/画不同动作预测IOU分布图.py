import numpy as np
import random
import os
from pathlib import Path

import matplotlib.pyplot as plt


json_data_path = Path(r"E:\实验室个人代码备份20230417\graduation\npy_data_with_skeleton")
# 提取所有雷达的标签
data_path=os.path.join(json_data_path,'total_all_labels.npy')
add_obliquity=False
data={'All':[]}

import pandas as pd

for i in os.listdir(r'E:\实验室个人代码备份20230417\graduation\DETR_0505_movement\savefig'):
    move=i.split('_')[0]
    iou=int(i.split('IOU')[-1].replace('.jpg',''))/1000
    if move not in data.keys():data[move]=[iou]
    else:data[move].append(iou)
    data['All'].append(iou)


plt.style.use('ggplot')
#处理中文乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
#坐标轴负号的处理
plt.rcParams['axes.unicode_minus']=False

i=0
# df = pd.DataFrame(data, columns=['Static','Walk','Trot','Sit Down','Stand Up','Random Action'])
fig,ax=plt.subplots()
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
h1=ax.violinplot(data['Static'],positions=[1],showmeans=True,showextrema=True,showmedians=False,quantiles=[0.25,0.75])
h2=ax.violinplot(data['Walk'],positions=[2],showmeans=True,showextrema=True,showmedians=False,quantiles=[0.25,0.75])
h3=ax.violinplot(data['Trot'],positions=[3],showmeans=True,showextrema=True,showmedians=False,quantiles=[0.25,0.75])
h4=ax.violinplot(data['Sit Down'],positions=[4],showmeans=True,showextrema=True,showmedians=False,quantiles=[0.25,0.75])
h5=ax.violinplot(data['Stand Up'],positions=[5],showmeans=True,showextrema=True,showmedians=False,quantiles=[0.25,0.75])
h6=ax.violinplot(data['Random Action'],positions=[6],showmeans=True,showextrema=True,showmedians=False,quantiles=[0.25,0.75])
h7=ax.violinplot(data['All'],positions=[7],showmeans=True,showextrema=True,showmedians=False,quantiles=[0.25,0.75])
h1_color=h1['bodies'][0].get_facecolor().flatten()
# h3_color=h3['bodies'][0].get_facecolor().flatten()
# h4_color=h4['bodies'][0].get_facecolor().flatten()

for key,value in data.items():
    print(key,sum(data[key])/len(data[key]))


import matplotlib.patches as mpatches
h1_patch = mpatches.Patch(color=h1_color)
h2_patch = mpatches.Patch(color=h1_color)
h3_patch = mpatches.Patch(color=h1_color)

h4_patch = mpatches.Patch(color=h1_color)

h5_patch = mpatches.Patch(color=h1_color)

h6_patch = mpatches.Patch(color=h1_color)
h7_patch = mpatches.Patch(color=h1_color)
# h3_patch = mpatches.Patch(color=h3_color, label='3-D Size of the Bbox')
# h4_patch = mpatches.Patch(color=h4_color, label='Obliquity of the Bbox(in range [0,π])')


# ax.legend(handles=[h1_patch,h2_patch,h3_patch,h4_patch,h5_patch,h6_patch],loc=4)
plt.xticks(range(1,8,1), labels=['Static','Walk','Trot','Sit Down','Stand Up','Random Action','All'],rotation=10)
plt.ylabel('IOU of Prediction and GT')
plt.xlabel('Data Movement Type')
plt.savefig('不同动作预测IOU分布图.jpg',dpi=600, bbox_inches='tight')
