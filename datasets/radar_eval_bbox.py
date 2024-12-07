# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import numpy as np
import torch
import math
import torch.nn.functional as F
import os 
import matplotlib.pyplot as plt

def iou_3d(box1, box2):

    #box [x1,y1,z1,x2,y2,z2]   分别是左上、右下的坐标（对角）

    area1 = (box1[0]-box1[3])*(box1[1]-box1[4])*(box1[5]-box1[2])
    area2 = (box2[0]-box2[3])*(box2[1]-box2[4])*(box2[5]-box2[2])
    area_sum = area1 + area2

    #计算重叠部分 设重叠box坐标为 [x1,y1,z1,x2,y2,z2]
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    z1 = max(box1[2], box2[2])
    x2 = max(box1[3], box2[3])
    y2 = max(box1[4], box2[4])
    z2 = min(box1[5], box2[5])
    if x1 <= x2 or y1 <= y2 or z1 >= z2:
        return 0
    else:
        inter_area = (x1-x2)*(y1-y2)*(z2-z1)

    return inter_area/(area_sum-inter_area)

def cxyzlwh_to_xyzxyz(outputs_dict):
    out=torch.stack((outputs_dict[:,:,0]+outputs_dict[:,:,3],outputs_dict[:,:,1]+outputs_dict[:,:,4],outputs_dict[:,:,2]-outputs_dict[:,:,5],
    outputs_dict[:,:,0]-outputs_dict[:,:,3],outputs_dict[:,:,1]-outputs_dict[:,:,4],outputs_dict[:,:,2]+outputs_dict[:,:,5]),2)
    return out

def plotting(output, target, target_boxes, IOU, pred_move, gt_move, obli):
    '''
    生成输出示意图        
    '''    
    IOU=("%.3f"%IOU)
    gt_bbox=target_boxes
    pred=output.detach().cpu().numpy()
    obli=obli.detach().cpu().numpy()
    gt=target

    index=str(len(os.listdir(os.path.join(os.getcwd(),'savefig')))+1)
    # if IOU>max(self.best_iou.keys()):       # 只有IOU有增加时画图
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
    # import matplotlib;matplotlib.use('tkagg')
    plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
    plt.switch_backend('agg')

    # 绘制散点图
    fig = plt.figure(dpi=300)
    ax = fig.add_axes(Axes3D(fig)) 
    # ax = Axes3D(fig)
    kwargs={0:{'linewidth':1, 'color':'red', 'linestyle':'--'}, 1:{'linewidth':1, 'color':'green', 'linestyle':'--',}, 2:{'linewidth':1, 'color':'blue', 'linestyle':'-'}}
    # body_list = ["Head","Neck","SpineShoulder","SpineMid","SpineBase","ShoulderRight","ElbowRight","WristRight",
    #              "ShoulderLeft","ElbowLeft","WristLeft","HipRight","KneeRight","AnkleRight","HipLeft","KneeLeft","AnkleLeft"]
    # 画人的真实姿态
    point_to_box = np.load('point_to_box.npy', allow_pickle='TRUE').item()
    for key,value in point_to_box.items():
        a=value[1]
        if (value[1]==gt_bbox[:6]).all():
            body_list=value[0][:]
            temp=np.concatenate((body_list[0][None,:], body_list[1][None,:], body_list[2][None,:],body_list[3][None,:],body_list[4][None,:],body_list[11][None,:],body_list[12][None,:],body_list[13][None,:]), axis=0)
            ax.plot3D(temp[:,0], temp[:,1], temp[:,2], **kwargs[2])
            
            temp=np.concatenate((body_list[4][None,:],body_list[14][None,:],body_list[15][None,:],body_list[16][None,:]), axis=0)
            ax.plot3D(temp[:,0], temp[:,1], temp[:,2], **kwargs[2])   
                        
            temp=body_list[0]
            temp=np.concatenate((body_list[7][None,:], body_list[6][None,:],body_list[5][None,:],body_list[2][None,:],body_list[8][None,:],body_list[9][None,:],body_list[10][None,:]), axis=0)
            ax.plot3D(temp[:,0], temp[:,1], temp[:,2], **kwargs[2])    
            # obliquity = np.arctan((body_list[2][1]-body_list[4][1])/(body_list[2][0]-body_list[4][0]))  # [-pi/2,pi/2] 用脊柱算倾角
            # obliquity = obliquity if obliquity > 0 else obliquity+np.pi     # 主视图倾角，范围 [0,pi]
            # lossobli=abs(obliquity-obli)
            break
    # 画两个整体框
    any={0:pred,1:gt}
    label_map={0:'Pred Bbox of Move: "'+pred_move+'" with IOU: '+str(IOU),1:'GT Bbox of Move: "'+gt_move+'"'}
    for i in range(2):
        x=any[i][0]
        y=any[i][1]
        z=any[i][5]
        dx=any[i][3]-any[i][0]
        dy=any[i][4]-any[i][1]
        dz=any[i][2]-any[i][5]
        
        xx = [x, x+dx, x+dx, x, x]
        yy = [y+dy, y+dy, y, y, y+dy]
        ax.plot3D(xx, yy, [z+dz]*5, **kwargs[i])
        ax.plot3D(xx[:3], yy[:3], [z]*3, **kwargs[i])
        ax.plot3D(xx[2:], yy[2:], [z]*3, **kwargs[i])
        for n in range(3):
            ax.plot3D([xx[n], xx[n]], [yy[n], yy[n]], [z, z+dz], **kwargs[i])
        ax.plot3D([xx[3], xx[3]], [yy[3], yy[3]], [z, z+dz], label=label_map[i], **kwargs[i])
        # 画中心点区别图
        if not i:
            # 预测框中心点
            tmp=np.array((x+0.5*dx,y+0.5*dy,z+0.5*dz))
            # ax.scatter3D(x+0.5*dx,y+0.5*dy,z+0.5*dz,s=10,color='gray')
            # 画倾角预测情况
            r=0.35
            ax.quiver(x+0.5*dx,y+0.5*dy,z+0.5*dz,r*math.cos(obli),r*math.sin(obli),0,**{'linewidth':1, 'color':'red', 'linestyle':'--'})
        else:
            loca=np.concatenate((tmp[None,:], np.array((x+0.5*dx,y+0.5*dy,z+0.5*dz))[None,:]), axis=0)
            ax.plot3D(loca[:,0],loca[:,1],loca[:,2],color='gray',linewidth=0.7,linestyle='--')
            ax.scatter3D(x+0.5*dx,y+0.5*dy,z+0.5*dz,s=10,color='green') 
            losskpts=round(float(np.linalg.norm(tmp-np.array((body_list[4][0],body_list[4][1],body_list[4][2])))),3)
            ax.scatter3D(tmp[0],tmp[1],tmp[2],s=10,color='gray',label='SpineBase Error: '\
                            +str(losskpts)+'m')
        

            
    ax.legend(loc='lower right')
    ax.set_zlabel('Z', fontdict={'size': 10, 'color': 'black'})
    ax.set_ylabel('Y', fontdict={'size': 10, 'color': 'black'})
    ax.set_xlabel('X', fontdict={'size': 10, 'color': 'black'})

    # 固定视角作图
    ax.view_init(elev=-70, azim=90)

    plt.savefig(os.path.join('savefig',gt_move+'_'+index+'_IOU'+str(IOU).split('.')[-1]+'.jpg'), bbox_inches='tight', dpi=300)
    print('saving figure '+gt_move+'_'+index+' with IOU: '+str(IOU))
    plt.close()


class RadarEvaluator(object):
    def __init__(self,plot_bbox_results):
        self.gt = []
        self.img_ids = []#可以不要
        self.prediction_nq=[]
        self.prediction=[]
        self.meaniou=0
        self.num_detect=0
        self.loss_kpts=0
        self.loss_bbox=0
        self.matcher_num_detect=0
        self.matcher_meaniou=0
        self.nq=1
        self.f_dict={0:'Static',1:'Walk',2:'Trot',3:'Sit Down',4:'Stand Up',5:'Random Action'}
        self.checklist=[]
        self.plot_bbox_results=plot_bbox_results

    def update(self, predictions_nq,predictions,targets):
        #self.img_ids.append(img_ids)
        self.prediction_nq.append(predictions_nq)
        self.prediction.append(predictions)
        self.gt.append(targets)
        
    def eval_iou(self,threshold):
        meaniou=0
        loss_kpts_final=0
        loss_bbox_final=0
        num_detect=0
        matcher_num_detect=0
        matcher_meaniou=0
        movement_right=0
        print('Plot: ',self.plot_bbox_results)
        for i in range(len(self.gt)):
            output_nq=self.prediction_nq[i]
            target=self.gt[i]
            iou_list=[]
            loss_bbox_list=[]
            loss_kpts_list=[]
            self.nq=output_nq.shape[0]
            for output in output_nq:
                kpts_out=output.squeeze()   #2*6
                matcher_choose=False
                if self.prediction[i]['kpts'].squeeze().equal(kpts_out):
                    matcher_choose=True
                    if int(self.prediction[i]['movement'])==target['movement'][0]:movement_right+=1
                kpts_tgt=target['kpts']
                outputs_dic1=(kpts_out[0].mul(torch.cos(kpts_out[1]+0.5*math.pi))).mul(torch.cos(kpts_out[2]))
                outputs_dic3=(kpts_out[0].mul(torch.sin(kpts_out[1]+0.5*math.pi))).mul(torch.cos(kpts_out[2]))
                outputs_dic2=kpts_out[0].mul(torch.sin(kpts_out[2]))
                outputs_dict=torch.stack([outputs_dic1,outputs_dic2,outputs_dic3],dim=0)
                outputs_dict=torch.cat((outputs_dict,kpts_out[3:6]),0) 
                loss_kpts=F.pairwise_distance(outputs_dict[:3], torch.tensor(kpts_tgt[:3]).cuda()).detach().cpu().numpy()
                loss_bbox=loss_kpts+F.pairwise_distance(outputs_dict[3:6], torch.tensor(kpts_tgt[3:6]).cuda()).detach().cpu().numpy()
                loss_kpts_list.append(loss_kpts)
                loss_bbox_list.append(loss_bbox)
                #将中心点、长宽高的表示方法转化为两点表示（左上前、右下后）
                out=torch.stack((outputs_dict[0]+outputs_dict[3],outputs_dict[1]+outputs_dict[4],outputs_dict[2]-outputs_dict[5],
                    outputs_dict[0]-outputs_dict[3],outputs_dict[1]-outputs_dict[4],outputs_dict[2]+outputs_dict[5]),0)
                tgt=np.stack((kpts_tgt[0]+kpts_tgt[3],kpts_tgt[1]+kpts_tgt[4],kpts_tgt[2]-kpts_tgt[5],
                    kpts_tgt[0]-kpts_tgt[3],kpts_tgt[1]-kpts_tgt[4],kpts_tgt[2]+kpts_tgt[5]),0)
                iou=iou_3d(out,tgt)
                iou_list.append(iou)
                loss_obli=abs(kpts_tgt[6]-kpts_out[6].detach().cpu().numpy())
                self.checklist.append([float(iou),loss_kpts,loss_bbox,loss_obli])
                # 传入预测中心点xyz坐标、真值中心点xyz坐标、真值bbox的数据、IOU值、预测动作、真值动作、预测倾角
                if self.plot_bbox_results:plotting(out, tgt, kpts_tgt, iou,self.f_dict[int(self.prediction[i]['movement'])],self.f_dict[target['movement'][0]],kpts_out[6])
                if matcher_choose:
                    matcher_meaniou+=iou_list[-1]
                    if iou_list[-1]>threshold:matcher_num_detect=matcher_num_detect+1
                
            if max(iou_list)>threshold: # nq个输出里面有一个超过0.5的话就算过了
                num_detect=num_detect+1
            meaniou+=max(iou_list)
            loss_kpts_final+=min(loss_kpts_list)
            loss_bbox_final+=min(loss_bbox_list)
        self.meaniou=meaniou/(i+1)
        self.matcher_meaniou=matcher_meaniou/(i+1)
        self.num_detect=num_detect
        self.matcher_num_detect=matcher_num_detect

        self.loss_bbox=loss_bbox_final/(i+1)
        self.loss_kpts=loss_kpts_final/(i+1)
        print('Prediction Accuracy of movement:',movement_right/(i+1))
        if self.nq>1:
            print("best_meaniou:",self.meaniou)
            print("matcher_meaniou:",self.matcher_meaniou)
            print("best_num_detect:",num_detect)
            print("matcher_num_detect:",self.matcher_num_detect)
            print("best_loss_kpts:",self.loss_kpts)
            print("best_loss_bbox:",self.loss_bbox)
        else:
            print("meaniou:",self.meaniou)
            print("num_detect:",num_detect)
            print("loss_kpts:",self.loss_kpts)
            print("loss_bbox:",self.loss_bbox)
        # self.checklist=np.array(self.checklist)
        # np.save('checklist.npy',self.checklist)
        # data=self.checklist
        # fig=plt.figure(dpi=300)
        # plt.subplot(311)
        # plt.scatter(data[:,1],data[:,0],s=0.5)
        # plt.xlim([0.0,0.2])
        # plt.xlabel('loss kpts')
        # plt.ylabel('IOU')
        # plt.subplot(312)
        # plt.scatter(data[:,2],data[:,0],s=0.5)
        # plt.xlim([0.0,0.2])
        # plt.xlabel('loss bbox')
        # plt.ylabel('IOU')
        # plt.subplot(313)
        # plt.scatter(data[:,3],data[:,0],s=0.5)
        # plt.xlim([0.0,0.2])
        # plt.ylabel('IOU')
        # plt.xlabel('loss obliquity')
        # fig.subplots_adjust(hspace=0.5) 
        # plt.savefig('loss与IOU关系.jpg',dpi=300,bbox_inches='tight')
            
            
            
            
            
            
        
    
