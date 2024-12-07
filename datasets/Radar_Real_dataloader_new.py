import random
import numpy as np
import math
import os
import random
from pathlib import Path
import numpy as np
import json

def data_process(V_data):
    data = V_data
    angle = np.angle(data)
    amplitude = np.abs(data)        # 提取相位、幅值

    pil_csv_raw = (np.sum(amplitude, axis=0)/12).astype(np.float32)   # 幅值取均值是因为每个channel的幅值基本是一样的，所以取个均值保留一份就可以了

    diff_phase=getphased(angle)#天线横向，纵向的相位差
    # 幅值拼接角度
    concat_data = np.concatenate([pil_csv_raw,diff_phase],axis=0).astype(np.float32) 
    return concat_data
  
def get_diff_phase(phase1,phase2):
    p2 = 2 * math.pi
    # 取相位差
    diff_phase = phase1 - phase2

    # 在做角度解绕，变回0到2pi区间内
    diff_phase = np.where(diff_phase < 0, diff_phase + p2, diff_phase)
    diff_phase = np.where(diff_phase > p2, diff_phase - p2, diff_phase)
    
    return diff_phase

def getphased(angle8):
    # 相位差某几个channel取平均是因为那几个channel对应的tr天线的相对位置是一样的，所以相位差分布也是一样的，取平均保留一份就可以了。
    array0 = np.array([[0,2],[4,6],[1,3],[5,7],[8,10],[9,11]])
    angle_C = get_diff_phase(angle8[array0[:,0]],angle8[array0[:,1]])
    diff_phase_0 = np.sum(angle_C,axis=0)/6
    
    array1 = np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]])
    angle_C = get_diff_phase(angle8[array1[:,0]],angle8[array1[:,1]])
    diff_phase_1 = np.sum(angle_C,axis=0)/6

    array2 = np.array([[0,8],[1,9],[2,10],[3,11]])
    angle_C = get_diff_phase(angle8[array2[:,0]],angle8[array2[:,1]])
    diff_phase_2 = np.sum(angle_C,axis=0)/4

    array3 = np.array([[4,8],[5,9],[6,10],[7,11]])      # 之前这里写的是[8,11]，写错了！！现在改成[7,11]！
    angle_C = get_diff_phase(angle8[array3[:,0]],angle8[array3[:,1]])
    diff_phase_3 = np.sum(angle_C,axis=0)/4

    diff_phase = np.concatenate([diff_phase_0,diff_phase_1,diff_phase_2,diff_phase_3],axis=0)
    # diff_phase = np.concatenate([diff_phase_0,diff_phase_1],axis=0)
    return diff_phase

def get_bbox_info(people_num,body_position_list):
    #             body_list = ["Head","Neck","SpineShoulder","SpineMid","SpineBase","ShoulderRight","ElbowRight","WristRight",
    #        "ShoulderLeft","ElbowLeft","WristLeft","HipRight","KneeRight","AnkleRight","HipLeft","KneeLeft","AnkleLeft"]

    for people in range(people_num):
        position_list = body_position_list[people*17:(people+1)*17]
        SpineShoulder = position_list[2]
        SpineBase = position_list[4]          # 脊柱底部作为中心点坐标

        bbox_info = SpineBase[:]
        # 倾角为从Kinect看向人时，人体中心线与X轴正方向的夹角，范围[0,pi]
        
        # 倾斜框有点难整，而且耗时应该很长，需要计算各点在中心线投影位置
        # 暂时先用水平框实现，倾斜角作为记录人体倾斜度的一个参数
        minlist = np.min(position_list,axis=0)
        maxlist = np.max(position_list,axis=0)

        bbox_length = np.max(np.concatenate([[SpineBase-minlist],[maxlist-SpineBase]],axis=0),axis=0)  
        for i in range(3):          # 三维坐标
            bbox_info=np.append(bbox_info,min(bbox_length[i],1.5))       # bbox中心点到单边长度，阈值设为1.5m
        obliquity = np.arctan((SpineShoulder[1]-SpineBase[1])/(SpineShoulder[0]-SpineBase[0]))  # [-pi/2,pi/2] 用脊柱算倾角
        obliquity = obliquity if obliquity > 0 else obliquity+np.pi     # 主视图倾角，范围 [0,pi]
        # bbox_info = np.append(bbox_info,obliquity)   # 列表最后一个元素是倾角
    return bbox_info,obliquity

class Gen_data():
    def __init__(self,args):
        # args.npy_data_path, args.data_total_or_mini, args.add_obliquity,args.extra_seed_in_data_generation
        self.name=args.data_total_or_mini
        self.add_obliquity=args.add_obliquity
        self.extra_seed_in_data_generation=args.extra_seed_in_data_generation
        self.data_path=os.path.join(args.npy_data_path,self.name+'_all_labels.npy')
        self.trainval_data_portion=args.trainval_data_portion
        if os.path.exists(self.data_path) and os.path.exists(self.data_path.replace('_all_labels.npy','_all_images.npy')):
            print('Reading already exist npy files')
            self.all_labels=np.load(self.data_path, allow_pickle='TRUE')
            self.all_images=np.load(self.data_path.replace('_all_labels.npy','_all_images.npy'), allow_pickle='TRUE')
            if self.add_obliquity: 
                print('add obliquity')
                for i in range(len(self.all_labels)): 
                    self.all_labels[i]['kpts']=np.append(self.all_labels[i]['kpts'],self.all_labels[i]['obliquity'])
                    del self.all_labels[i]['obliquity']
            for i in range(len(self.all_labels)): 
                self.all_labels[i]['movement']=[self.all_labels[i]['movement']]
            print('Done with reading npy files')
        else:
            json_data_path = Path('/home/sjtu3090/data/HUAWEI/simu_data/Real_data/json_data/json_data')
            # 提取所有雷达的标签
            self.all_labels = []
            self.all_images = []
            self.point_to_box = {}
            self.movement_dict={}
            f=open('labeled_json_list.txt','r',encoding='utf-8')
            self.f_dict={'静止':0,'走路':1,'小跑':2,'坐下':3,'站起':4,'随机动作':5}
            for i in f.readlines():
                i=i.strip()
                [filename,movement_ch]=i.split('：')
                self.movement_dict[filename]=self.f_dict[movement_ch]

            json_file = os.listdir(json_data_path)
            for i,line in enumerate(json_file):
                if i%100==0:print(i)
                file_name = os.path.join(json_data_path,line)
                
                json_data = json.load(open(file_name,'r',encoding="utf-8"))
                if not json_data['Target exist']:
                    print(str(i) + "no target")
                    continue
                #提取雷达数据
                
                V_data = []
                for j in range(12):
                    real_dir_key = 'virtual ' + str(j+1) + ' real'
                    V_Real = np.array(json_data[real_dir_key]).astype(np.float32)
                    imag_dir_key = 'virtual ' + str(j+1) + ' imag'
                    V_Imag = np.array(json_data[imag_dir_key]).astype(np.float32)
                    V_complex_data = V_Real+V_Imag*1j
                    V_data.append(V_complex_data)
                V_data = np.array(V_data)
                final_data = data_process(V_data)
                self.all_images.append(final_data)

                # 提取17个人体骨架关键点标签
                body_list = ["Head","Neck","SpineShoulder","SpineMid","SpineBase","ShoulderRight","ElbowRight","WristRight",
                "ShoulderLeft","ElbowLeft","WristLeft","HipRight","KneeRight","AnkleRight","HipLeft","KneeLeft","AnkleLeft"]

                body_position_list = []

                people_num = 1
                for k in range(people_num):  # 只有一个人
                    for item in body_list:
                        point_name = item + str(k+1)
                        point_position = np.array(json_data[point_name]).astype(np.float32)
                        body_position_list.append(point_position)
                        
                bbox_info,obliquity=get_bbox_info(people_num,body_position_list)
                # bbox_info是由 [倾角,中心点xyz坐标,单边xyz方向长度] 构成的长度为7的一维数组  目前都是单人数据
                    
                self.all_labels.append({"categories":[1],"kpts":bbox_info,"obliquity":obliquity,'filename':line,'movement':self.movement_dict[line]})
                if self.name=='mini':
                    if i >= 1000:  # 只载入多少文件
                        break            
                
            # np.save('point_to_box.npy', self.point_to_box)
            print('Saving npy files')
            np.save(self.data_path, self.all_labels)
            np.save(self.data_path.replace('_all_labels.npy','_all_images.npy'), self.all_images)
            print('Done with saving npy files')
            for i in range(len(self.all_labels)): 
                self.all_labels[i]['movement']=[self.all_labels[i]['movement']]            
            if self.add_obliquity: 
                print('add obliquity')
                for i in range(len(self.all_labels)): 
                    self.all_labels[i]['kpts']=np.append(self.all_labels[i]['kpts'],self.all_labels[i]['obliquity'])
                    del self.all_labels[i]['obliquity']


    def gen_data(self,mode,batch_size,gen_time):
        all_rand_list=np.arange(0,len(self.all_labels)) 
        if self.extra_seed_in_data_generation:random.seed(1)
        random.shuffle(all_rand_list)
        
        eval_num = 4000 if  self.name=='total' else 400   # 4000
        
        if mode == "train":
            rand_list=all_rand_list[eval_num:len(self.all_labels)]      # 前eval_num组用于测试，避开    
        else: # val
            start_index=int(self.trainval_data_portion*eval_num)
            rand_list=all_rand_list[start_index:eval_num+start_index]

        if batch_size*(gen_time+1) < len(rand_list):
            rand = rand_list[batch_size*gen_time:batch_size*(gen_time+1)]  
            data_exhaust = False
        else: 
            rand = rand_list[(len(rand_list)-batch_size):]
            data_exhaust = True
        
        batch_imgs = np.array([self.all_images[r] for r in rand])
        batch_labels = [self.all_labels[r] for r in rand]   

        return batch_imgs, batch_labels, data_exhaust
