"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import math
import numpy as np

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, num_joints,cost_class: float = 1, cost_coord: float = 1,cost_bbox: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_coord: This is the relative weight of the L1 error of the keypoint coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_coord = cost_coord
        self.cost_bbox = cost_bbox
        self.num_joints = num_joints
        assert cost_class != 0 or cost_coord != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets,num_queries):

        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_label_one_hot = outputs["pred_logits"].softmax(-1)
        out_label_szy = out_label_one_hot[:,:,1]
        out_label_szy = out_label_szy.unsqueeze(-1)
        
        # 标签label的tensor格式
        target_label_list = []
        for batch_sample in targets:
            classes = batch_sample['categories']
            batch_label_list = []
            for class_one in classes:
                batch_label_list.append(float(class_one))
            target_label_list.append(batch_label_list)
        target_label_tensor = torch.tensor(target_label_list)
        target_label_tensor = target_label_tensor.unsqueeze(-1)
        target_label_tensor = target_label_tensor.to(out_label_szy.device)
        
        cost_label = torch.cdist(out_label_szy.float(), target_label_tensor.float(), p=2)

        outputs_coord = outputs['pred_coords'] #40*num_queries*3
        if len(outputs_coord.shape) == 2:
            outputs_coord = outputs_coord.unsqueeze(1)
        

        
        # 网络输出：球坐标转XYZ坐标
        outputs_dic1=(outputs_coord[:,:,0].mul(torch.cos(outputs_coord[:,:,1]+0.5*math.pi))).mul(torch.cos(outputs_coord[:,:,2]))
        outputs_dic3=(outputs_coord[:,:,0].mul(torch.sin(outputs_coord[:,:,1]+0.5*math.pi))).mul(torch.cos(outputs_coord[:,:,2]))
        outputs_dic2=outputs_coord[:,:,0].mul(torch.sin(outputs_coord[:,:,2]))
        out_kpt=torch.stack([outputs_dic1,outputs_dic2,outputs_dic3],dim=2)
        # 网络输出：纯距离R
        # out_kpt = outputs_coord
        
        
        # 网络输出：bbox边界信息
        out_bbox_boundary=outputs_coord[:,:,3:6]
        
        # bbox边界标签
        tgt_bbox_boundary = torch.tensor(np.array([v["kpts"][3:6] for v in targets]))
        tgt_bbox_boundary=tgt_bbox_boundary.to(out_kpt.device)
        if len(tgt_bbox_boundary.shape) == 2:
            tgt_bbox_boundary = tgt_bbox_boundary.unsqueeze(1)
            
        # 中心点标签
        tgt_kpt = torch.tensor(np.array([v["kpts"][:3] for v in targets]))
        tgt_kpt=tgt_kpt.to(out_kpt.device)
        if len(tgt_kpt.shape) == 2:
            tgt_kpt = tgt_kpt.unsqueeze(1)
        
        #标签GT:球坐标转XYZ
        # tgt_dic1 = (tgt_kpt[:, :, 0].mul(torch.cos(tgt_kpt[:, :, 1]))).mul(torch.cos(tgt_kpt[:, :, 2]))
        # tgt_dic2 = (tgt_kpt[:, :, 0].mul(torch.sin(tgt_kpt[:, :, 1]))).mul(torch.cos(tgt_kpt[:, :, 2]))
        # tgt_dic3 = tgt_kpt[:, :, 0].mul(torch.sin(tgt_kpt[:, :, 2]))
        # xyz_tgt_kpt = torch.stack([tgt_dic1, tgt_dic2, tgt_dic3], dim=2)
        
        #标签GT:本身就是XYZ
        #tgt_dic1 = tgt_kpt[:, :, 0]
        #tgt_dic2 = tgt_kpt[:, :, 1]
        #tgt_dic3 = tgt_kpt[:, :, 2]
        #xyz_tgt_kpt = torch.stack([tgt_dic1, tgt_dic2, tgt_dic3], dim=2)
        
        #标签GT:纯距离R
        xyz_tgt_kpt = tgt_kpt
        
        
        
        # 欧式距离误差计算 p = 2 ==> L2 norm 这个函数很吊 #确实吊
        cost_kpt = torch.cdist(out_kpt.float(), xyz_tgt_kpt.float(), p=2)  
        cost_boundary = torch.cdist(out_bbox_boundary.float(), tgt_bbox_boundary.float(),p=2)
                
        # Final cost matrix
        # matcher这里不考虑倾角loss
        C = self.cost_coord * cost_kpt + self.cost_class * cost_label + self.cost_bbox * cost_boundary
  
        # C = C.transpose(1, 2).cpu()  # [40,3,3]
        C=C.view(batch_size,num_queries,-1).cpu()
        # tgt_num = torch.tensor([len(v['categories'][v['categories'] == 1]) for v in targets]) # 统计真值个数
        tgt_num = torch.tensor([v['categories'].count(1) for v in targets]) #
        

        indices = [linear_sum_assignment(c[:,0:tgt_num[i]]) for i,c in enumerate(C)] 
        full_indices = [linear_sum_assignment(c) for c in C]   # [(行索引，列索引),*40个]
        return indices,[(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices],full_indices,[(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in full_indices]


def build_matcher(args):
    num_joints = args.classes_num
    cost_class = 1.0
    cost_coord = 3.0
    cost_bbox=2.0
    num_queries=args.num_queries
    return HungarianMatcher(num_joints=num_joints, cost_class=cost_class, cost_coord=cost_coord,cost_bbox=cost_bbox)