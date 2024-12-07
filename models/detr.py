# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
from asyncio.log import logger
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from util.misc import NestedTensor, nested_tensor_from_tensor_list

from .backbone import build_backbone
from .new_matcher import build_matcher
from .transformer import build_transformer
import math
import os


global src_idx

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, add_obliquity= False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.add_obliquity=add_obliquity
        hidden_dim = transformer.d_model
        self.movement_num=6
        self.class_embed = nn.Linear(hidden_dim, self.movement_num + 1)
        #self.class_embed = MLP(hidden_dim, hidden_dim*2,num_classes + 1,3)
        self.bbox_embed_range = MLP(hidden_dim, hidden_dim*4, 1, 3)
        self.bbox_embed_angle = MLP(hidden_dim, hidden_dim*4, 2, 3)#第一个数字表示输出的维度，三维坐标设置成3
        self.bbox_embed_boundary = MLP(hidden_dim, hidden_dim*4, 3, 3)
        if self.add_obliquity:
            self.bbox_embed_obliquity = MLP(hidden_dim, hidden_dim*4, 1, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.hs=None

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # print(samples)
        # input('samples...')
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        feature, pos = self.backbone(samples)
        #src, mask = features[-1].decompose()
        #assert mask is not None
        mask=None
        # hsi,memory1 = self.transformer(feature[:,0,:,:].squeeze(), mask, self.query_embed.weight, pos)
        # hsq,memory2 = self.transformer(feature[:,1,:,:].squeeze(), mask, self.query_embed.weight, pos)
        hs, memory = self.transformer(feature, mask, self.query_embed.weight, pos)
        self.eb=hs[-1].squeeze()  # 40*512
        outputs_class = self.class_embed(hs)    # 3*256*1*7
        # outputs_coord = self.bbox_embed(hs).sigmoid()#(R、azimuth angle、elevation angle)  # BUG 这个sigmoid用错了
        #outputs_coord = self.bbox_embed(hs).sigmoid() * 12
        
        outputs_coord_range = self.bbox_embed_range(hs).sigmoid() * 12
        outputs_coord_angle = (self.bbox_embed_angle(hs).sigmoid()-0.5) * math.pi   # 3*256*1*1
        outputs_coord_boundary = self.bbox_embed_boundary(hs).sigmoid() * 1.5
        if not self.add_obliquity:
            outputs_coord=torch.cat((outputs_coord_range, outputs_coord_angle,outputs_coord_boundary,), 3)
        else:
            outputs_coord_obliquity = self.bbox_embed_obliquity(hs).sigmoid() * math.pi
            outputs_coord=torch.cat((outputs_coord_range, outputs_coord_angle,outputs_coord_boundary,outputs_coord_obliquity,), 3)
        out = {'pred_logits': outputs_class[-1], 'pred_coords': outputs_coord[-1]}
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_coords': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,args):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.movement_num=6
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.point_to_box=None
        self.best_iou={0:''}
        self.args=args
        empty_weight = torch.ones(self.movement_num + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    
    @torch.no_grad()
    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        if target.numel() == 0:
            return [torch.zeros([], device=output.device)]
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def loss_labels(self, outputs, targets, indices): # 为什么要对所有点都计算交叉熵，不应该是匹配的那几组点就好了吗？
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        global src_idx
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        
        # 输出的onehot格式：（256*1）*2
        output_class = src_logits[src_idx]
        
        # 标签label的onehot格式
        target_label_list = []
        for batch_sample in targets:
            classes = batch_sample['movement']
            batch_label_list = []
            for class_one in classes:
                batch_label_list.append(int(class_one))
            target_label_list.append(batch_label_list)
        target_label_tensor = torch.tensor(target_label_list)
        target_label_one_hot  = F.one_hot(target_label_tensor, num_classes=self.movement_num + 1)
        target_class_one_hot = target_label_one_hot[tgt_idx]
        
        
        
        target_class_entropy = torch.topk(target_class_one_hot, 1,dim=1)[1].squeeze(1)
        target_class_entropy = target_class_entropy.to(output_class.device)
        loss_ce = F.cross_entropy(output_class  ,target_class_entropy, self.empty_weight)

        losses = {'loss_ce': loss_ce}
        pred_index=torch.max(output_class,dim=1)[1]
        losses['class_error']=torch.count_nonzero(pred_index-target_class_entropy)/len(pred_index)

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        tgt_lengths = pred_logits.new_ones(pred_logits.shape[0]) * self.num_classes
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

            
    def loss_boxes(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_coords' in outputs
        global src_idx
        src_idx = self._get_src_permutation_idx(indices)  # always (0, 1, 2, .., 16)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        
        outputs_coord=outputs['pred_coords']
        outputs_dic1=(outputs_coord[:,:,0].mul(torch.cos(outputs_coord[:,:,1]+0.5*math.pi))).mul(torch.cos(outputs_coord[:,:,2]))
        outputs_dic3=(outputs_coord[:,:,0].mul(torch.sin(outputs_coord[:,:,1]+0.5*math.pi))).mul(torch.cos(outputs_coord[:,:,2]))
        outputs_dic2=outputs_coord[:,:,0].mul(torch.sin(outputs_coord[:,:,2]))
        outputs_dict=torch.stack([outputs_dic1,outputs_dic2,outputs_dic3],dim=2)
        outputs_dict=torch.cat((outputs_dict,outputs_coord[:,:,3:6]),2) # 40*1*6
        src_boxes = outputs_dict[src_idx]   # 1*40*6
        # 如果只有一个点，需要升维
        if self.num_classes == 1:
            src_boxes = src_boxes.unsqueeze(0)
            
        target_boxes = torch.tensor(np.array([v["kpts"][:6] for v in targets]))   #40*6
        if len(target_boxes.shape) == 2:
            target_boxes = target_boxes.unsqueeze(1)#(batch,num_queries,6) 40*1*6
        target_boxes = target_boxes[tgt_idx]    #40*6

        # 如果只有一个点，需要升维
        if self.num_classes == 1:
            target_boxes = target_boxes.unsqueeze(0)    #1*40*6
        target_boxes=target_boxes.to(src_boxes.device)
        
        

        # loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        # loss_bbox = nn.MSELoss(reduction='none')(src_boxes[:,:,:3], target_boxes[:,:,:3]) + nn.MSELoss(reduction='none')(src_boxes[:,:,3:6], target_boxes[:,:,3:6])
        loss_bbox = F.pairwise_distance(src_boxes[0,:,:3], target_boxes[0,:,:3]) + F.pairwise_distance(src_boxes[0,:,3:6], target_boxes[0,:,3:6])

        losses = {}
        losses['loss_bbox'] = loss_bbox.mean()

        src_xyz=self.cxyzlwh_to_xyzxyz(src_boxes)   # 40*1*6
        tgt_xyz=self.cxyzlwh_to_xyzxyz(target_boxes)
        
        #IOU
        iou,union=self.iou_3d(src_xyz,tgt_xyz)
        
        # loss_iou=-torch.log(iou+0.001)
        # #GIOU
        # Ac=self.get_Ac(src_xyz,tgt_xyz)
        # loss_giou=1-(iou-(Ac-union)/Ac)
        
        # DIOU
        loss_diou=self.get_diou_loss(src_xyz,tgt_xyz,iou)
        losses['loss_iou'] = loss_diou.mean()
        loss_diou=loss_diou.detach().cpu().numpy()
        
        return losses

        
    def loss_kpts(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        
        # loss_kpts是回归损失，loss_labels是分类损失。此处类别一致，故只考虑回归损失
        
        assert 'pred_coords' in outputs
        # match gt --> pred
        src_idx = self._get_src_permutation_idx(indices)  # always (0, 1, 2, .., 16)
        tgt_idx = self._get_tgt_permutation_idx(indices)  # must be in range(0, 100)


        outputs_coord = outputs['pred_coords'][:,:,:3]
        if len(outputs_coord.shape) == 2:
            outputs_coord = outputs_coord.unsqueeze(1)
            
        # 输出的：XYZ坐标
        outputs_dic1=(outputs_coord[:,:,0].mul(torch.cos(outputs_coord[:,:,1]+0.5*math.pi))).mul(torch.cos(outputs_coord[:,:,2]))
        outputs_dic3=(outputs_coord[:,:,0].mul(torch.sin(outputs_coord[:,:,1]+0.5*math.pi))).mul(torch.cos(outputs_coord[:,:,2]))
        outputs_dic2=outputs_coord[:,:,0].mul(torch.sin(outputs_coord[:,:,2]))
        outputs_dict=torch.stack([outputs_dic1,outputs_dic2,outputs_dic3],dim=2)
        # 输出的：纯距离 R
        # outputs_dict = outputs_coord    #40*3*1

        src_kpts = outputs_dict[src_idx]    #1*80*1

        # 如果只有一个点，需要升维
        if self.num_classes == 1:
            src_kpts = src_kpts.unsqueeze(0)
        
        # targets=targets.to(out_kpt.device)
        tgtkpts_1 = torch.tensor(np.array([v["kpts"][:3] for v in targets]))  # 40*2*1
        if len(tgtkpts_1.shape) == 2:
            tgtkpts_1 = tgtkpts_1.unsqueeze(1)
        tgt_kpt = tgtkpts_1[tgt_idx]

        # 如果只有一个点，需要升维
        if self.num_classes == 1:
            tgt_kpt = tgt_kpt.unsqueeze(0)
        tgt_kpt=tgt_kpt.to(src_kpts.device)
        
        
        # 标签的
        xyz_tgt_kpt = tgt_kpt

        # 第二种欧式距离计算loss方式
        src_kpts_cdist = src_kpts.transpose(0,1)
        xyz_tgt_kpt_cdist = xyz_tgt_kpt.transpose(0,1)
        loss_cdist_all = torch.cdist(src_kpts_cdist.float(),xyz_tgt_kpt_cdist.float(),p=2)
        loss_cdist = loss_cdist_all.mean()

        losses = {'loss_kpts': loss_cdist * self.num_classes}
        return losses
     
    
    def loss_obliquity(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        
        # loss_kpts是回归损失，loss_labels是分类损失。此处类别一致，故只考虑回归损失
        
        assert 'pred_coords' in outputs
        # match gt --> pred
        src_idx = self._get_src_permutation_idx(indices)  # always (0, 1, 2, .., 16)
        tgt_idx = self._get_tgt_permutation_idx(indices)  # must be in range(0, 100)


        outputs_coord = outputs['pred_coords']
        if len(outputs_coord.shape) == 2:
            outputs_coord = outputs_coord.unsqueeze(1)
            
        # 输出的：倾角
        outputs_dict=outputs_coord[:,:,6]

        src_kpts = outputs_dict[src_idx] 

        # 如果只有一个点，需要升维
        if self.num_classes == 1:
            src_kpts = src_kpts.unsqueeze(0)
        
        # targets=targets.to(out_kpt.device)
        tgtkpts_1 = torch.tensor([v["kpts"][6] for v in targets]) 
        # if len(tgtkpts_1.shape) == 2:
        tgtkpts_1 = tgtkpts_1.unsqueeze(1)
        tgt_kpt = tgtkpts_1[tgt_idx]

        # 如果只有一个点，需要升维
        if self.num_classes == 1:
            tgt_kpt = tgt_kpt.unsqueeze(0)
        tgt_kpt=tgt_kpt.to(src_kpts.device)
        
        # 标签的
        xyz_tgt_kpt = tgt_kpt

        # 第二种欧式距离计算loss方式
        src_kpts_cdist = src_kpts.transpose(0,1)
        xyz_tgt_kpt_cdist = xyz_tgt_kpt.transpose(0,1)
        loss_cdist_all = torch.abs(src_kpts_cdist-xyz_tgt_kpt_cdist)
        loss_cdist = loss_cdist_all.mean()

        losses = {'loss_obliquity': loss_cdist * self.num_classes}
        return losses        
    
    
    def iou_3d(self,box1, box2):
    
        #box [x1,y1,z1,x2,y2,z2]   分别是左上、右下的坐标（对角）

        area1 = (box1[:,:,0]-box1[:,:,3])*(box1[:,:,1]-box1[:,:,4])*(box1[:,:,5]-box1[:,:,2])
        area2 = (box2[:,:,0]-box2[:,:,3])*(box2[:,:,1]-box2[:,:,4])*(box2[:,:,5]-box2[:,:,2])
        area_sum = area1 + area2

        #计算重叠部分 设重叠box坐标为 [x1,y1,z1,x2,y2,z2]
        x1 = torch.minimum(box1[:,:,0], box2[:,:,0])
        y1 = torch.minimum(box1[:,:,1], box2[:,:,1])
        z1 = torch.maximum(box1[:,:,2], box2[:,:,2])
        x2 = torch.maximum(box1[:,:,3], box2[:,:,3])
        y2 = torch.maximum(box1[:,:,4], box2[:,:,4])
        z2 = torch.minimum(box1[:,:,5], box2[:,:,5])

        inter_area = (x1-x2)*(y1-y2)*(z2-z1)
        inter_area[x1 <= x2]=0  #xyz轴存在不重叠的情况设交集为0
        inter_area[y1 <= y2]=0
        inter_area[z1 >= z2]=0

        return inter_area/(area_sum-inter_area),(area_sum-inter_area)
    
    def get_Ac(self,box1, box2):
        
        #box [x1,y1,z1,x2,y2,z2]   分别是左上、右下的坐标（对角）

        #计算Ac 设重叠box坐标为 [x1,y1,z1,x2,y2,z2]
        x1 = torch.maximum(box1[:,:,0], box2[:,:,0])
        y1 = torch.maximum(box1[:,:,1], box2[:,:,1])
        z1 = torch.minimum(box1[:,:,2], box2[:,:,2])
        x2 = torch.minimum(box1[:,:,3], box2[:,:,3])
        y2 = torch.minimum(box1[:,:,4], box2[:,:,4])
        z2 = torch.maximum(box1[:,:,5], box2[:,:,5])

        return (x1-x2)*(y1-y2)*(z2-z1)
    
    def cxyzlwh_to_xyzxyz(self,outputs_dict):
        out=torch.stack((outputs_dict[:,:,0]+outputs_dict[:,:,3],outputs_dict[:,:,1]+outputs_dict[:,:,4],outputs_dict[:,:,2]-outputs_dict[:,:,5],
        outputs_dict[:,:,0]-outputs_dict[:,:,3],outputs_dict[:,:,1]-outputs_dict[:,:,4],outputs_dict[:,:,2]+outputs_dict[:,:,5]),2)
        return out

    def get_diou_loss(self, box1, box2, iou):
        #box [x1,y1,z1,x2,y2,z2]   分别是左上、右下的坐标（对角）

        #计算Ac 设重叠box坐标为 [x1,y1,z1,x2,y2,z2]
        x1 = torch.maximum(box1[:,:,0], box2[:,:,0]) # 1*40
        y1 = torch.maximum(box1[:,:,1], box2[:,:,1])
        z1 = torch.minimum(box1[:,:,2], box2[:,:,2])
        x2 = torch.minimum(box1[:,:,3], box2[:,:,3])
        y2 = torch.minimum(box1[:,:,4], box2[:,:,4])
        z2 = torch.maximum(box1[:,:,5], box2[:,:,5])

        Ac1=torch.stack([x1,y1,z1],dim=2) # 1*40*3
        Ac2=torch.stack([x2,y2,z2],dim=2)

        # 最小全含长方体对角线欧氏距离
        c2 = (F.pairwise_distance(Ac1[0,:,:], Ac2[0,:,:], p=2))**2 # 40

        # 中心点欧氏距离
        d2 = (F.pairwise_distance(box1[0,:,:3], box2[0,:,:3], p=2))**2 

        loss_diou = 1-(iou-d2/c2)

        return loss_diou
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'kpts': self.loss_kpts,
            'boxes': self.loss_boxes,
            'obliquity': self.loss_obliquity,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets, target_weights=2176,num_queries=50):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        list_indices,indices,list_full_indices,list_full_indices = self.matcher(outputs_without_aux, targets,num_queries)

        # Compute all the requested losses
        losses = {}
        # print(self.losses)
        for loss in self.losses:
            if loss == 'cardinality':
                continue
            elif loss == 'labels':
                losses.update(self.get_loss(loss, outputs, targets, list_full_indices))
                # losses['class_error'] = 100
                # continue
                
            else: # kpts、bbox、obliquity
                losses.update(self.get_loss(loss, outputs, targets, indices))


        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # if 'aux_outputs' in outputs:
        #     for i, aux_outputs in enumerate(outputs['aux_outputs']):
        #         indices = self.matcher(aux_outputs, targets)
        #         for loss in self.losses:
        #             kwargs = {}
        #             if loss == 'labels':
        #                 # Logging is enabled only for the last layer
        #                 kwargs = {'log': False}
        #             elif loss == 'kpts':
        #                 kwargs = {'weights': target_weights}
        #             l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
        #             l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
        #             losses.update(l_dict)

        return losses



class PostProcess(nn.Module):
    """ This module converts the model's output into the format we want"""
    @torch.no_grad()
    def forward(self, outputs):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = outputs['pred_logits']


        prob = F.softmax(out_logits, -1)
        #a,b=prob.max(-1)
        scores, labels = prob.max(-1)
        # outputs_dic1=(outputs['pred_coords'][:,:,0].mul(torch.cos(outputs['pred_coords'][:,:,1]))).mul(torch.cos(outputs['pred_coords'][:,:,2]))
        # outputs_dic2=(outputs['pred_coords'][:,:,0].mul(torch.sin(outputs['pred_coords'][:,:,1]))).mul(torch.cos(outputs['pred_coords'][:,:,2]))
        # outputs_dic3=outputs['pred_coords'][:,:,0].mul(torch.sin(outputs['pred_coords'][:,:,2]))
        # out_kpts=torch.stack([outputs_dic1,outputs_dic2,outputs_dic3],dim=2)
        out_kpts = outputs['pred_coords']
        global src_idx
        out_kpts = out_kpts[src_idx]

        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        # 输出的onehot格式：（256*1）*2
        output_class = src_logits[src_idx]
        pred_index=torch.max(output_class,dim=1)[1]
        #results = [{'scores': s, 'labels': l, 'kpts': b} for s, l, b in zip(scores, labels, boxes)]
        results = [{'scores': s, 'labels': l, 'kpts': b, 'movement': m} for s, l, b, m in zip(scores, labels, out_kpts, pred_index)]
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = args.classes_num

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        add_obliquity=args.add_obliquity,
        
    )
    matcher = build_matcher(args)
    if not args.add_obliquity:
        weight_dict = {'loss_ce': 5, 'loss_bbox': args.bbox_loss_coef, 'loss_kpts': 10,  'loss_iou' : args.iou_loss_coef}
    else:
        weight_dict = {'loss_ce': 5, 'loss_bbox': args.bbox_loss_coef, 'loss_kpts': 10,  'loss_iou' : args.iou_loss_coef, 'loss_obliquity' : args.obliquity_loss_coef}

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality','kpts'] if not args.add_obliquity else ['labels', 'boxes', 'cardinality','kpts','obliquity']

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses,args=args)
    criterion.to(device)
    postprocessors = {'kpts': PostProcess()}
        

    return model, criterion, postprocessors
