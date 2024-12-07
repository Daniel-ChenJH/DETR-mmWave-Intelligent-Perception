# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import torch
from torch import nn
from util.misc import NestedTensor
from .position_encoding import build_position_encoding

# class Joiner(nn.Sequential):
#     def __init__(self,  position_embedding,input_size):
#         super().__init__( position_embedding)
#         self.mlp = nn.Sequential(
# 			nn.Linear(input_size, 512),
# 			nn.ReLU(inplace=True),
# 		)
#     def forward(self, tensor_list: NestedTensor):
#         #0是backbone，1是embedding
#         tmptensor1i=tensor_list.tensors[:,0,:,:].squeeze().transpose(1,2)
#         tmptensor2i=self.mlp(tmptensor1i)
#         outi=tmptensor2i.transpose(1,2)
        
#         tmptensor1q=tensor_list.tensors[:,1,:,:].squeeze().transpose(1,2)
#         tmptensor2q=self.mlp(tmptensor1q)
#         outq=tmptensor2q.transpose(1,2)
#         '''
#         for i in range(2):
#             tmptensor1=tensor_list.tensors[:,i,:,:].squeeze().transpose(1,2)
#             tmptensor2=self.mlp(tmptensor1)
#             out[:,i,:,:]=tmptensor2.transpose(1,2)
#         '''
#         out=torch.stack((outi, outq), 1)
#         pos = self[0](tensor_list.tensors[:,0,:,:].squeeze()).to(out.device)
#         return  out, pos
    
class Joiner_Two(nn.Sequential):
    def __init__(self,  position_embedding,hid_dim1=2048,hid_dim2=2048):
        super().__init__(position_embedding)
        '''
        self.mlp1 = nn.Sequential(
			nn.Linear(64, 128),
			nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(128, 128),
		)
        self.mlp2 = nn.Sequential(
			nn.Linear(192, 384),
			nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(384, 384),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(384, 384),
		)
        '''
        self.mlp1 = nn.Sequential(
            nn.Linear(16, hid_dim1),
            nn.ReLU(),
            nn.Linear(hid_dim1, hid_dim1),
            nn.ReLU(),
            nn.Linear(hid_dim1, 128),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(64, hid_dim2),
            nn.ReLU(),
            nn.Linear(hid_dim2, hid_dim2),
            nn.ReLU(),
            nn.Linear(hid_dim2, 384),
        )
    def forward(self, tensor_list: NestedTensor):
        #0是backbone，1是embedding

        tensor_abs = tensor_list.tensors.transpose(1,2)[:,:,0:16]
        tensor_pha = tensor_list.tensors.transpose(1,2)[:,:,16:80]
        tensor_abs = self.mlp1(tensor_abs)
        tensor_pha = self.mlp2(tensor_pha)
        tensor_abs = tensor_abs.transpose(1,2)
        tensor_pha = tensor_pha.transpose(1,2)
        out = torch.cat((tensor_abs,tensor_pha),1)
        # pos 分开进行单独的位置编码

        pos_abs = self[0](tensor_abs).to(out.device)
        pos_pha = self[0](tensor_pha).to(out.device)
        pos = torch.cat((pos_abs,pos_pha),1)
        
        return  out, pos

def build_backbone(args):
    #输出进行位置编码
    position_embedding = build_position_encoding(args)

    model = Joiner_Two(position_embedding)
    #model.num_channels = backbone.num_channels
    model.num_channels=512
    return model