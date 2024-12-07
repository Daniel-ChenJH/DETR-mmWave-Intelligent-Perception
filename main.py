# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
# from importlib.resources import path
import json
import random
import time
import os
import sys
from pathlib import Path

import numpy as np
import torch

import util.misc as utils

from engine import  train_one_epoch, evaluate_radar
from models import build_model

from datasets import Radar_Real_dataloader_new

# nohup python -u main.py >xxx.log 2>&1 &
# fuser -v /dev/nvidia*

class Logger(object):
    # CJH
    def __init__(self, fileN=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")):
        if not os.path.exists('logging'):os.mkdir('logging')
        if sys.platform =='win32':fileN=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        fileN = "logging-"+fileN+".log" 
        self.terminal = sys.stdout
        self.log = open(os.path.join('logging',fileN), "a+",encoding='utf-8')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
        
    def flush(self):
        self.log.flush()



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # new is an acclerated data loader.
    parser.add_argument('--data_total_or_mini',default='total', type=str)  
    parser.add_argument('--add_obliquity',default=True, type=bool)
    parser.add_argument('--extra_seed_in_data_generation',default=True, type=bool)
    parser.add_argument('--trainval_data_portion',default=0.8, type=float)
    parser.add_argument('--npy_data_path',default=r"E:\实验室个人代码备份20230417\graduation\npy_data_with_skeleton", type=str) 
    parser.add_argument('--plot_bbox_results',default=True, type=bool)
    parser.add_argument('--plot_umap_results',default=False, type=bool)
    parser.add_argument('--lr', default=1e-4, type=float)  # 不调 1e-4
    parser.add_argument('--lr_backbone', default=1e-5, type=float)  # 调大了不收敛
    # 正则化系数
    parser.add_argument('--regularization', default=0, type=float)

    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)      # 300
    parser.add_argument('--lr_drop', default=80, type=int)
    parser.add_argument('--clip_max_norm', default=0.05, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--iou_threshold', default=0.5, type=float)#iou阈值，默认0.5
    # * Backbone
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=3, type=int,
                        help="Number of encoding layers in the transformer")  # 1 或 3
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")  # 1 或 3
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.05, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")  # 暂时按4，可以考虑4, 6, 8
    parser.add_argument('--num_queries', default=1, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss 
    # 有传参时默认为True，否则为False
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)") # 是否禁用辅助解码损耗（每层的损耗），默认不禁用
    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--iou_loss_coef', default=5, type=float)
    parser.add_argument('--obliquity_loss_coef', default=5, type=float)
    parser.add_argument('--output_dir', default="./save_model",
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    # parser.add_argument('--resume', default='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default="\\save_model\\checkpoint0000.pth", help='resume from checkpoint')
    #parser.add_argument('--resume', default='/home/sjtu/data/zyh/DETR_bbox_825_zyh/save_model/checkpoint.pth',help='resume from checkpoint')
    # parser.add_argument('--resume', default=r'/home/sjtu3090/data/cjh/graduation/DETR_0505_movement/save_model/checkpointbest.pth',help='resume from checkpoint')  # 加载权重文件
    parser.add_argument('--resume', default=r"E:\实验室个人代码备份20230417\graduation\DETR_0507_movement_final\save_model\checkpointbest.pth",help='resume from checkpoint')  # 加载权重文件


    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='start epoch')  # 计数从N开始

    parser.add_argument('--eval', default='val')#"train,val,test"

    parser.add_argument('--classes_num', default=1, type=int)
    return parser

result_recording = {'iou': [], 'best_iou': [],'num_detect': []}
def main(args):
    # 输出同时重定向到文件
    sys.stdout = Logger()
    import shutil
    if args.plot_bbox_results:
        if os.path.exists('savefig'):shutil.rmtree('savefig')
        os.mkdir('savefig')

    if args.plot_umap_results:
        if os.path.exists('umaps'):shutil.rmtree('umaps')
        os.mkdir('umaps')


    #初始化分布式训练模式（由输入决定）
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)
    assert args.data_total_or_mini in ['total','mini']
    assert args.trainval_data_portion >= 0 and args.trainval_data_portion <= 1
    if args.trainval_data_portion!=0:assert args.extra_seed_in_data_generation
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    #根据参数构建模型、loss函数和后处理方法
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.9)

    
    # 数据加载
    print('using new dataloader')
    data_loader_all = Radar_Real_dataloader_new.Gen_data(args)

    output_dir = Path(args.output_dir)
    #从某个训练阶段恢复，加载当时的权重、优化器、学习率等
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        if args.eval=='train' and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            
    if args.eval=='val':
        # TODO: 还没跑通
        #测试而不训练
        test_stats, radar_evaluator = evaluate_radar(args.start_epoch,model, criterion, postprocessors,
                                              data_loader_all, device, args.output_dir,args.iou_threshold,args.plot_bbox_results,args.plot_umap_results)
        return

    iou=[]
    best_iou=0
    num_detect=[]
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        print('Training epoch '+str(epoch+1)+'...')
        train_stats = train_one_epoch(
            model, criterion, data_loader_all, optimizer, device, epoch,
            args.clip_max_norm,batch_size=args.batch_size,num_queries=args.num_queries, reg=args.regularization)
        lr_scheduler.step()

        ## 先不测试
        # TODO: 测试实现
        
        test_stats, radar_evaluator = evaluate_radar(epoch,
            model, criterion, postprocessors, data_loader_all, device, args.output_dir,args.iou_threshold,args.plot_bbox_results,args.plot_umap_results
        )

        iou.append(radar_evaluator.meaniou)
        num_detect.append(radar_evaluator.num_detect)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 40 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            if (best_iou<radar_evaluator.meaniou and epoch>5):
                checkpoint_paths.append(output_dir / f'checkpointbest.pth')
                best_iou=radar_evaluator.meaniou
                
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                      **{f'test_{k}': v for k, v in test_stats.items()},
                      'epoch': epoch,
                      'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
            if radar_evaluator is not None:
                result_recording['iou'].append(float(radar_evaluator.meaniou))
                result_recording['best_iou'].append(float(best_iou))
                result_recording['num_detect'].append(float(radar_evaluator.num_detect))
                # input('one epoch')
                with open('result_record.json', 'w') as file:
                    json.dump(result_recording, file, indent=2)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
