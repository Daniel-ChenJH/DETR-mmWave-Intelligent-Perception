# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from tkinter import N
from typing import Iterable
import torch
import numpy as np
import util.misc as utils

from datasets.radar_eval_bbox import RadarEvaluator

import json
import umap.umap_ as umap
import umap.plot

loss_recording = {'loss_ce': [], 'loss_iou': [], 'loss_bbox': [],'kpts_mean': [], 'kpts_std': [], 'input_img': []}
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, batch_size: int=2,num_queries:int=50, reg:float=0.0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    gen_time=0
    data_exhaust=False
    while not data_exhaust:
        np_inputs, np_labels, data_exhaust = data_loader.gen_data(batch_size = batch_size, mode = "train", gen_time = gen_time)
        gen_time+=1
        inputs = torch.tensor(np_inputs).to(device)


        targets = tuple(np_labels)
        
        outputs = model(inputs)
        reg_loss=0
        for param in model.parameters():
            reg_loss+=torch.sum(abs(param))
        
        loss_dict = criterion(outputs, targets)

        # loss_recording['loss_iou'].append(loss_dict['loss_iou'].tolist())
        loss_recording['loss_bbox'].append(loss_dict['loss_bbox'].tolist())
        loss_recording['kpts_mean'].append(outputs['pred_coords'].mean().tolist())
        loss_recording['kpts_std'].append(outputs['pred_coords'].std().tolist())
        with open('loss_record.json', 'w') as file:
            json.dump(loss_recording, file, indent=2)
        
        weight_dict = criterion.weight_dict
        a=loss_dict.keys()
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)+reg*reg_loss

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Train data generation time:"+str(gen_time))
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            
            
@torch.no_grad()
def evaluate_radar(epoch,model, criterion, postprocessors, data_loader, device, output_dir,threshold,plot_bbox_results,plot_umap_results):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    radar_evaluator = RadarEvaluator(plot_bbox_results)

    gen_time=0
    data_exhaust=False
    model_eval_eb=None
    eval_labels=None
    while not data_exhaust:
        np_inputs, np_labels, data_exhaust = data_loader.gen_data(batch_size = 40, mode = "eval", gen_time = gen_time)
        gen_time+=1

        inputs = torch.tensor(np_inputs).to(device)


        targets = tuple(np_labels)
        outputs = model(inputs)     # bs*nq*6
        loss_dict = criterion(outputs, targets)   
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # print("loss_dict_reduced",type(loss_dict_reduced))
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                                **loss_dict_reduced_scaled,
                                **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['kpts'](outputs)   # 这里实际上就是 results = outputs['pred_coords'][src_idx]，而src_idx来自new_matcher的结果
        results_nq = outputs['pred_coords']    # bs*nq*6 带nq的指有nq个输出
        if plot_umap_results:
            eval_labels=np.append(eval_labels, np.array([i['movement'][0] for i in np_labels])) if eval_labels is not None else np.array([i['movement'][0] for i in np_labels])
            model_eval_eb=torch.cat((model_eval_eb, model.eb,), 0) if model_eval_eb is not None else model.eb
        if radar_evaluator is not None:
            for target, output_nq, output in zip(targets, results_nq, results):
                radar_evaluator.update(output_nq, output, target)
    # print("Validation data generation time:"+str(gen_time))
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if plot_umap_results:
        model_eval_eb=model_eval_eb.detach().cpu().numpy()
        # eval_labels=eval_labels.astype(str)
        checklist=['Static','Walk','Trot','Sit Down','Stand Up','Random Action']
        eval_labels=eval_labels.tolist()
        for i in range(len(eval_labels)):
            eval_labels[i]=checklist[eval_labels[i]]
        eval_labels=np.array(eval_labels)
        save={}        
        mapper = umap.UMAP(n_neighbors=40, random_state=42).fit(model_eval_eb)
        save['labels']=eval_labels
        save['eb']=model_eval_eb
        np.save('epoch0_umap_data.npy',save)
        fig, ax = umap.plot.plt.subplots(1, 1, figsize=(6,6),dpi=120)
        umap.plot.points(mapper, labels=eval_labels,ax=ax)
        umap.plot.plt.savefig(os.path.join(os.getcwd(),'umaps/epoch_'+str(epoch)+'.png'),dpi=300,bbox_inches='tight')
        umap.plot.plt.close()
        print('Saving umap of epoch: ',epoch)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if radar_evaluator is not None:
        if 'kpts' in postprocessors.keys():
            stats['radar_eval_bbox'] = radar_evaluator.eval_iou(threshold)

    return stats, radar_evaluator
