# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os
import sys
import cv2
import math
import json
import torch
import numpy as np
from typing import Iterable
from tqdm import tqdm

import util.misc_sptsv2 as utils
from util.visualize import vis_output_seqs, extract_result_from_output_seqs, convert_rec_to_str

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    lr_scheduler: list = [0], print_freq: int = 10, text_length: int = 25):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    optimizer.param_groups[0]['lr'] = lr_scheduler[epoch]
    optimizer.param_groups[1]['lr'] = lr_scheduler[epoch] * 0.1

    for samples, input_box_seqs, input_label_seqs, output_box_seqs, output_label_seqs in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device); input_box_seqs = input_box_seqs.to(device); input_label_seqs = input_label_seqs.to(device); output_box_seqs = output_box_seqs.to(device)
        output_label_seqs = output_label_seqs.to(device)
        if not all(input_label_seqs.tolist()):
            continue
        output_seqs = torch.cat([output_box_seqs.flatten(),output_label_seqs.flatten() ])
        outputs_box, outputs_label = model(samples, input_box_seqs, input_label_seqs, text_length)
        outputs_box = outputs_box.reshape(-1, outputs_box.shape[-1])
        outputs_label = outputs_label.reshape(-1, outputs_label.shape[-1])
        outputs = torch.cat([outputs_box,outputs_label],0)
        loss = criterion(outputs, output_seqs.flatten())

        loss_dict = {'at':loss}
        weight_dict = {'at':1}
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
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
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, output_dir, chars, start_index, visualize=False, text_length=25):
    model.eval()
    criterion.eval()
    chars = list(chars)
    import time
    cnt = 0
    total = 0
    results = []
    for samples, targets in tqdm(data_loader):
        batch = len(targets)
        targets = targets[: batch // 2]
        samples.mask = samples.mask[: batch // 2, :, :]
        samples.tensors = samples.tensors[: batch // 2, :, :, :]
        samples = samples.to(device)
        dataset_names = [target['dataset_name'] for target in targets]
        targets = [{k: v.to(device) for k, v in t.items() if k != 'dataset_name'} for t in targets]
        seq = torch.ones(len(targets), 1).to(samples.mask) * start_index
        torch.cuda.synchronize()
        t0 = time.time()
        outputs = model(samples, seq,seq, text_length)
        torch.cuda.synchronize()
        t1 = time.time()
        cnt += 1
        total += t1-t0
        print(total/cnt)
        if outputs == None:
            continue
        outputs, values, rec_scores = outputs
        if visualize:
            samples_ = samples.to(torch.device('cpu')); outputs_ = outputs.cpu()
            vis_images = vis_output_seqs(samples_, outputs_, rec_scores, False)
            for vis_image, target, dataset_name in zip(vis_images, targets, dataset_names):
                save_path = os.path.join(output_dir, 'vis', dataset_name, '{:06d}.jpg'.format(target['image_id'].item()))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, vis_image)

        outputs = outputs.cpu(); values = values.cpu(); rec_scores = rec_scores.cpu()
        for target, output, value, rec_score in zip(targets, outputs, values, rec_scores):
            image_id = target['image_id'].item()
            output, split_index = extract_result_from_output_seqs(output, rec_score, return_index=True)
            split_values = [value[split_index[i]:split_index[i+1]] for i in range(0, len(split_index)-1)]
            center_pts = output['center_pts']; rec_labels = output['rec']; rec_scores = output['key_rec_score']
            rec_labels = convert_rec_to_str(rec_labels, chars)
            
            orig_h, orig_w = target['orig_size']#; img_h, img_w = target['size']
            for center_pt, rec_label, rec_score, split_value in zip(center_pts, rec_labels, rec_scores, split_values):
                if center_pt.numel() != 2:
                    continue
                center_pt = center_pt.numpy().reshape(-1, 2).astype(np.float64)
                center_pt[:, 0] *= (float(orig_w.item()) / 1000); center_pt[:, 1] *= (float(orig_h.item()) / 1000)
                polygon_pts = center_pt.tolist()
                result = {
                    'image_id': image_id,
                    'category_id': 1,
                    'polys': polygon_pts,
                    'rec': rec_label,
                    'score': split_value.mean().item(),
                    'value': split_value.numpy().tolist(),
                    'rec_score': rec_score.numpy().tolist()
                }
                results.append(result)

    json_path = os.path.join(output_dir, 'results', dataset_name+'.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    results_json = json.dumps(results, indent=4)
    with open(json_path, 'w') as f:
        f.write(results_json)    
