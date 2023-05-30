# Copyright (2023) Bytedance Ltd. and/or its affiliates
from sys import set_coroutine_origin_tracking_depth
from unicodedata import category
import cv2
import torch
import bezier
import numpy as np

    
def extract_result_from_output_seqs(
     seqs, rec_score, key_pts='center_pts', key_label='rec', return_index=False,
     end_index=1097, bins=1000, padding_bins=0, pad_rec=True
    ):
    target = {
        key_pts: [],
        key_label: [],
        'key_rec_score': []
    }
    # pts_len = 16 if key_pts=='bezier_pts' else 4
    pts_len = 2
    category_start_index = bins + 2*padding_bins
    rec_score = rec_score[:, category_start_index:category_start_index+95]
    split_index = [0, ]; index = 0; rec_index = 0
    while(True):
        if index >= len(seqs) or seqs[index] == end_index:
            break
        pts = seqs[index:index+pts_len]; index += pts_len

        pts = torch.clamp(pts, max=category_start_index-1) - padding_bins
        if not pad_rec:
            label = []
            while(index < len(seqs)):
                if seqs[index] >= category_start_index and seqs[index] < end_index:
                    label.append(seqs[index] - category_start_index)
                else:
                    break 
                index += 1
        else:
            label = seqs[index:index+25]; index += 25
            label = torch.clamp(label, min=category_start_index, max=end_index-1) - category_start_index
            if end_index - 1 in label:
                if torch.min(torch.where(label==end_index-1)[0]) == 0:
                    import pdb;pdb.set_trace()
                label = label[:torch.min(torch.where(label==end_index-1)[0])]
        split_index.append(index)
        target[key_pts].append(pts)
        target[key_label].append(label)
        target['key_rec_score'].append(rec_score[rec_index:rec_index+25].softmax(-1))
        rec_index = rec_index +25
    if return_index:
        return target, split_index
    return target

def draw_short_line(image, bezier_pts):
    cv2.line(
        image, 
        (bezier_pts[0, 0], bezier_pts[0, 1]),
        (bezier_pts[7, 0], bezier_pts[7, 1]),
        (0, 0, 255), 1
    )
    cv2.line(
        image, 
        (bezier_pts[3, 0], bezier_pts[3, 1]),
        (bezier_pts[4, 0], bezier_pts[4, 1]),
        (0, 0, 255), 1
    )
    return image

def draw_bezier_curves(image, bezier_pts):
    curve = bezier.Curve.from_nodes(bezier_pts)   
    x_vals = np.linspace(0, 1, 100)
    data = curve.evaluate_multi(x_vals)
    data = data.transpose().astype(np.int32)
    cv2.polylines(image, [data], False, (0, 0, 255), 1)
    return image, data

def draw_center_points(image, center_pts):
    for pt in center_pts:
        try:
            cv2.circle(image, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)
        except:
            import pdb;pdb.set_trace()
    return image

def tensor_to_cv2image(tensor, remove_padding=True):
    image = tensor.numpy()
    image = image.transpose((1, 2, 0))
    image = image * 255
    if remove_padding:
        image_ = np.sum(image, -1)
        image_h = np.sum(image_, 1)
        if 0 in image_h:
            h_border = np.min(np.where(image_h == 0)[0])
        else:
            h_border = image.shape[0]
        image_w = np.sum(image_, 0)
        if 0 in image_w:
            w_border = np.min(np.where(image_w == 0)[0])
        else:
            w_border = image.shape[1]
        image = image[:h_border, :w_border]
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def convert_pt_to_pixel(pts, height, width, bins=1000):
    new_pts = []
    for pt in pts:
        pt = pt.float() / bins 
        pt[::2] *= width 
        pt[1::2] *= height 
        new_pts.append(pt)
        # import pdb;pdb.set_trace()
    return new_pts

def convert_rec_to_str(rec_labels, chars):
    strs = []
    for rec_label in rec_labels:
        str = []
        for char in rec_label:
            if char < len(chars):
                str.append(chars[char])
        str = ''.join(str)
        strs.append(str)
    return strs

def draw_text(image, text, pt):
    cv2.putText(image, text, (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return image

def vis_output_seqs(samples, output_seqs, rec_scores, remove_padding=False, pad_rec=False):
    chars = list(' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')
    tensors = samples.tensors
    targets = []
    # targets = [extract_result_from_output_seqs(ele, pad_rec=pad_rec) for ele in output_seqs]  
    for ele, rec_score in zip(output_seqs, rec_scores):
        targets.append(extract_result_from_output_seqs(ele, rec_score, pad_rec=pad_rec))
    center_pts = [convert_pt_to_pixel(target['center_pts'], tensors.shape[2], tensors.shape[3]) for target in targets]
    rec_labels = [convert_rec_to_str(target['rec'], chars) for target in targets]
    vis_images = []
    for image_, mask_, center_pts_, rec_labels_ in zip(samples.tensors, samples.mask, center_pts, rec_labels):
        h_border = torch.max(torch.where(mask_==False)[0]) + 1
        w_border = torch.max(torch.where(mask_==False)[1]) + 1
        image_ = image_[:, :h_border, :w_border]
        image_ = tensor_to_cv2image(image_, remove_padding).copy()
        for bezier_pt_, rec_label_ in zip(center_pts_, rec_labels_):
            if bezier_pt_.numel() != 2:
                continue
            bezier_pt_ = bezier_pt_.numpy().reshape(-1, 2)
            bezier_pt_[:, 0] = bezier_pt_[:, 0]*(image_.shape[1]/tensors.shape[3])
            bezier_pt_[:, 1] = bezier_pt_[:, 1]*(image_.shape[0]/tensors.shape[2])
            image_ = draw_center_points(image_, bezier_pt_)
            image_ = draw_text(image_, rec_label_, bezier_pt_[0])
        vis_images.append(image_)

    return vis_images
