# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from pathlib import Path

import os
import math
import random
import torch
from torch.utils import data
import torch.utils.data
from torch.utils.data import dataset
import torchvision
from torch.utils.data import ConcatDataset
import numpy as np
import datasets.sptsv2_transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, dataset_name, max_length):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, dataset_name, max_length)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img1, target1 = self._transforms(img, target)
            img2, target2 = self._transforms(img, target)
        return img1, img2, target1, target2


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, dataset_name='', max_length=25):
        self.return_masks = return_masks
        self.dataset_name = dataset_name
        self.max_length = max_length

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        target['dataset_name'] = self.dataset_name


        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        recog = [obj['rec'][:self.max_length] for obj in anno]
        recog = torch.tensor(recog, dtype=torch.long).reshape(-1, self.max_length)
        target["rec"]  = recog

        bezier_pts = [obj['bezier_pts'] for obj in anno]
        bezier_pts = torch.tensor(bezier_pts, dtype=torch.float32).reshape(-1, 16)
        target['bezier_pts'] = bezier_pts
        center_pts = torch.zeros(bezier_pts.shape[0], 2)
        for i in range(bezier_pts.shape[0]):
            tmp = bezier_pts[i]
            if tmp[-1].item() == 0 and tmp[-2].item() == 0:
                xc = 0; yc = 0; count = 0
                tmp = tmp.view(-1, 2)
                for j in range(tmp.shape[0]):
                    if tmp[j][0].item() == 0 and tmp[j][1].item() == 0:
                        continue
                    else:
                        xc += tmp[j][0].item(); yc += tmp[j][1].item(); count += 1
                if count > 0:        
                    xc = xc/count
                    yc = yc/count
                else:
                    xc = 0; yc = 0
            else:
                polygon = bezier_to_polygon(tmp)
                length = int(len(polygon)/2)
                top = torch.tensor(polygon[:length])
                botton = torch.tensor(polygon[length:])
                count = 0
                x1 = 0; x2 = 0; y1 = 0; y2 = 0
                for j in range(length):
                    if top[j][0] == 0 and top[j][1] == 0:
                        continue
                    else:
                        x1 += top[j][0].item(); y1 += top[j][1].item(); count += 1
                if count > 0:
                    xt = x1/count; yt = y1/count
                else:
                    xt = 0;yt = 0
                count = 0
                x1 = 0; x2 = 0; y1 = 0; y2 = 0
                for j in range(length):
                    if botton[j][0] == 0 and botton[j][1] == 0:
                        continue
                    else:
                        x2 += botton[j][0].item(); y2 += botton[j][1].item(); count += 1
                if count > 0:        
                    xb = x2/count; yb = y2/count   
                else:
                    xb = 0;yb = 0
                xc = (xt + xb)/2
                yc = (yt + yb)/2      

            # xc_new, yc_new = dynamic_point((xc, yc), 10)   
            # xc_new = min(max(0, xc_new), w) 
            # yc_new = min(max(0, yc_new), h)   
            # center_pts[i][0] = xc_new
            # center_pts[i][1] = yc_new
            center_pts[i][0] = xc
            center_pts[i][1] = yc            
        
        target['center_pts'] = center_pts
        assert target['center_pts'].shape[0] == target['bezier_pts'].shape[0]
        return image, target

def dynamic_point(pts, radius):
    theta = random.uniform(0, 1)*2*math.pi
    x_new = pts[0] + radius*math.cos(theta)
    y_new = pts[1] - radius*math.sin(theta)
    return x_new, y_new


def bezier_to_polygon(bezier):
    u = np.linspace(0, 1, 20)
    bezier = np.array(bezier)
    bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
    points = np.outer((1 - u) ** 3, bezier[:, 0]) \
        + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
        + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
        + np.outer(u ** 3, bezier[:, 3])
    
    # convert points to polygon
    points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)
    return points.tolist()

def make_coco_transforms(image_set, max_size_train, min_size_train, max_size_test, min_size_test,
                         crop_min_ratio, crop_max_ratio, crop_prob, rotate_max_angle, rotate_prob,
                         brightness, contrast, saturation, hue, distortion_prob):

    transforms = []
    if image_set == 'train':
        transforms.append(T.RandomSizeCrop(crop_min_ratio, crop_max_ratio, True, crop_prob))
        transforms.append(T.RandomRotate(rotate_max_angle, rotate_prob))
        transforms.append(T.RandomResize(min_size_train, max_size_train))
        transforms.append(T.RandomDistortion(brightness, contrast, saturation, hue, distortion_prob))
    if image_set == 'val':
        transforms.append(T.RandomResize([min_size_test], max_size_test))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(None, None))

    return T.Compose(transforms)



def build(image_set, args):
    root = Path(args.data_root)
    if image_set == 'train':
        dataset_names = args.train_dataset.split(':')
    elif image_set == 'val':
        dataset_names = args.val_dataset.split(':')
    
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name == 'totaltext_train':
            img_folder = root / "totaltext" / "train_images"; ann_file = root / "totaltext" / "train.json"
        elif dataset_name == 'totaltext_val':
            img_folder = root / "totaltext" / "test_images"; ann_file = root / "totaltext" / "test.json"
        elif dataset_name == 'mlt_train':
            img_folder = root / "mlt2017" / "MLT_train_images"; ann_file = root / "mlt2017" / "train.json"
        elif dataset_name == 'ctw1500_train':
            img_folder = root / "CTW1500" / "ctwtrain_text_image"; ann_file = root / "CTW1500" / "annotations" / "train_ctw1500_maxlen25_v2.json"
        elif dataset_name == 'ctw1500_val':
            img_folder = root / "CTW1500" / "ctwtest_text_image"; ann_file = root / "CTW1500" / "annotations" / "test_ctw1500_maxlen25.json"
        elif dataset_name == 'syntext1_train':
            img_folder = root / "syntext1" / "syntext_word_eng"; ann_file = root / "syntext1" / "train.json"
        elif dataset_name == 'syntext2_train':
            img_folder = root / "syntext2" / "emcs_imgs"; ann_file = root / "syntext2" / "train.json"
        elif dataset_name == 'cocotextv2_train':
            img_folder = root / "cocotextv2" / "train2014"; ann_file = root / "cocotextv2" / "cocotext.v2.rewrite.json"
        elif dataset_name == 'ic13_train':
            img_folder = root / "icdar2013" / "train_images"; ann_file = root / "icdar2013" / "ic13_train.json"
        elif dataset_name == 'ic15_train':
            img_folder = root / "icdar2015" / "train_images"; ann_file = root / "icdar2015" / "ic15_train.json"
        elif dataset_name == 'ic13_val':
            img_folder = root / "icdar2013" / "test_images"; ann_file = root / "icdar2013" / "ic13_test.json"
        elif dataset_name == 'ic15_val':
            img_folder = root / "icdar2015" / "test_images"; ann_file = root / "icdar2015" / "ic15_test.json"
        elif dataset_name == 'inversetext':
            img_folder = root / "inversetext" / "test_images"; ann_file = root / "inversetext" / "test_poly.json"
        else:
            raise NotImplementedError
        
        transforms = make_coco_transforms(image_set, args.max_size_train, args.min_size_train,
              args.max_size_test, args.min_size_test, args.crop_min_ratio, args.crop_max_ratio,
              args.crop_prob, args.rotate_max_angle, args.rotate_prob, args.brightness, args.contrast,
              args.saturation, args.hue, args.distortion_prob)
        dataset = CocoDetection(img_folder, ann_file, transforms=transforms, return_masks=args.masks, dataset_name=dataset_name, max_length=args.max_length)
        datasets.append(dataset)
    
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return dataset
