# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import cv2
import PIL
import torch
import random
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.misc_sptsv2 import interpolate

def crop(image, target, region):
    rg_ymin, rg_xmin, rg_h, rg_w = region
    rg_xmax = rg_xmin + rg_w
    rg_ymax = rg_ymin + rg_h 

    boxes = target['boxes'].clone()
    pre_keep = torch.zeros((boxes.shape[0]), dtype=torch.bool)
    while True:
        ov_xmin = torch.clamp(boxes[:, 0], min=rg_xmin)
        ov_ymin = torch.clamp(boxes[:, 1], min=rg_ymin)
        ov_xmax = torch.clamp(boxes[:, 2], max=rg_xmax)
        ov_ymax = torch.clamp(boxes[:, 3], max=rg_ymax)
        ov_h = ov_ymax - ov_ymin
        ov_w = ov_xmax - ov_xmin 
        keep = torch.bitwise_and(ov_w>0, ov_h>0)
        if (keep == False).all():
            return None, None
        if keep.equal(pre_keep):
            break

        keep_boxes = boxes[keep]
        img_h, img_w = target["size"]

        rg_xmin = rg_xmin if rg_xmin < min(keep_boxes[:, 0]) else int(max(min(keep_boxes[:, 0]), 0))
        rg_ymin = rg_ymin if rg_ymin < min(keep_boxes[:, 1]) else int(max(min(keep_boxes[:, 1]), 0))
        rg_xmax = rg_xmax if rg_xmax > max(keep_boxes[:, 2]) else int(min(max(keep_boxes[:, 2]), img_w-1))
        rg_ymax = rg_ymax if rg_ymax > max(keep_boxes[:, 3]) else int(min(max(keep_boxes[:, 3]), img_h-1))

        pre_keep = keep.clone()

    region = (rg_ymin, rg_xmin, rg_ymax-rg_ymin, rg_xmax-rg_xmin)
    cropped_image = F.crop(image, *region)

    target['size'] = torch.as_tensor([rg_ymax-rg_ymin, rg_xmax-rg_xmin])
    fields = ['labels', 'area', 'iscrowd', 'rec']
    if 'boxes' in target:
        boxes = target['boxes']
        cropped_boxes = boxes - torch.as_tensor([rg_xmin, rg_ymin]*2)
        target['boxes'] = cropped_boxes
        fields.append('boxes')
    
    if 'masks' in target:
        target['masks'] = target['masks'][:, rg_ymin:rg_ymax, rg_xmin:rg_xmax]
        fields.append('masks')

    if 'bezier_pts' in target:
        bezier_pts = target['bezier_pts']
        cropped_beizer_pts = bezier_pts - torch.as_tensor([rg_xmin, rg_ymin]*8)
        target['bezier_pts'] = cropped_beizer_pts
        fields.append('bezier_pts')

    if 'center_pts' in target:
        center_point = target['center_pts']
        cropped_center_point = center_point - torch.as_tensor([rg_xmin, rg_ymin])
        target['center_pts'] = cropped_center_point
        fields.append('center_pts')       

    for field in fields:
        target[field] = target[field][keep]

    return cropped_image, target

def crop_(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region
    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    if "bezier_pts" in target:
        bezier_pts = target['bezier_pts']
        scaled_bezier_pts = bezier_pts * torch.as_tensor([ratio_width, ratio_height] * 8)
        target['bezier_pts'] = scaled_bezier_pts

    if "center_pts" in target:
        center_point = target['center_pts']
        scaled_center_point = center_point * torch.as_tensor([ratio_width, ratio_height])
        target['center_pts'] = scaled_center_point
    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, is_ratio: bool, prob: float):
        self.min_size = min_size
        self.max_size = max_size
        self.is_ratio = is_ratio
        self.prob = prob

    def __call__(self, img: PIL.Image.Image, target: dict):
        if random.random() < self.prob and len(target['boxes']) > 0:
            max_try = 100
            for _ in range(max_try):
                if not self.is_ratio:
                    w = random.randint(self.min_size, min(img.width, self.max_size))
                    h = random.randint(self.min_size, min(img.height, self.max_size))
                else:
                    w = int(img.width * random.uniform(self.min_size, self.max_size))
                    h = int(img.height * random.uniform(self.min_size, self.max_size))
                region = T.RandomCrop.get_params(img, [h, w])
                img_, target_ = crop(img.copy(), target.copy(), region)
                if not img_ is None:
                    return img_, target_
            print('Can not be cropped')
        return img, target

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        # image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        if "bezier_pts" in target:
            target['bezier_pts'] = target['bezier_pts'] / torch.tensor([w, h]*8, dtype=torch.float32)
        if "center_pts" in target:
            target['center_pts'] = target['center_pts'] / torch.tensor([w, h], dtype=torch.float32)    
        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

def read_word_list(word_list_file):
    words = open(word_list_file, 'r').read().splitlines()
    words = words[1:]
    words = [word.split('\t')[0] for word in words]
    words = [word.split('/')[0] for word in words]
    return words

class RandomRotate(object):
    def __init__(self, max_angle, prob):
        self.max_angle = max_angle
        self.prob = prob 
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            angle = random.uniform(-self.max_angle, self.max_angle)
            image_w, image_h = image.size
            center_pt = (image_w//2, image_h//2)
            rotation_matrix = cv2.getRotationMatrix2D(center_pt, angle, 1)
            image = image.rotate(angle, expand=True)

            cur_w, cur_h = image.size
            target['size'] = torch.as_tensor([cur_h, cur_w])
            pad_w = (cur_w - image_w) / 2; pad_h = (cur_h - image_h) / 2

            center_point = target['center_pts'].numpy().copy()
            center_point = center_point.reshape(-1, 1, 2)
            center_point = np.pad(center_point, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1)
            center_point = np.dot(center_point, rotation_matrix.transpose())
            center_point[:, :, 0] += pad_w; center_point[:, :, 1] += pad_h
            center_point = center_point.reshape(-1, 2)
            target['center_pts'] = torch.from_numpy(center_point).to(target['center_pts'])
           
            boxes = target['boxes'].numpy().copy()
            boxes = boxes.reshape(-1, 2, 2)
            boxes = np.pad(boxes, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1)
            boxes = np.dot(boxes, rotation_matrix.transpose())
            boxes[:, :, 0] += pad_w; boxes[:, :, 1] += pad_h
            boxes = boxes.reshape(-1, 4)
            target['boxes'] = torch.from_numpy(boxes).to(target['boxes'])

            bezier_pts = target['bezier_pts'].numpy().copy()
            bezier_pts = bezier_pts.reshape(-1, 8, 2)
            bezier_pts = np.pad(bezier_pts, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1)
            bezier_pts = np.dot(bezier_pts, rotation_matrix.transpose())
            bezier_pts[:, :, 0] += pad_w; bezier_pts[:, :, 1] += pad_h
            bezier_pts = bezier_pts.reshape(-1, 16)
            target['bezier_pts'] = torch.from_numpy(bezier_pts).to(target['bezier_pts'])

        return image, target 

class RandomDistortion(object):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0.5):
        self.prob = prob
        self.tfm = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, target=None):
        if np.random.random() < self.prob:
            return self.tfm(img), target
        else:
            return img, target

class NoisyLabel(object):
    def __init__(self, bins, padding_bins, num_box, start_index, padding_index, 
                 end_index, no_known_index, noise_index, category_start_index,
                 pts_key, label_key, pad_rec_index, word_list_file, characters,
                 pad_rec):
        self.bins = bins
        self.padding_bins = padding_bins
        self.num_box = num_box
        self.start_index = start_index
        self.padding_index = padding_index
        self.end_index = end_index
        self.no_known_index = no_known_index
        self.noise_index = noise_index
        self.category_start_index = category_start_index
        self.pts_key = pts_key
        self.label_key = label_key
        self.pad_rec_index= pad_rec_index
        self.characters = characters
        self.pad_rec = pad_rec
        if not word_list_file is None:
            self.words = read_word_list(word_list_file)
        else:
            self.words = None

    def __call__(self, targets):
        input_seqs_ = []; output_seqs_ = []
        input_box_seqs_ = []
        input_label_seqs_ = []
        output_seq_box_ = []
        output_seq_label_ = []
        for target in targets:
            if target[self.pts_key].shape[0] > self.num_box - 2:
                keep = random.sample(range(target[self.pts_key].shape[0]), self.num_box-2)
                keep = torch.as_tensor(keep)
                target[self.pts_key] = target[self.pts_key][keep]
                target[self.label_key] = target[self.label_key][keep]
            box = (target[self.pts_key] * self.bins).floor().int() + self.padding_bins
            box = torch.clamp(box, min=0, max=self.category_start_index-1).long()
            label = target[self.label_key] + self.category_start_index
            # label[label == self.category_start_index + 95] = self.no_known_index # preserve oov characters
            label = label.long()
            box_label = torch.cat([box, label], dim=-1)
            idx = torch.randperm(box_label.shape[0])
            box = box[idx]
            label = label[idx]
            box_label = box_label[idx]
            num_random_box = self.num_box - box_label.shape[0]
            random_box = torch.rand(num_random_box, box.shape[1]).to(target[self.pts_key])
            padding_bins_ratio = self.padding_bins / self.bins 
            random_box = random_box * (1 + 2*padding_bins_ratio) - padding_bins_ratio
            random_box = (random_box * self.bins).floor().int() + self.padding_bins
            label_lens = []
            if self.label_key == 'rec':
                random_label = torch.ones(num_random_box, label.shape[1]).to(label) * self.pad_rec_index
                if not self.words is None:
                    random_word_indices = torch.randint(0, len(self.words)-1, (num_random_box,))
                    for i, index in enumerate(random_word_indices):
                        word = self.words[index]
                        label_lens.append(len(word))
                        char_indices = torch.tensor([self.characters.index(ele) for ele in word]).to(random_label)
                        random_label[i, :len(char_indices)].copy_(char_indices)
                else:
                    for i in range(num_random_box):
                        word_len = random.randint(0, label.shape[1])
                        random_word = torch.randint(0, len(self.characters), (word_len,)).to(random_label)
                        random_label[i, :word_len] = random_word
            else:
                raise NotImplementedError
            random_label = random_label + self.category_start_index
            random_label = random_label.long()
            random_box = random_box.long()
            random_box_label_input = torch.cat([random_box, random_label], dim=-1)
            random_box_target = torch.ones(num_random_box, box.shape[1]).to(box_label) * self.no_known_index
            random_label_target = torch.ones(num_random_box, label.shape[1]).to(box_label) * self.no_known_index
            random_label_target[:, -1] = self.noise_index
            input_box_seq = torch.cat([torch.ones(1).to(box) * self.start_index, box.flatten(),random_box.flatten()])
            input_box_seqs_.append(input_box_seq)
            input_label_seq = torch.cat([box_label, random_box_label_input], dim=0)
            input_label_seqs_.append(input_label_seq.flatten())

            output_seq_box = torch.cat([box.flatten(), torch.ones(1).to(box_label)*self.end_index, random_box_target.flatten()])
            output_seq_box_.append(output_seq_box)
            output_seq_label = torch.cat([label, random_label_target], dim=0)     
            output_seq_label_.append(output_seq_label.flatten())      

        max_seq_len = max([len(seq) for seq in input_box_seqs_])
        input_box_seqs = torch.ones(len(input_box_seqs_), max_seq_len).to(input_box_seq) * self.padding_index
        max_seq_len = max([len(seq) for seq in input_label_seqs_])
        input_label_seqs = torch.ones(len(input_label_seqs_), max_seq_len).to(input_box_seq) * self.padding_index
        max_seq_len = max([len(seq) for seq in output_seq_box_])
        output_box_seqs = torch.ones(len(output_seq_box_), max_seq_len).to(output_seq_box) * self.padding_index
        max_seq_len = max([len(seq) for seq in output_seq_label_])
        output_label_seqs = torch.ones(len(output_seq_label_), max_seq_len).to(output_seq_box) * self.padding_index

        for i in range(output_label_seqs.shape[0]):
            input_box_seqs[i, :len(input_box_seqs_[i])].copy_(input_box_seqs_[i])
            input_label_seqs[i, :len(input_label_seqs_[i])].copy_(input_label_seqs_[i])
            output_box_seqs[i, :len(output_seq_box_[i])].copy_(output_seq_box_[i])
            output_label_seqs[i, :len(output_seq_label_[i])].copy_(output_seq_label_[i])

        return input_box_seqs,input_label_seqs, output_box_seqs, output_label_seqs