# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch.utils.data
import torchvision

from .ocr_dataset import build as build_ocr


def build_dataset(image_set, args):
    if args.dataset_file == 'ocr':
        return build_ocr(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
