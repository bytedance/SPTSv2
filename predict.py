# Copyright (2023) Bytedance Ltd. and/or its affiliates
import time
import json
import torch
import random
import argparse
import datetime
import numpy as np
from pathlib import Path
from PIL import Image
import datasets.sptsv2_transforms as T
import cv2
import util.misc_sptsv2 as utils
from util.data import process_args
from models import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('Set SPTSv2', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    # Data parameters
    parser.add_argument('--bins', type=int, default=1000)
    parser.add_argument('--chars', type=str, default=' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')
    parser.add_argument('--padding_bins', type=int, default=0)
    parser.add_argument('--num_box', type=int, default=60)
    parser.add_argument('--pts_key', type=str, default='center_pts')
    parser.add_argument('--no_known_char', type=int, default=95)
    parser.add_argument('--pad_rec_index', type=int, default=96)
    parser.add_argument('--pad_rec', action='store_true')
    parser.add_argument('--max_size_test', type=int, default=1824)
    parser.add_argument('--min_size_test', type=int, default=1000)
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--window_size', default=5, type=int,
                        help="swin transformer window size")
    parser.add_argument('--obj_num', default=60, type=int,
                        help="number of text lines in training stage") 
    parser.add_argument('--max_length', default=25, type=int,
                        help="number of text lines in training stage")                                      
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--depths', default=6, type=int,
                        help="swin transformer structure")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--img_path', default="", type=str, help='path for the image to be detected')
    return parser

def main(args):
    args = process_args(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, criterion = build_model(args)
    model.to(device)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    image = Image.open(args.img_path)
    image = image.convert('RGB')

    w_ori,h_ori = image.size

    #transform
    transform = T.Compose([
        T.RandomResize([args.min_size_test], args.max_size_test),
        T.ToTensor(),
        T.Normalize(None, None)
    ])

    image_new = transform(image,None)

    c,h,w = image_new[0].shape
    image_new = image_new[0].view(1,c,h,w).to(device)
    seq = torch.ones(1, 1).to(device,dtype=torch.long) * args.start_index
    model.eval()

    # get predictions
    output = model(image_new,seq,seq)
    outputs, values, _ = output
    N = (outputs[0].shape[0])//27
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for i in range(N):
        v = values[0][27*i:(27)*i+27].mean().item()
        if v > 0.922:
            text = ''
            pts_x = outputs[0][27*i].item() * (float(w_ori) / 1000)
            pts_y = outputs[0][27*i+1].item() * (float(h_ori) / 1000)
            for c in outputs[0][27*i+2:27*i+27].tolist():
                if 1000 < c <1096:
                        text += args.chars[c-1000]
                else:
                    break
            cv2.circle(img, (int(pts_x), int(pts_y)), 3, (255, 0, 0), -1)
            cv2.putText(img, text, (int(pts_x), int(pts_y)), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
    
    cv2.imwrite('test_'+args.img_path.split('/')[-1],img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SPTSv2 yyds', parents=[get_args_parser()])
    args = parser.parse_args()
    args.img_path = 'IMG/0000245.jpg'
    args.resume = 'your_weight_path'
    args.pre_norm = True
    args.pad_rec = True
    main(args)
