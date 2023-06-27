# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import time
import json
import torch
import random
import argparse
import datetime
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc_sptsv2 as utils
from util.data import process_args
from datasets import build_dataset
from engine_sptsv2 import evaluate, train_one_epoch
from models import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('Set SPTSv2', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--code_dir', type=str, default='.')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--warmup_min_lr', type=float, default=0.0001)
    parser.add_argument('--min_lr', type=float, default=0.00001)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--finetune', action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # Data parameters
    parser.add_argument('--bins', type=int, default=1000)
    parser.add_argument('--chars', type=str, default=' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')
    parser.add_argument('--padding_bins', type=int, default=0)
    parser.add_argument('--num_box', type=int, default=60)
    parser.add_argument('--pts_key', type=str, default='center_pts')
    parser.add_argument('--no_known_char', type=int, default=95)
    parser.add_argument('--pad_rec_index', type=int, default=96)
    parser.add_argument('--pad_rec', action='store_true')
    parser.add_argument('--dict_name', type=str, default='en_US.dic')
    parser.add_argument('--use_dict', action='store_true')
    parser.add_argument('--max_size_train', type=int, default=1600)
    parser.add_argument('--min_size_train', type=int, nargs='+', default=[640, 672, 704, 736, 768, 800, 832, 864, 896])
    parser.add_argument('--max_size_test', type=int, default=1824)
    parser.add_argument('--min_size_test', type=int, default=1024)
    parser.add_argument('--crop_min_ratio', type=float, default=0.5)
    parser.add_argument('--crop_max_ratio', type=float, default=1.0)
    parser.add_argument('--crop_prob', type=float, default=1.0)
    parser.add_argument('--rotate_max_angle', type=int, default=30)
    parser.add_argument('--rotate_prob', type=float, default=0.3)
    parser.add_argument('--brightness', type=float, default=0.5)
    parser.add_argument('--contrast', type=float, default=0.5)
    parser.add_argument('--saturation', type=float, default=0.5)
    parser.add_argument('--hue', type=float, default=0.5)
    parser.add_argument('--distortion_prob', type=float, default=0.5)

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
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--transformer_type', type=str, default='vanilla', help='vanilla, linear')

    # dataset parameters
    parser.add_argument('--dataset_file', default='ocr')
    parser.add_argument('--train_dataset', type=str)
    parser.add_argument('--val_dataset', type=str)
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--local_rank", type=int, default=0)

    return parser


def main(args):
    utils.init_distributed_mode(args)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    args = process_args(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, criterion = build_model(args)
    model.to(device)
    print(model)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
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
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False) if not dataset_val is None else None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if not dataset_val is None else None

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn(args), num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn(args), num_workers=args.num_workers) if not dataset_val is None else None

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        if dataset_val is None:
            print('Wrong validating dataset!')
            return 
        evaluate(model, criterion,
                 data_loader_val, device, 
                 args.output_dir, args.chars, 
                 args.start_index, args.visualize)
        return

    print("Start training")
    start_time = time.time()
    if not args.finetune:
        warmup_lr = [args.warmup_min_lr + ((args.lr - args.warmup_min_lr) * i / args.warmup_epochs) for i in range(args.warmup_epochs)]
        decay_lr = [max(i * args.lr / args.epochs, args.min_lr) for i in range(args.epochs - args.warmup_epochs)]
        decay_lr.reverse()
        learning_rate_schedule = warmup_lr + decay_lr
    else:
        learning_rate_schedule = [args.lr] * args.epochs
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, learning_rate_schedule, args.print_freq, args.max_length)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SPTSv2 yyds', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
