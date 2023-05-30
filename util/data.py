# Copyright (2023) Bytedance Ltd. and/or its affiliates
def process_args(args):
    args.category_start_index = args.bins + args.padding_bins*2 
    num_char_classes = len(args.chars) + 1
    if args.pad_rec: 
        num_char_classes += 1
    args.end_index = args.category_start_index + num_char_classes
    args.start_index = args.end_index + 1 
    args.noise_index = args.start_index + 1 
    args.padding_index = args.noise_index + 1
    args.no_known_index = args.padding_index
    return args