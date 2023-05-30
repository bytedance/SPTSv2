#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python3  main.py \
        --train_dataset totaltext_train \
        --val_dataset totaltext_val \
        --max_length 25 \
        --data_root your_data_path \
        --batch_size 1 \
        --depths 6 \
        --lr 0.0005 \
        --pre_norm \
        --num_workers 8 \
        --eval \
        --resume your_weight_path \
        --output_dir your_output_path \
        --visualize \
        --padding_bins 0 \
        --pad_rec
