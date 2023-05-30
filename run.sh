#!/bin/bash

bs=$1
lr=$2
out=$3
data=$4

python3 -m torch.distributed.launch --master_port=3141 --nproc_per_node 8 --use_env main.py --data_root ${data} --batch_size ${bs} --lr ${lr} --output_dir ${out} \
        --train_dataset totaltext_train:ic13_train:ic15_train:mlt_train:syntext1_train:syntext2_train \
        --val_dataset totaltext_val \
        --dec_layers 6 \
        --max_length 25 \
        --pad_rec \
        --pre_norm \
        --rotate_prob 0.3 \
        --train \
        --depths 6 \
        --padding_bins 0 \
        --epochs 280 \
        --warmup_epochs 5 