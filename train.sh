#!/bin/bash

exp_folder="exps/exps016"
mkdir -p $exp_folder

CUDA_VISIBLE_DEVICES=1 nohup \
python -u main.py \
    --deterministic \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --epochs 1000 \
    --val_frequency 2 \
    --debug_frequency 2 \
    --num_debug_save 5 \
    --max_grad_norm 1.0 \
    --image_shape '32,64,64' \
    --node_len 3 \
    --base_channels 24 \
    --num_classes 4 \
    --batch_size 16 \
    --seed 1025 \
    --save_folder ${exp_folder}/debug \
    --data_file '/home/lyf/Research/auto_trace/neuron2seq/data/Task002_ntt_256/debug_nsample16.pkl' \
    > ${exp_folder}/train.log &
