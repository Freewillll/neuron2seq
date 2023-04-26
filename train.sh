#!/bin/bash

exp_folder="exps/exps005"
mkdir -p $exp_folder

CUDA_VISIBLE_DEVICES=0 nohup \
python -u main.py \
    --deterministic \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --epochs 20 \
    --val_frequency 10 \
    --debug_frequency 10 \
    --num_debug_save 5 \
    --max_grad_norm 1.0 \
    --image_shape '32,64,64' \
    --node_len 3 \
    --base_channels 24 \
    --num_classes 4 \
    --batch_size 16 \
    --seed 1025 \
    --save_folder ${exp_folder}/debug \
    --data_file '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task002_ntt_256/data_splits.pkl' \
    > ${exp_folder}/train.log &