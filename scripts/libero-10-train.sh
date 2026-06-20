#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset libero \
    --epoch 15 \
    --VLA_path ../openvla-main/openvla-7b-finetuned-libero-object \
    --VLA_task_suite libero_10 \
    --batch_size 16 \
    --pre_train True \
    --pre_train_weight "" \