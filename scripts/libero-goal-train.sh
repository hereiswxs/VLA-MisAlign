#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset libero \
    --epoch 15 \
    --VLA_path ../openvla-main/openvla-7b-finetuned-libero-goal \
    --VLA_task_suite libero_goal \
    --batch_size 16 \
    --pre_train True \
    --pre_train_weight "/home/student/DongXiaorong/openvla-main/experiments/robot/history/day_2025_07_13/perturbation_generator.pth" \