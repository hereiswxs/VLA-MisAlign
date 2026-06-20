#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset libero \
    --epoch 15 \
    --VLA_path ../openvla-main/openvla-7b-finetuned-libero-object \
    --VLA_task_suite libero_object \
    --batch_size 16 \
    --pre_train False \
    --pre_train_weight /home/student/DongXiaorong/Madv_VLA/log_ours/libero/libero_object/EVAL-libero_object-2025_08_31-17_56_26/codes/libero_object_perturbation_generator.pth \