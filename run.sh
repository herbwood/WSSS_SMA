#!/bin/bash
EXP_NAME=sma
SAVEPATH=/PATH/TO/SAVE/WEIGHTS

CUDA_VISIBLE_DEVICES=0 python run_sample.py \
    --wandb=${EXP_NAME} \
    --description="sma baseline" \
    --train_cam_sma_pass \
    --make_cam_sma_pass \
    --eval_cam_pass \
    --lambda1=1 \
    --lambda2=0.3 \
    --lambda3=0.3 \
    --lambda4=0.2 \
    --target_epoch=6 \
    --cam_network=net.resnet50_dfa \
    --cam_num_epoches=10 \
    --log_name=logs/${EXP_NAME} \
    --cam_weights_name=${SAVEPATH}/${EXP_NAME}/sess/res50_cam_${EXP_NAME}.pth \
    --irn_weights_name=${SAVEPATH}/${EXP_NAME}/sess/res50_irn_${EXP_NAME}.pth \
    --cam_out_dir=${SAVEPATH}/${EXP_NAME}/result/cam \
    --ir_label_out_dir=${SAVEPATH}/${EXP_NAME}/result/ir_label \
    --sem_seg_out_dir=${SAVEPATH}/${EXP_NAME}/result/sem_seg \
    --ins_seg_out_dir=${SAVEPATH}/${EXP_NAME}/result/ins_seg \
    --cam_vis_out_dir=${SAVEPATH}/${EXP_NAME}/result/cam_vis