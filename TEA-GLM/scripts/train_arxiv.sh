#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline


llm='./vicuna-7b-v1.5' # 'meta-llama/Llama-2-7b-hf'
seed=0
num_token=5
prefix='TEA-GLM_arxiv_pretrain-token5-alldata'
pretrain_gnn='GraphSAGE_arxiv_1000_tp.pth'


accelerate launch \
    --config_file accelerate_config/my_config_0.yaml \
    train_glm.py \
        --freeze_llama \
        --dataset arxiv \
        --pretrain_gnn $pretrain_gnn \
        --att_d_model 2048 \
        --gnn_output 4096 \
	    --grad_steps 2 \
        --batch_size 8 \
        --num_token $num_token \
        --clip_grad_norm 1.0 \
        --backbone $llm \
        --epoch 30 \
	    --weight_decay 0. \
        --max_text_length 700 \
        --gen_max_length 64 \
	    --lr 1e-4 \
        --prefix $prefix \
        --seed $seed \
        --best_epoch 29
