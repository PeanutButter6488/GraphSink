#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

# Graph-token shuffling experiment (research question 5).
#
# Per sample, swap `num_swap` sink positions with `num_swap` non-sink positions
# inside the graph block and run inference on the permuted inputs_embeds.
# Records per-sample swap positions + the K-space indices involved, so the
# aggregator can compute per-index prediction-change ratios vs a baseline run.
#
# Prerequisite: baseline sink-records JSONL at
#   ./analysis/{test_dataset}/global_stats/{prefix}_sink_records.jsonl
#
# Per-seed outputs:
#   ./results/{test_dataset}/{prefix}_reposition_swap_k{num_swap}_seed{S}_model_results.txt
#   ./results/{test_dataset}/{prefix}_reposition_swap_k{num_swap}_seed{S}_model_labels.txt
#   ./analysis/{test_dataset}/global_stats/{prefix}_reposition_swap_k{num_swap}_seed{S}_reposition_records.jsonl

datasets=('arxiv:700' 'cora:700' 'pubmed:850')

dataset='arxiv'
num_token=5
prefix='TEA-GLM_citation_meanpool'
llm='./vicuna-7b-v1.5'

num_swap=2
start_seed=42
n_seeds=5
end_seed=$((start_seed + n_seeds - 1))


for pair in "${datasets[@]}"
do
    IFS=':' read -r test_dataset max_text_length <<< "$pair"
    for seed in $(seq "$start_seed" "$end_seed"); do
        echo "Reposition: seed=$seed, num_swap=$num_swap, max_text_length=$max_text_length, dataset=$test_dataset"
        accelerate launch \
            --config_file accelerate_config/config_single_gpu.yaml \
            train_glm.py \
                --freeze_llama \
                --inference \
                --best_epoch 49 \
                --dataset $dataset \
                --test_dataset $test_dataset \
                --att_d_model 2048 \
                --gnn_output 4096 \
                --grad_steps 1 \
                --batch_size 4 \
                --num_token $num_token \
                --clip_grad_norm 1.0 \
                --backbone $llm \
                --epoch 1 \
                --weight_decay 0.1 \
                --max_text_length $max_text_length \
                --prefix $prefix \
                --reposition_mode swap_sink_nonsink \
                --num_swap $num_swap \
                --reposition_seed $seed
    done
done
