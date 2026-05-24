#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

# Sink re-emergence experiment (research question 4) — top-2 / K=5 design.
#
# After pruning the TOP-2 detected sink tokens per sample, run an extra forward
# pass on the shortened sequence and check whether new sink tokens emerge among
# the remaining 3 graph tokens. Outputs are rank-indexed (rank 3..K by baseline
# sink score among survivors), not absolute K-position, so per-sample
# comparisons remain valid.
#
# Gate: pruning_mode=top2 AND --sink_reoccur (train_glm.py:249-256).
#
# Prerequisite: the baseline sink-records JSONL with the new
# `layer_token_scores` field must exist at
#   ./analysis/{test_dataset}/global_stats/{prefix}_seed42_sink_records.jsonl
# (we explicitly pass --sink_record_path below so the lookup does not fall
# back to the older unsuffixed file which lacks the field).
#
# Outputs written to ./analysis/{test_dataset}/global_stats/:
#   {prefix}_prune_top2_seed42_sink_reoccur_records.jsonl
#   {prefix}_prune_top2_seed42_sink_reoccur_summary.json
#   {prefix}_prune_top2_seed42_sink_reoccur_summary.md
#   {prefix}_prune_top2_seed42_sink_reoccur_promotion_by_rank.png
#   {prefix}_prune_top2_seed42_sink_reoccur_count_joint.png
#   {prefix}_prune_top2_seed42_sink_distribution_shift.png
# Top-2 prune + greedy decoding is deterministic, so a single run is sufficient.
# We use seed=42 to match the baseline sink-records file naming.

datasets=('arxiv:700' 'cora:700' 'pubmed:850')

dataset='arxiv'
num_token=5
prefix='TEA-GLM_citation_meanpool'
llm='./vicuna-7b-v1.5'

seed=42


for pair in "${datasets[@]}"
do
    IFS=':' read -r test_dataset max_text_length <<< "$pair"
    echo "Sink re-emergence (top-2): seed=$seed, max_text_length=$max_text_length, dataset=$test_dataset"
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
            --prune_sink_tokens \
            --pruning_mode top2 \
            --sink_record_path ./analysis/${test_dataset}/global_stats/${prefix}_seed42_sink_records.jsonl \
            --seed $seed \
            --sink_reoccur
done
