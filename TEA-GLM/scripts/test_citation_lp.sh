#!/bin/bash
#
# Run TEA-GLM link-prediction inference + RQ1/RQ2 sink analysis on
# arxiv / cora / pubmed using the existing NC-trained citation checkpoint.
#
# Requires the TEA-GLM patch that adds --task lp routing:
#   - config.py:        new --task {nc,lp} flag
#   - utils/instruction_preprocess.py: loads {ds}_LP_dataset_{mode}.json when task=lp
#   - train_glm.py:     writes outputs to ./analysis/{ds}_lp/... and
#                       ./results/{ds}_lp/... so they don't overwrite NC artifacts
#
# Sink analysis runs unconditionally in the eval branch when no pruning/
# reposition flag is set (train_glm.py:303 onward), so this script just
# enables --inference and lets the script populate:
#   ./analysis/{ds}_lp/global_stats/
#       {prefix}_sink_records.jsonl                         (per-sample sinks)
#       {prefix}_sink_dim_summary.json                      (RQ1)
#       {prefix}_sink_dim_mean_activation.png               (RQ1)
#       {prefix}_topdims_all_activation.png                 (RQ1, all)
#       {prefix}_topdims_sink_only_activation.png           (RQ1, sink-only)
#       {prefix}_query_to_graph_attention.{json,png}        (RQ2)
#       {prefix}_sink_token_histogram.png                   (sink position dist.)
#   ./results/{ds}_lp/{prefix}_model_results.txt
#   ./results/{ds}_lp/{prefix}_model_labels.txt
#
# Note: the bundled checkpoint (TEA-GLM_citation_meanpool, epoch 49) was
# trained for NC, so LP yes/no accuracy will be poor; sink structure is
# still meaningful because RQ1/RQ2 are about hidden-state geometry.

export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

# pair = test_dataset:max_text_length (LP prompts contain two abstracts;
# 1024 covers cora/pubmed comfortably and arxiv almost always; bump if
# you see truncation warnings from preprocess()).
datasets=('arxiv:1024' 'cora:1024' 'pubmed:1024')

dataset='arxiv'                         # training-side dataset (controls gnn_input)
num_token=5
prefix='TEA-GLM_citation_meanpool'
llm='./vicuna-7b-v1.5'
seed=42
best_epoch=49

for pair in "${datasets[@]}"; do
    IFS=':' read -r test_dataset max_text_length <<< "$pair"

    echo "=================================================="
    echo "TEA-GLM LP | test_dataset=${test_dataset} | max_text_length=${max_text_length}"
    echo "Outputs:    analysis/${test_dataset}_lp/  results/${test_dataset}_lp/"
    echo "=================================================="

    accelerate launch \
        --config_file accelerate_config/config_single_gpu.yaml \
        train_glm.py \
            --freeze_llama \
            --inference \
            --task lp \
            --best_epoch ${best_epoch} \
            --dataset ${dataset} \
            --test_dataset ${test_dataset} \
            --att_d_model 2048 \
            --gnn_output 4096 \
            --grad_steps 1 \
            --batch_size 4 \
            --num_token ${num_token} \
            --clip_grad_norm 1.0 \
            --backbone ${llm} \
            --epoch 1 \
            --weight_decay 0.1 \
            --max_text_length ${max_text_length} \
            --prefix ${prefix} \
            --seed ${seed} \
            --append_seed_suffix
done
