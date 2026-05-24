#!/bin/bash
#
# TEA-GLM LP — random non-sink pruning (k=2 by default) on arxiv / cora / pubmed.
# Mirrors scripts/test_citation_sinkanalysis_random.sh but with --task lp.
#
# Per sample, prune `num_prune` random non-sink graph tokens. --seed picks
# WHICH non-sink tokens to prune (compute_prune_positions_batch in
# utils/sink_pruning.py uses random.Random(seed ^ ...)), so varying the seed
# exercises different prune subsets.
#
# Prerequisite: baseline LP run that wrote
#   ./analysis/{test_dataset}_lp/global_stats/{prefix}_sink_records.jsonl
#
# Per-seed output (no rename needed, suffix already encodes seed):
#   ./results/{test_dataset}_lp/{prefix}_prune_random{num_prune}_seed{S}_model_{results,labels}.txt

export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

datasets=('arxiv:1024' 'cora:1024' 'pubmed:1024')

dataset='arxiv'
num_token=5
num_prune=2
prefix='TEA-GLM_citation_meanpool'
llm='./vicuna-7b-v1.5'
best_epoch=49

pruning_mode='random'
start_seed=123
n_seeds=5
end_seed=$((start_seed + n_seeds - 1))

# scripts/test_citation_lp.sh runs the baseline with --append_seed_suffix, so
# the sink records land at {prefix}_seed{baseline_seed}_sink_records.jsonl.
# Override BASELINE_SEED if your baseline used a different seed.
baseline_seed="${BASELINE_SEED:-42}"

for pair in "${datasets[@]}"; do
    IFS=':' read -r test_dataset max_text_length <<< "$pair"

    sink_records="./analysis/${test_dataset}_lp/global_stats/${prefix}_seed${baseline_seed}_sink_records.jsonl"
    if [ ! -f "$sink_records" ]; then
        echo "[${test_dataset}] Missing LP sink records: $sink_records"
        echo "Run scripts/test_citation_lp.sh first to produce them. Skipping."
        continue
    fi

    for seed in $(seq "$start_seed" "$end_seed"); do
        echo "=================================================="
        echo "[LP][${test_dataset}] pruning_mode=${pruning_mode} seed=${seed} num_prune=${num_prune}"
        echo "max_text_length=${max_text_length}"
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
                --prune_sink_tokens \
                --pruning_mode ${pruning_mode} \
                --num_prune ${num_prune} \
                --sink_record_path "${sink_records}" \
                --seed ${seed}
    done
done
