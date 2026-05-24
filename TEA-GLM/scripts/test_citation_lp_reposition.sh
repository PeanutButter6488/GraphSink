#!/bin/bash
#
# TEA-GLM LP — sink/non-sink swap reposition on arxiv / cora / pubmed.
# Mirrors scripts/test_citation_reposition.sh but with --task lp.
#
# Per sample, swap `num_swap` sink positions with `num_swap` non-sink positions
# inside the graph block, then run inference on the permuted inputs_embeds.
# Each seed picks a different (sink, non-sink) swap pairing.
#
# Prerequisite: baseline LP run that wrote
#   ./analysis/{test_dataset}_lp/global_stats/{prefix}_sink_records.jsonl
#
# Per-seed output:
#   ./results/{test_dataset}_lp/{prefix}_reposition_swap_k{num_swap}_seed{S}_model_{results,labels}.txt
#   ./analysis/{test_dataset}_lp/global_stats/{prefix}_reposition_swap_k{num_swap}_seed{S}_reposition_records.jsonl

export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

datasets=('arxiv:1024' 'cora:1024' 'pubmed:1024')

dataset='arxiv'
num_token=5
prefix='TEA-GLM_citation_meanpool'
llm='./vicuna-7b-v1.5'
best_epoch=49

num_swap=2
start_seed=42
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
        echo "[LP][${test_dataset}] reposition swap_sink_nonsink seed=${seed} num_swap=${num_swap}"
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
                --reposition_mode swap_sink_nonsink \
                --num_swap ${num_swap} \
                --sink_record_path "${sink_records}" \
                --reposition_seed ${seed}
    done
done
