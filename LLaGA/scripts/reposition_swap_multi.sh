#!/bin/bash

set -euo pipefail

model_path="${MODEL_PATH:-Runjin/llaga-vicuna-7b-simteg-ND-general_model-2-layer-mlp-projector}"
model_base="${MODEL_BASE:-lmsys/vicuna-7b-v1.5-16k}"
mode="${CONV_MODE:-v1}"
datasets="${DATASETS:-cora pubmed arxiv}"   # space-separated
task="${TASK:-nc}"
emb="${EMB:-simteg}"
use_hop="${USE_HOP:-2}"
sample_size="${SAMPLE_SIZE:-10}"
template="${TEMPLATE:-ND}"

num_swap="${NUM_SWAP:-2}"
n_runs="${N_RUNS:-5}"
run_start="${RUN_START:-0}"
seed_base="${SEED_BASE:-42}"
max_samples="${MAX_SAMPLES:-500}"
cache_dir="${CACHE_DIR:-./checkpoint}"
output_dir="${OUTPUT_DIR:-results_phc3mn}"

mkdir -p "${output_dir}"

for dataset in ${datasets}; do
  base_output_name="${dataset}_${task}_${template}_predictions.jsonl"
  output_path="${output_dir}/${base_output_name}"
  sink_records_path="analysis/${dataset}_${template}/sink_records.jsonl"

  if [[ ! -f "${sink_records_path}" ]]; then
    echo "[${dataset}] Missing sink records: ${sink_records_path}"
    echo "Generate sink_records.jsonl first before running swap. Skipping."
    continue
  fi

  echo "=================================================="
  echo "[${dataset}] swap_sink_nonsink reposition over ${n_runs} runs"
  echo "num_swap: ${num_swap} | seed_base: ${seed_base} | run_start: ${run_start}"
  echo "(per-run reposition_seed = seed_base + run_idx; each run picks a different random sink/nonsink swap)"
  echo "max_samples: ${max_samples} | base output: ${output_path}"
  echo "=================================================="

  for ((i=0; i<n_runs; i++)); do
    run_idx=$((run_start + i))
    seed=$((seed_base + run_idx))
    echo ""
    echo ">>> [${dataset}] Run $((i+1))/${n_runs} (run_idx=${run_idx}, seed=${seed})"
    python eval/eval_pretrain.py \
      --model_path "${model_path}" \
      --model_base "${model_base}" \
      --conv_mode "${mode}" \
      --dataset "${dataset}" \
      --pretrained_embedding_type "${emb}" \
      --use_hop "${use_hop}" \
      --sample_neighbor_size "${sample_size}" \
      --answers_file "${output_path}" \
      --task "${task}" \
      --cache_dir "${cache_dir}" \
      --template "${template}" \
      --reposition_mode swap_sink_nonsink \
      --num_swap "${num_swap}" \
      --reposition_seed "${seed}" \
      --run_idx "${run_idx}" \
      --max_samples "${max_samples}" \
      --use_existing_sink_records
  done
done
