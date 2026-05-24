#!/bin/bash

set -euo pipefail

model_path="${MODEL_PATH:-Runjin/llaga-vicuna-7b-simteg-ND-general_model-2-layer-mlp-projector}"
model_base="${MODEL_BASE:-lmsys/vicuna-7b-v1.5-16k}"
mode="${CONV_MODE:-v1}"
datasets="${DATASETS:-cora}"   # space-separated
task="${TASK:-nc}"
emb="${EMB:-simteg}"
use_hop="${USE_HOP:-2}"
sample_size="${SAMPLE_SIZE:-10}"
template="${TEMPLATE:-ND}"

n_runs="${N_RUNS:-1}"
run_start="${RUN_START:-0}"
seed="${SEED:-42}"
pruning_mode="${PRUNING_MODE:-top2}"   # top2 | all
cache_dir="${CACHE_DIR:-./checkpoint}"

for dataset in ${datasets}; do
  data_dir="./dataset/${dataset}/"
  output_dir="results_phc3mn/pruning_${dataset}"
  base_output_name="${dataset}_${task}_${template}_predictions.jsonl"
  output_path="${output_dir}/${base_output_name}"
  sink_records_path="analysis/${dataset}_${template}/sink_records.jsonl"

  mkdir -p "${output_dir}"

  if [[ ! -f "${sink_records_path}" ]]; then
    echo "[${dataset}] Missing sink records: ${sink_records_path}"
    echo "Generate sink_records.jsonl first before running sink pruning. Skipping."
    continue
  fi

  echo "=================================================="
  echo "[${dataset}] sink pruning (mode=${pruning_mode}) over ${n_runs} run(s)"
  echo "seed (fixed): ${seed} | run_start: ${run_start}"
  echo "base output: ${output_path}"
  echo "=================================================="

  for ((i=0; i<n_runs; i++)); do
    run_idx=$((run_start + i))
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
      --data_dir "${data_dir}" \
      --pruning \
      --pruning_mode "${pruning_mode}" \
      --seed "${seed}" \
      --sink_records_path "${sink_records_path}" \
      --run_idx "${run_idx}" \
      --max_samples 500 \
      --use_existing_sink_records
  done
done
