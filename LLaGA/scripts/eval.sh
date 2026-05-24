#!/bin/bash

set -euo pipefail

model_path="${MODEL_PATH:-Runjin/llaga-vicuna-7b-simteg-ND-general_model-2-layer-mlp-projector}"
model_base="${MODEL_BASE:-lmsys/vicuna-7b-v1.5-16k}"
mode="${CONV_MODE:-v1}"
task="${TASK:-nc}"
emb="${EMB:-simteg}"
use_hop="${USE_HOP:-2}"
sample_size="${SAMPLE_SIZE:-10}"
template="${TEMPLATE:-ND}"

# Deterministic single run for baselines.
seed="${SEED:-42}"
temperature="${TEMPERATURE:-0.0}"

# cora runs end-to-end; pubmed and arxiv get capped.
datasets=("cora")
max_samples="${MAX_SAMPLES:-500}"

output_dir="${OUTPUT_DIR:-results_phc3mn}"
mkdir -p "${output_dir}"

for dataset in "${datasets[@]}"; do
  base_output_name="${dataset}_${task}_${template}_predictions_finalresults.jsonl"
  output_path="${output_dir}/${base_output_name}"

  echo "=================================================="
  echo "Baseline | dataset=${dataset}"
  echo "Output: ${output_path} | seed=${seed} | temperature=${temperature}"
  echo "=================================================="

  cmd=(
    python eval/eval_pretrain.py
    --model_path "${model_path}"
    --model_base "${model_base}"
    --conv_mode "${mode}"
    --dataset "${dataset}"
    --pretrained_embedding_type "${emb}"
    --use_hop "${use_hop}"
    --sample_neighbor_size "${sample_size}"
    --answers_file "${output_path}"
    --task "${task}"
    --cache_dir ./checkpoint
    --template "${template}"
    --temperature "${temperature}"
    --seed "${seed}"
    --max_samples "${max_samples}"
  )


  "${cmd[@]}"
done
