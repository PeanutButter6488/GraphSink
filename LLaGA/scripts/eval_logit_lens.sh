#!/bin/bash
# Run LLaGA NC evaluation with --logit_lens to produce a per-layer x graph-token
# top-1 token heatmap (analogous to TEA-GLM/utils/logit_lens.py).
#
# Heatmap is saved to:
#   analysis/<dataset>_<template>/logit_lens/logit_lens.png
#
# Sample filter: only NC samples whose detected sink graph-token indices are a
# non-empty subset of {4,5,6,7,8} are used for aggregation (matches the
# TEA-GLM "sinks at fixed positions" filter — see eval/eval_pretrain.py).

set -euo pipefail

model_path="${MODEL_PATH:-Runjin/llaga-vicuna-7b-simteg-ND-general_model-2-layer-mlp-projector}"
model_base="${MODEL_BASE:-lmsys/vicuna-7b-v1.5-16k}"
mode="${CONV_MODE:-v1}"
task="${TASK:-nc}"
emb="${EMB:-simteg}"
use_hop="${USE_HOP:-2}"
sample_size="${SAMPLE_SIZE:-10}"
template="${TEMPLATE:-ND}"

seed="${SEED:-42}"
temperature="${TEMPERATURE:-0.0}"

datasets=(${DATASETS:-arxiv pubmed cora})
max_samples="${MAX_SAMPLES:-500}"

output_dir="${OUTPUT_DIR:-results_phc3mn}"
mkdir -p "${output_dir}"

for dataset in "${datasets[@]}"; do
  output_path="${output_dir}/${dataset}_${task}_${template}_logit_lens.jsonl"

  echo "=================================================="
  echo "Logit-lens | dataset=${dataset} template=${template} task=${task}"
  echo "Output: ${output_path} | seed=${seed} | temperature=${temperature}"
  echo "=================================================="

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
    --cache_dir ./checkpoint \
    --template "${template}" \
    --temperature "${temperature}" \
    --seed "${seed}" \
    --max_samples "${max_samples}" \
    --logit_lens \
    --use_existing_sink_records
done
