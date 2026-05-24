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

direction="${DIRECTION:-sinks_to_nonsink_even}"
fraction="${FRACTION:-0.5}"
source_idx="${SOURCE_IDX:-11}"
layers="${LAYERS:-all}"

# Deterministic single run: temperature=0 (greedy) + fixed seed.
seed="${SEED:-42}"
temperature="${TEMPERATURE:-0.0}"

# Datasets to evaluate. cora is small enough to run end-to-end; the others get capped.
datasets=("cora" "pubmed" "arxiv")
max_samples="${MAX_SAMPLES:-500}"

output_dir="${OUTPUT_DIR:-results_phc3mn}"
mkdir -p "${output_dir}"

for dataset in "${datasets[@]}"; do
  base_output_name="${dataset}_${task}_${template}_predictions.jsonl"
  output_path="${output_dir}/${base_output_name}"

  echo "=================================================="
  echo "Redistribute | dataset=${dataset} | direction=${direction} | fraction=${fraction} | layers=${layers}"
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
    --use_existing_sink_records
    --redistribute
    --redistribute_direction "${direction}"
    --redistribute_fraction "${fraction}"
    --redistribute_source_idx "${source_idx}"
    --redistribute_layers "${layers}"
    --run_idx 1
  )

  if [[ "${dataset}" != "cora" ]]; then
    cmd+=(--max_samples "${max_samples}")
  fi

  "${cmd[@]}"
done
