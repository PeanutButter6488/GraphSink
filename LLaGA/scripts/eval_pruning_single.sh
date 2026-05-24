#!/bin/bash

set -euo pipefail

model_path="${MODEL_PATH:-Runjin/llaga-vicuna-7b-simteg-ND-general_model-2-layer-mlp-projector}"
model_base="${MODEL_BASE:-lmsys/vicuna-7b-v1.5-16k}"
mode="${CONV_MODE:-v1}"
dataset="${DATASET:-cora}"
task="${TASK:-nc}"
emb="${EMB:-simteg}"
use_hop="${USE_HOP:-2}"
sample_size="${SAMPLE_SIZE:-10}"
template="${TEMPLATE:-ND}"

dim_zeroout_num="${DIM_ZEROOUT_NUM:-1}"
dim_zeroout_seed="${DIM_ZEROOUT_SEED:-0}"
dim_zeroout_target="${DIM_ZEROOUT_TARGET:-nonsink}"

output_dir="${OUTPUT_DIR:-results_phc3mn}"
base_output_name="${BASE_OUTPUT_NAME:-${dataset}_${task}_${template}_predictions.jsonl}"
output_path="${output_dir}/${base_output_name}"

mkdir -p "${output_dir}"

extra_args=()
if [ "${dim_zeroout_num}" -gt 0 ]; then
  extra_args+=(
    --dim_zeroout_num "${dim_zeroout_num}"
    --dim_zeroout_seed "${dim_zeroout_seed}"
    --dim_zeroout_target "${dim_zeroout_target}"
  )
fi

echo "=================================================="
echo "Running eval_pretrain.py"
echo "dataset: ${dataset}"
echo "task: ${task}"
echo "target: ${dim_zeroout_target}"
echo "num dims: ${dim_zeroout_num}"
echo "seed: ${dim_zeroout_seed}"
echo "base output: ${output_path}"
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
  --cache_dir ../../checkpoint \
  --template "${template}" \
  "${extra_args[@]}"