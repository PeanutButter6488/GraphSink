#!/bin/bash

set -euo pipefail

model_path="${MODEL_PATH:-Runjin/llaga-vicuna-7b-simteg-ND-general_model-2-layer-mlp-projector}"
model_base="${MODEL_BASE:-lmsys/vicuna-7b-v1.5-16k}"
mode="${CONV_MODE:-v1}"
dataset="${DATASET:-pubmed}"
task="${TASK:-nc}"
emb="${EMB:-simteg}"
use_hop="${USE_HOP:-2}"
sample_size="${SAMPLE_SIZE:-10}"
template="${TEMPLATE:-ND}"

n_runs="${N_RUNS:-5}"
run_start="${RUN_START:-1}"
seed="${SEED:-42}"

output_dir="${OUTPUT_DIR:-results_phc3mn}"
base_output_name="${BASE_OUTPUT_NAME:-${dataset}_${task}_${template}_predictions.jsonl}"
base_root="${base_output_name%.jsonl}"

mkdir -p "${output_dir}"

echo "=================================================="
echo "Running baseline eval over ${n_runs} runs"
echo "dataset: ${dataset} | seed (fixed): ${seed} | run_start: ${run_start}"
echo "output dir: ${output_dir}"
echo "=================================================="

for ((i=0; i<n_runs; i++)); do
  run_idx=$((run_start + i))
  output_path="${output_dir}/${base_root}_seed${seed}_run${run_idx}.jsonl"
  echo ""
  echo ">>> Run $((i+1))/${n_runs} (seed=${seed}, run_idx=${run_idx})"
  echo ">>> Output: ${output_path}"
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
    --seed "${seed}" \
    --run_idx "${run_idx}" \
    --max_samples 500 \
    --use_existing_sink_records
done
