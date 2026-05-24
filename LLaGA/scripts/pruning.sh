#!/bin/bash

set -euo pipefail

cd /standard/AikyamLab/ding/LLaGA
mkdir -p logs logs_phc3mn

module load miniforge
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-tea}"

model_path="${MODEL_PATH:-Runjin/llaga-vicuna-7b-simteg-ND-general_model-2-layer-mlp-projector}"
model_base="${MODEL_BASE:-lmsys/vicuna-7b-v1.5-16k}"
mode="${CONV_MODE:-v1}"
dataset="${DATASET:-cora}"
task="${TASK:-nc}"
emb="${EMB:-simteg}"
use_hop="${USE_HOP:-2}"
sample_size="${SAMPLE_SIZE:-10}"
template="${TEMPLATE:-ND}"
num_remove="${NUM_REMOVE:-1}"
seed="${SEED:-42}"
pruning_mode="${PRUNING_MODE:-all}"
data_dir="${DATA_DIR:-./dataset/${dataset}/}"

# Simulate one Slurm array task locally; each task is one independent run.
array_task_id="${ARRAY_TASK_ID:-0}"
run_idx=$((array_task_id + 1))

output_path="${OUTPUT_PATH:-results_phc3mn/pruning_${dataset}/${dataset}_${task}_${template}_predictions_local.jsonl}"
sink_records_path="${SINK_RECORDS_PATH:-analysis/${dataset}_${template}/sink_records.jsonl}"
cache_dir="${CACHE_DIR:-../../checkpoint}"

echo "=== Pruning Run ==="
echo "Array task id: ${array_task_id}"
echo "Run idx: ${run_idx}"
echo "Seed (fixed): ${seed}"
echo "Dataset: ${dataset}"
echo "Task: ${task}"
echo "Template: ${template}"
echo "Pruning mode: ${pruning_mode}"
echo "Tokens removed per sample: ${num_remove}"
echo "Data dir: ${data_dir}"
echo "Sink records: ${sink_records_path}"
echo "Output path: ${output_path}"
echo "Cache dir: ${cache_dir}"
echo "Conda env: ${CONDA_ENV:-tea}"
echo "==================="

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
  --cache_dir "${cache_dir}"
  --template "${template}"
  --data_dir "${data_dir}"
  --pruning
  --pruning_mode "${pruning_mode}"
)

# Only add these when needed
if [[ "${PRUNING_NONSINK:-0}" == "1" ]]; then
  if [[ ! -f "${sink_records_path}" ]]; then
    echo "Missing sink records: ${sink_records_path}"
    echo "Generate sink_records.jsonl first before using PRUNING_NONSINK=1."
    exit 1
  fi
  cmd+=(--pruning_nonsink --num_prune "${num_remove}" --seed "${seed}" --run_idx "${run_idx}" --sink_records_path "${sink_records_path}")
fi

"${cmd[@]}"