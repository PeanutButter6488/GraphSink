#!/bin/bash

set -euo pipefail

model_path="${MODEL_PATH:-Runjin/llaga-vicuna-7b-simteg-ND-general_model-2-layer-mlp-projector}"
model_base="${MODEL_BASE:-lmsys/vicuna-7b-v1.5-16k}"
mode="${CONV_MODE:-v1}"
datasets="${DATASETS:-cora pubmed arxiv}"           # space-separated
reposition_modes="${REPOSITION_MODES:-front_top2 front_all}"   # deterministic modes
task="${TASK:-nc}"
emb="${EMB:-simteg}"
use_hop="${USE_HOP:-2}"
sample_size="${SAMPLE_SIZE:-10}"
template="${TEMPLATE:-ND}"

seed="${SEED:-42}"
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
    echo "Generate sink_records.jsonl first before running reposition. Skipping."
    continue
  fi

  for rmode in ${reposition_modes}; do
    echo "=================================================="
    echo "[${dataset}] reposition_mode=${rmode} (deterministic, single run)"
    echo "seed=${seed} | max_samples=${max_samples}"
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
      --cache_dir "${cache_dir}" \
      --template "${template}" \
      --reposition_mode "${rmode}" \
      --reposition_seed "${seed}" \
      --seed "${seed}" \
      --run_idx 0 \
      --max_samples "${max_samples}" \
      --use_existing_sink_records
  done
done
