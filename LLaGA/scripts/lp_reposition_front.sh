#!/bin/bash
#
# LLaGA LP — deterministic front-reposition modes on arxiv / cora / pubmed.
# Mirrors scripts/reposition.sh but with --task lp.
#
# Modes (deterministic, single run each):
#   front_top2 — move the top-2 sinks to the front of the graph block
#   front_all  — move ALL detected sinks to the front of the graph block
#
# Prerequisite: baseline LP run that wrote
#   analysis/{dataset}_{template}_LP/sink_records.jsonl

set -euo pipefail

model_path="${MODEL_PATH:-Runjin/llaga-vicuna-7b-simteg-ND-general_model-2-layer-mlp-projector}"
model_base="${MODEL_BASE:-lmsys/vicuna-7b-v1.5-16k}"
mode="${CONV_MODE:-v1}"
datasets="${DATASETS:-cora pubmed arxiv}"
reposition_modes="${REPOSITION_MODES:-front_top2 front_all}"
task="lp"
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
  sink_records_path="analysis/${dataset}_${template}_LP/sink_records.jsonl"

  if [[ ! -f "${sink_records_path}" ]]; then
    echo "[${dataset}] Missing LP sink records: ${sink_records_path}"
    echo "Run scripts/eval_lp.sh first to produce them. Skipping."
    continue
  fi

  for rmode in ${reposition_modes}; do
    echo "=================================================="
    echo "[LP][${dataset}] reposition_mode=${rmode} (deterministic, single run)"
    echo "seed=${seed}  max_samples=${max_samples}"
    echo "sink_records: ${sink_records_path}"
    echo "base output:  ${output_path}"
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
