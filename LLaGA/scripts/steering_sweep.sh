#!/bin/bash
#
# Sweep contrastive steering on LLaGA: {project, subtract} x {0.1..1.0} on layers 23-30.
# 2 modes x 10 strengths = 20 runs. Each run lands in its own JSONL because
# eval_pretrain.py auto-suffixes the answers_file with the steering config.
#
# Run from the LLaGA/ directory:
#   bash scripts/steering_sweep.sh
#
# Override anything via env vars, e.g.:
#   DATASET=pubmed LAYERS=28-31 MAX_SAMPLES=200 bash scripts/steering_sweep.sh

set -euo pipefail

model_path="${MODEL_PATH:-Runjin/llaga-vicuna-7b-simteg-ND-general_model-2-layer-mlp-projector}"
model_base="${MODEL_BASE:-lmsys/vicuna-7b-v1.5-16k}"
conv_mode="${CONV_MODE:-v1}"
dataset="${DATASET:-cora}"
emb="${EMB:-simteg}"
use_hop="${USE_HOP:-2}"
sample_size="${SAMPLE_SIZE:-10}"
template="${TEMPLATE:-ND}"
task="${TASK:-nc}"

layers="${LAYERS:-23-30}"
target="${TARGET:-query}"
source_mode="${SOURCE:-per_sample}"  # 'global' or 'per_sample'
max_samples="${MAX_SAMPLES:-500}"

output_dir="${OUTPUT_DIR:-results_phc3mn}"
mkdir -p "${output_dir}"
base_answers_file="${output_dir}/${dataset}_${task}_${template}_steering.jsonl"

# Match Python's f"{x:g}" — strips trailing zeros so 1.0 -> "1".
strengths=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
modes=(project)

# eval_pretrain.py builds the suffix as:
#   _steer_<mode>_<source_tag>_L<sanitized_layers>_s<strength>_t<target>
# source_tag is 'ps' (per_sample) or 'gl' (global).
source_tag="ps"; [[ "${source_mode}" == "global" ]] && source_tag="gl"
predicted_path() {
  local mode="$1" strength="$2"
  local root="${base_answers_file%.jsonl}"
  echo "${root}_steer_${mode}_${source_tag}_L${layers}_s${strength}_t${target}.jsonl"
}

total=$(( ${#modes[@]} * ${#strengths[@]} ))
i=0
for mode in "${modes[@]}"; do
  for strength in "${strengths[@]}"; do
    i=$((i + 1))
    out=$(predicted_path "${mode}" "${strength}")

    echo "=================================================="
    echo "[${i}/${total}] mode=${mode} strength=${strength} layers=${layers}"
    echo "          -> ${out}"
    echo "=================================================="

    if [[ -s "${out}" ]]; then
      echo "Output already exists, skipping."
      continue
    fi

    python eval/eval_pretrain.py \
      --model_path "${model_path}" \
      --model_base "${model_base}" \
      --conv_mode "${conv_mode}" \
      --dataset "${dataset}" \
      --pretrained_embedding_type "${emb}" \
      --use_hop "${use_hop}" \
      --sample_neighbor_size "${sample_size}" \
      --answers_file "${base_answers_file}" \
      --task "${task}" \
      --cache_dir ./checkpoint \
      --template "${template}" \
      --steering_mode "${mode}" \
      --steering_strength "${strength}" \
      --steering_layers "${layers}" \
      --steering_target "${target}" \
      --steering_source "${source_mode}" \
      --max_samples "${max_samples}" \
      --use_existing_sink_records
  done
done

echo "Sweep complete. Outputs in ${output_dir}/"
