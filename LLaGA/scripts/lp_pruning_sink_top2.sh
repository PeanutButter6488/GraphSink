#!/bin/bash
#
# LLaGA LP — sink-token pruning (top-2 mode) on arxiv / cora / pubmed.
# Mirrors scripts/pruning_sink_multi.sh but with --task lp.
#
# Prerequisite: baseline LP run that wrote sink_records.jsonl to
#   analysis/{dataset}_{template}_LP/sink_records.jsonl
# (i.e. you've already run scripts/eval_lp.sh)
#
# Per-run output:
#   results_phc3mn/pruning_lp_{dataset}/{dataset}_lp_{template}_predictions_prune_sinktoken_top2_run{N}.jsonl

set -euo pipefail

model_path="${MODEL_PATH:-Runjin/llaga-vicuna-7b-simteg-ND-general_model-2-layer-mlp-projector}"
model_base="${MODEL_BASE:-lmsys/vicuna-7b-v1.5-16k}"
mode="${CONV_MODE:-v1}"
datasets="${DATASETS:-cora pubmed arxiv}"
task="lp"
emb="${EMB:-simteg}"
use_hop="${USE_HOP:-2}"
sample_size="${SAMPLE_SIZE:-10}"
template="${TEMPLATE:-ND}"

n_runs="${N_RUNS:-1}"
run_start="${RUN_START:-0}"
seed="${SEED:-42}"
pruning_mode="${PRUNING_MODE:-top2}"   # top2 | all
max_samples="${MAX_SAMPLES:-500}"
cache_dir="${CACHE_DIR:-./checkpoint}"

# LP-tuned sink detection (used when sink records are regenerated; ignored when
# --use_existing_sink_records reads the saved jsonl).
sink_dims="${SINK_DIMS:-2533,2789,363}"
sink_threshold="${SINK_THRESHOLD:-20.0}"

for dataset in ${datasets}; do
  data_dir="./dataset/${dataset}/"
  output_dir="results_phc3mn/pruning_lp_${dataset}"
  base_output_name="${dataset}_${task}_${template}_predictions.jsonl"
  output_path="${output_dir}/${base_output_name}"
  sink_records_path="analysis/${dataset}_${template}_LP/sink_records.jsonl"

  mkdir -p "${output_dir}"

  if [[ ! -f "${sink_records_path}" ]]; then
    echo "[${dataset}] Missing LP sink records: ${sink_records_path}"
    echo "Run scripts/eval_lp.sh first to produce them. Skipping."
    continue
  fi

  echo "=================================================="
  echo "[LP][${dataset}] sink pruning (mode=${pruning_mode}) over ${n_runs} run(s)"
  echo "template=${template}  use_hop=${use_hop}  seed=${seed}  run_start=${run_start}"
  echo "sink_records: ${sink_records_path}"
  echo "base output:  ${output_path}"
  echo "=================================================="

  for ((i=0; i<n_runs; i++)); do
    run_idx=$((run_start + i))
    echo ""
    echo ">>> [LP][${dataset}] Run $((i+1))/${n_runs} (run_idx=${run_idx}, seed=${seed})"
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
      --sink_dims "${sink_dims}" \
      --sink_threshold "${sink_threshold}" \
      --run_idx "${run_idx}" \
      --max_samples "${max_samples}" \
      --use_existing_sink_records
  done
done
