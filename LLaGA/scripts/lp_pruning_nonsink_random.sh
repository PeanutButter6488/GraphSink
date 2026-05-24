#!/bin/bash
#
# LLaGA LP — random non-sink pruning (k=2 by default) on arxiv / cora / pubmed.
# Mirrors scripts/pruning_multi.sh but with --task lp.
#
# Per sample, prune NUM_REMOVE random non-sink graph tokens. Uses a different
# per-run seed (seed_base + run_idx) so each run picks a different random
# non-sink subset. The set of "sinks to exclude" is chosen by --pruning_mode.
#
# Prerequisite: baseline LP run that wrote
#   analysis/{dataset}_{template}_LP/sink_records.jsonl
#
# Per-run output:
#   results_phc3mn/pruning_lp_{dataset}/{dataset}_lp_{template}_predictions_prune_nonsinktoken_{NUM_REMOVE}_run{N}.jsonl

set -euo pipefail

model_path="${MODEL_PATH:-Runjin/llaga-vicuna-7b-simteg-ND-general_model-2-layer-mlp-projector}"
model_base="${MODEL_BASE:-lmsys/vicuna-7b-v1.5-16k}"
mode="${CONV_MODE:-v1}"
datasets="${DATASETS:-arxiv}"
task="lp"
emb="${EMB:-simteg}"
use_hop="${USE_HOP:-2}"
sample_size="${SAMPLE_SIZE:-10}"
template="${TEMPLATE:-ND}"

num_remove="${NUM_REMOVE:-2}"
n_runs="${N_RUNS:-5}"
run_start="${RUN_START:-0}"
seed_base="${SEED_BASE:-42}"
pruning_mode="${PRUNING_MODE:-top2}"   # which sinks to exclude from the non-sink pool
max_samples="${MAX_SAMPLES:-500}"
cache_dir="${CACHE_DIR:-./checkpoint}"

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
  echo "[LP][${dataset}] random non-sink pruning over ${n_runs} runs"
  echo "num_remove=${num_remove}  seed_base=${seed_base}  run_start=${run_start}"
  echo "(per-run seed = seed_base + run_idx → different non-sink subset each run)"
  echo "sink_records: ${sink_records_path}"
  echo "base output:  ${output_path}"
  echo "=================================================="

  for ((i=0; i<n_runs; i++)); do
    run_idx=$((run_start + i))
    seed=$((seed_base + run_idx))
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
      --pruning_nonsink \
      --pruning_mode "${pruning_mode}" \
      --num_prune "${num_remove}" \
      --seed "${seed}" \
      --sink_records_path "${sink_records_path}" \
      --sink_dims "${sink_dims}" \
      --sink_threshold "${sink_threshold}" \
      --run_idx "${run_idx}" \
      --max_samples "${max_samples}" \
      --use_existing_sink_records
  done
done
