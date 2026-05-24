#!/bin/bash

set -euo pipefail

# Sink re-emergence experiment for LLaGA.
#
# Prunes ALL detected sink graph tokens per sample, then re-runs sink detection
# on the shortened sequence. Outputs (per dataset) under
# ./analysis/{dataset}_{template}/:
#   sink_reoccur.jsonl                    — per-sample post-prune sink records
#   sink_reoccur_distribution.png         — histogram of post-prune sink positions
#   sink_distribution_shift.png           — overlay: baseline vs post-prune
#
# Gate (eval_pretrain.py:432-436):
#   --sink_reoccur AND --pruning AND --pruning_mode all AND NOT --pruning_nonsink
#
# Prerequisite per dataset: baseline ./analysis/{dataset}_{template}/sink_records.jsonl
# (LLaGA has no seed-suffixed baselines; the canonical file is the source).

cd /standard/AikyamLab/ding/glm_sink/LLaGA
mkdir -p logs logs_phc3mn

module load miniforge
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-tea}"

model_path="${MODEL_PATH:-Runjin/llaga-vicuna-7b-simteg-ND-general_model-2-layer-mlp-projector}"
model_base="${MODEL_BASE:-lmsys/vicuna-7b-v1.5-16k}"
mode="${CONV_MODE:-v1}"
task="${TASK:-nc}"
emb="${EMB:-simteg}"
use_hop="${USE_HOP:-2}"
sample_size="${SAMPLE_SIZE:-10}"
template="${TEMPLATE:-ND}"
num_remove="${NUM_REMOVE:-2}"
seed="${SEED:-42}"
cache_dir="${CACHE_DIR:-./checkpoint}"

read -r -a datasets <<< "${DATASETS:-arxiv cora pubmed}"

for dataset in "${datasets[@]}"; do
    output_path="results_phc3mn/reoccur_${dataset}/${dataset}_${task}_${template}_predictions.jsonl"
    sink_records_path="analysis/${dataset}_${template}/sink_records.jsonl"

    if [[ ! -f "${sink_records_path}" ]]; then
        echo "[skip] ${dataset}: missing baseline sink records at ${sink_records_path}"
        continue
    fi

    echo "=== LLaGA sink re-emergence: ${dataset} ==="
    echo "Template: ${template} | Task: ${task} | Hop: ${use_hop} | NeighborSample: ${sample_size}"
    echo "Baseline records: ${sink_records_path}"
    echo "Output: ${output_path}"
    echo "==========================================="

    "${CONDA_PREFIX}/bin/python" eval/eval_pretrain.py \
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
        --pruning \
        --pruning_mode all \
        --num_prune "${num_remove}" \
        --seed "${seed}" \
        --sink_records_path "${sink_records_path}" \
        --sink_reoccur \
        --use_existing_sink_records \
        --max_samples 500
done
