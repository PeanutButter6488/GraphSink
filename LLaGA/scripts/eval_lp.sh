#!/bin/bash
#
# Run LLaGA link-prediction inference + RQ1/RQ2 sink analysis on
# arxiv / cora / pubmed.
#
# What it does:
#   - calls eval/eval_pretrain.py with --task lp --template HO
#     (HO loads edge_sampled_2_10_only_test.jsonl for each dataset)
#   - --attention_probe enables RQ2 query-to-graph attention aggregation
#     (RQ1 sink-dimension activation curves are computed unconditionally;
#      see the activation_topdims_state branch in eval_pretrain.py:1265-1308)
#   - per-dataset analysis artifacts land in:
#       analysis/{dataset}_HO/sink_dim_mean_activation.png            (RQ1, all)
#       analysis/{dataset}_HO/sink_only_dim_mean_activation.png       (RQ1, sink-only)
#       analysis/{dataset}_HO/cross_attention_layer_vs_graph_heatmap.png  (RQ2)
#       analysis/{dataset}_HO/sink_records.jsonl                      (per-sample sinks)
#       analysis/{dataset}_HO/rq_arrays/*.npy                         (numerical dumps)
#
# Run after LLaGA's HO/general checkpoint has been downloaded (default below).
# Override DATASETS / MAX_SAMPLES / SEED via environment if needed:
#   DATASETS="arxiv cora pubmed" MAX_SAMPLES=500 ./scripts/eval_lp.sh

set -euo pipefail

model_path="${MODEL_PATH:-Runjin/llaga-vicuna-7b-simteg-ND-general_model-2-layer-mlp-projector}"
model_base="${MODEL_BASE:-lmsys/vicuna-7b-v1.5-16k}"
mode="${CONV_MODE:-v1}"

# HO template for LP. README says use_hop=4 for HO; sample_size=10 matches the
# bundled edge_sampled_2_10_only_test.jsonl filename pattern (HO branch in
# resolve_eval_paths hard-codes the 2_10 file regardless of use_hop).
task="lp"
template="ND"
emb="${EMB:-simteg}"
use_hop="${USE_HOP:-2}"
sample_size="${SAMPLE_SIZE:-10}"

seed="${SEED:-42}"
temperature="${TEMPERATURE:-0.0}"
max_samples="${MAX_SAMPLES:-500}"

# LP-tuned sink detection. The defaults in eval_pretrain.py (1512,2298,2533 @ 20.0)
# are calibrated for NC and detect 0 sinks on LP — at layer -2 the LP top dims
# across arxiv/cora/pubmed are 2533, 2789, 363 with magnitudes peaking ~5.
# Override per dataset via SINK_DIMS / SINK_THRESHOLD if needed.
sink_dims="${SINK_DIMS:-1415, 1512, 2533}"
sink_threshold="${SINK_THRESHOLD:-20.0}"

datasets=(${DATASETS:-cora pubmed arxiv})

output_dir="${OUTPUT_DIR:-results_phc3mn}"
mkdir -p "${output_dir}"

for dataset in "${datasets[@]}"; do
  output_path="${output_dir}/${dataset}_${task}_${template}_predictions.jsonl"

  echo "=================================================="
  echo "LLaGA LP | dataset=${dataset} | template=${template} | use_hop=${use_hop}"
  echo "Output:   ${output_path}"
  echo "Sinks:    analysis/${dataset}_${template}/"
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
    --cache_dir ./checkpoint \
    --template "${template}" \
    --temperature "${temperature}" \
    --seed "${seed}" \
    --max_samples "${max_samples}" \
    --sink_dims "${sink_dims}" \
    --sink_threshold "${sink_threshold}" \
    --attention_probe

  # Score the predictions (yes/no accuracy from eval_res.eval_lp).
  python eval/eval_res.py \
    --dataset "${dataset}" \
    --task "${task}" \
    --res_path "${output_path}"
done
