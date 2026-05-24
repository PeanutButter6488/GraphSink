#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

datasets=('arxiv:700' 'cora:700' 'pubmed:850')


dataset='arxiv'
num_token=5
num_prune=2  # only used for pruning_mode=random; must match test_single.sh
prefix='TEA-GLM_citation_meanpool'
llm='./vicuna-7b-v1.5'

# Pruning experiment (requires the baseline sink-records JSONL at
# ./analysis/{test_dataset}/global_stats/{prefix}_sink_records.jsonl — run
# without pruning_mode first to produce it). Pruning runs READ that file via
# load_sink_records and do NOT regenerate it (train_glm.py gates the save by
# `not args.prune_sink_tokens`), so the baseline records stay clean.
#
#   pruning_mode: 'top2' | 'all' | 'random'
#   n_runs:       number of repeated runs per dataset, all sharing --seed.
#
# Now that InstructGLM.g_step uses do_sample=True/temperature=0.7, multiple
# runs at the same seed exercise the sampler. The --seed value is fixed so
# the prune set itself stays identical across runs (this matters for
# pruning_mode=random, where seed picks which non-sink tokens are pruned).
# Variance across runs comes from generation sampling only.
#
# test_single.sh writes per-run files using `pruning_output_suffix(mode, ..., seed)`
# (e.g. _prune_all_seed1, _prune_top2_seed1, _prune_random2_seed1). After each
# run we rename _seed${seed} -> _run${run} so successive same-seed runs land
# in their own files.
pruning_mode='top2'
seed=42
n_runs=1

case "$pruning_mode" in
    top2)   prune_tag="_prune_top2" ;;
    all)    prune_tag="_prune_all" ;;
    random) prune_tag="_prune_random${num_prune}" ;;
    *) echo "Unknown pruning_mode: $pruning_mode" >&2; exit 1 ;;
esac


for pair in "${datasets[@]}"
do
    IFS=':' read -r test_dataset max_text_length <<< "$pair"
    for run in $(seq 1 "$n_runs"); do
        echo "Testing (pruning_mode=$pruning_mode, run=$run, seed=$seed) with max_text_length $max_text_length on dataset $test_dataset"
        bash ./scripts/test_single.sh $dataset $test_dataset $num_token $max_text_length $prefix $llm $pruning_mode $seed

        # Rename _seed${seed} -> _run${run} for this run's outputs so
        # successive same-seed runs do not overwrite each other.
        seed_tag="${prune_tag}_seed${seed}"
        run_tag="${prune_tag}_seed${seed}_run${run}"

        for kind in results labels; do
            src="./results/${test_dataset}/${prefix}${seed_tag}_model_${kind}.txt"
            dst="./results/${test_dataset}/${prefix}${run_tag}_model_${kind}.txt"
            if [ -f "$src" ]; then
                mv "$src" "$dst"
                echo "  renamed $(basename "$src") -> $(basename "$dst")"
            else
                echo "  WARNING: expected $src not found; skipping rename for run=$run"
            fi
        done

        # If --sink_reoccur is ever added to test_single.sh, also rename the
        # extra artifacts written under ./analysis/.../global_stats/.
        analysis_dir="./analysis/${test_dataset}/global_stats"
        for ext in "_sink_reoccur_records.jsonl" "_sink_reoccur_distribution.png"; do
            src="${analysis_dir}/${prefix}${seed_tag}${ext}"
            dst="${analysis_dir}/${prefix}${run_tag}${ext}"
            if [ -f "$src" ]; then
                mv "$src" "$dst"
                echo "  renamed $(basename "$src") -> $(basename "$dst")"
            fi
        done
    done
done
