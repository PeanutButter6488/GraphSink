#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

# Vanilla (no pruning) multi-run sweep — run the baseline inference `n_runs`
# times with the SAME --seed under sampling decoding (do_sample=True,
# temperature=0.7 in InstructGLM.g_step). Variance across runs comes from the
# stochastic sampler / non-deterministic CUDA kernel selection rather than
# from changing the seed.
#
# test_single.sh writes per-run files as ${prefix}_seed${seed}_model_*.txt
# (using --append_seed_suffix). We rename those to
# ${prefix}_seed${seed}_run${N}_model_*.txt after each run so multiple
# same-seed runs do not overwrite each other AND so different invocations of
# this script with different `seed` values land in distinct files.
#
# Final files per run:
#   ./results/{test_dataset}/{prefix}_seed{S}_run{N}_model_results.txt
#   ./results/{test_dataset}/{prefix}_seed{S}_run{N}_model_labels.txt

#datasets=('arxiv:700' 'pubmed:700' 'cora:850')
datasets=('arxiv:700' 'cora:700' 'pubmed:850')

dataset='arxiv'
num_token=5
# prefix='TEA-GLM_arxiv_pretrain-token20-alldata-416'
prefix='TEA-GLM_citation_meanpool'
llm='./vicuna-7b-v1.5'

seed=123
n_runs=5


for pair in "${datasets[@]}"
do
    IFS=':' read -r test_dataset max_text_length <<< "$pair"
    for run in $(seq 1 "$n_runs"); do
        echo "Vanilla baseline: run=$run, seed=$seed, max_text_length=$max_text_length, dataset=$test_dataset"
        # 7th positional arg (pruning_mode) left empty -> baseline;
        # 8th positional arg carries the seed -> test_single.sh adds
        #   --seed $seed --append_seed_suffix, producing _seed${seed} files.
        bash ./scripts/test_single.sh $dataset $test_dataset $num_token $max_text_length $prefix $llm "" $seed

        # Rename the _seed{seed} outputs to _seed{seed}_run{run} so
        # successive runs (and runs at different seeds) each land in their
        # own file.
        for kind in results labels; do
            src="./results/${test_dataset}/${prefix}_seed${seed}_model_${kind}.txt"
            dst="./results/${test_dataset}/${prefix}_seed${seed}_run${run}_model_${kind}.txt"
            if [ -f "$src" ]; then
                mv "$src" "$dst"
                echo "  renamed $(basename "$src") -> $(basename "$dst")"
            else
                echo "  WARNING: expected $src not found; skipping rename for run=$run"
            fi
        done
    done
done
