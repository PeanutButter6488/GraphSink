#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

datasets=('arxiv:700' 'cora:700' 'pubmed:850')
#datasets=('arxiv:700')


dataset='arxiv'
num_token=5
num_prune=2  # random pruning: number of non-sink tokens removed per sample
prefix='TEA-GLM_citation_meanpool'
llm='./vicuna-7b-v1.5'

# Random-pruning sweep — DEDICATED to pruning_mode='random'.
#
# Unlike top2 / all (where the prune set is determined by the saved sink
# records and seed only affects sampling), random pruning uses --seed to
# pick WHICH non-sink graph tokens are pruned (compute_prune_positions_batch
# in utils/sink_pruning.py uses `random.Random(seed ^ ...)`). So varying the
# seed here is meaningful: each seed exercises a different prune subset.
#
# Generation also runs with do_sample=True/temperature=0.7 in
# InstructGLM.g_step, so the per-seed outputs reflect both prune-subset
# variance and sampling variance.
#
# Output files are written with the natural _prune_random{num_prune}_seed{N}
# suffix from pruning_output_suffix() — no rename needed.
#
# Per-seed outputs:
#   ./results/{test_dataset}/{prefix}_prune_random{num_prune}_seed{N}_model_results.txt
#   ./results/{test_dataset}/{prefix}_prune_random{num_prune}_seed{N}_model_labels.txt
#
# Seed range:
#   start_seed = first seed in the sweep (e.g. 1 to start at seed=1, or 123 to
#                start at seed=123). Pick a non-overlapping start_seed so a
#                follow-up sweep does not collide with a previous one.
#   n_seeds    = number of consecutive seeds to run (start_seed .. start_seed+n_seeds-1).
pruning_mode='random'
start_seed=128
n_seeds=5
end_seed=$((start_seed + n_seeds - 1))


for pair in "${datasets[@]}"
do
    IFS=':' read -r test_dataset max_text_length <<< "$pair"
    for seed in $(seq "$start_seed" "$end_seed"); do
        echo "Testing (pruning_mode=$pruning_mode, seed=$seed) with max_text_length $max_text_length on dataset $test_dataset"
        bash ./scripts/test_single.sh $dataset $test_dataset $num_token $max_text_length $prefix $llm $pruning_mode $seed
    done
done
