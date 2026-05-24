#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

datasets=('arxiv:700' 'pubmed:850' 'cora:700')
#datasets=('arxiv:700')


dataset='arxiv'
num_token=5
# prefix='TEA-GLM_arxiv_pretrain-token20-alldata-416'
prefix='TEA-GLM_citation_meanpool'
llm='./vicuna-7b-v1.5'

# Vanilla (no-pruning) run with a specific seed. Leave `seed` empty for a
# classic unseeded baseline (result files written without _seed{N} suffix).
# When `seed` is set, test_single.sh auto-adds --seed $seed --append_seed_suffix
# so the run writes to {prefix}_seed{seed}_model_results.txt and the analysis
# artifacts get the same suffix (won't overwrite the unseeded baseline).
seed=42


for pair in "${datasets[@]}"
do
    IFS=':' read -r test_dataset max_text_length <<< "$pair"

    echo "Testing (vanilla, seed=${seed:-none}) with max_text_length $max_text_length on dataset $test_dataset"
    LOGIT_LENS=1 bash ./scripts/test_single.sh $dataset $test_dataset $num_token $max_text_length $prefix $llm "" $seed
done
