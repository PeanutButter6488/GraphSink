#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

# Positional args:
#   $1 dataset                 (train dataset name)
#   $2 test_dataset
#   $3 num_token
#   $4 max_text_length
#   $5 prefix
#   $6 backbone (llm path)
#   $7 pruning_mode  (optional: top2 | all | random; empty = baseline, no pruning)
#   $8 seed          (optional)
#       - pruning_mode=random   : seed picks the random non-sink tokens to prune
#       - pruning_mode=""       : seed drives a vanilla multi-seed baseline sweep;
#                                 --append_seed_suffix is added automatically so
#                                 runs do not overwrite each other's result files

pruning_mode="${7:-}"
seed="${8:-}"

extra_flags=()
if [ -n "$pruning_mode" ]; then
    pruning_seed="${seed:-42}"
    extra_flags+=(--prune_sink_tokens --pruning_mode "$pruning_mode" --seed "$pruning_seed")
    if [ "$pruning_mode" = "random" ]; then
        extra_flags+=(--num_prune 2)
    fi
elif [ -n "$seed" ]; then
    # Vanilla baseline with an explicit seed → separate output per seed.
    extra_flags+=(--seed "$seed" --append_seed_suffix)
fi

# Opt-in logit-lens diagnostic via env var (set LOGIT_LENS=1 from the caller).
if [ "${LOGIT_LENS:-0}" = "1" ]; then
    extra_flags+=(--logit_lens)
fi

accelerate launch \
    --config_file accelerate_config/config_single_gpu.yaml \
    train_glm.py \
        --freeze_llama \
        --inference \
        --best_epoch 49 \
        --dataset $1 \
        --test_dataset $2 \
        --att_d_model 2048 \
        --gnn_output 4096 \
        --grad_steps 1 \
        --batch_size 4 \
        --num_token $3 \
        --clip_grad_norm 1.0 \
        --backbone $6 \
        --epoch 1 \
        --weight_decay 0.1 \
        --max_text_length $4 \
        --prefix $5 \
        "${extra_flags[@]}"
