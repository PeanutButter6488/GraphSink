import json
import time
import transformers
import torch
import os
import gc
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta
from typing import Any, Dict
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from model import *
from utils import *

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

from transformers import LlamaTokenizer
import numpy as np, os

from plotting import plot_logit_lens_heatmap


def main(args, SEED):
    run_time = time.strftime("%Y%m%d%H%M", time.localtime())
    wandb_name = f"{args.prefix}_EXP{SEED}_{run_time}"
    if args.inference:
        group = f"{args.dataset}_{args.test_dataset}"
    else:
        group = f"{args.dataset}"
    accelerator.init_trackers(project_name=f"{args.project}",
                              init_kwargs={"wandb":
                                               {"tags": [args.dataset, args.backbone],
                                                "group": group,
                                                "name": wandb_name,
                                                "config": args}
                                           },
                              )

    seed_everything(seed=SEED)
    accelerator.print(args)

    with accelerator.main_process_first():

        tokenizer = LlamaTokenizer.from_pretrained(args.backbone)
        tokenizer.pad_token=tokenizer.unk_token
        special={'additional_special_tokens': ['<Node {}>'.format(i) for i in range(1, 110)]}   # Add a new special token as place holder
        tokenizer.add_special_tokens(special)

    accelerator.wait_for_everyone()

    cur_device = torch.cuda.current_device()
    trainer = TrainerBase(args, cur_device)

    accelerator.print('Building DataLoader')
    if not args.inference:
        train_loader = trainer.create_dataloader(tokenizer)
    else:
        test_loader = trainer.create_dataloader(tokenizer)
        train_loader = test_loader

    accelerator.print('Building Model')
    first_model, model = trainer.create_model()
    torch.set_default_tensor_type(torch.FloatTensor)

    accelerator.print('Building Optimizer')
    optimizer, warmup_scheduler = trainer.create_optimizer_and_scheduler(first_model, model, train_loader)

    trainable_params, all_param = print_trainable_params(first_model, model)
    accelerator.print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    if not os.path.exists('./saved_model/first_model'):
        os.mkdir('./saved_model/first_model')

    first_model_path = './saved_model/first_model/{}_fm_{}_epoch{}_{}.pth'
    model_path = './saved_model/model/{}_m_{}_epoch{}_{}.pth'


    if not args.inference:
        first_model, model, train_loader, optimizer, warmup_scheduler = accelerator.prepare(first_model, model, train_loader,
                                                                                        optimizer, warmup_scheduler)
        accelerator.print('Training')
        num_training_steps = args.epoch * len(train_loader)
        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(args.epoch):

            model.train()
            first_model.train()
            epoch_loss, accum_loss = 0., 0.

            for step, batch in enumerate(train_loader):
                with accelerator.accumulate(model):
                    optimizer.zero_grad()

                    input_ids = batch['input_ids']
                    is_node = batch['is_node']
                    labels = batch["target_ids"]
                    attention_mask = batch['attn_mask']
                    graph = batch['graph']
                    
                    embeds = first_model(
                        input_ids=input_ids,
                        is_node=is_node,
                        graph=graph
                    )
                    output=model(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)

                    loss = output['loss']
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
                    accelerator.clip_grad_norm_(optimizer.param_groups[1]['params'], 0.1)
                    optimizer.step()
                    warmup_scheduler.step()
                    epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()


                if (step + 1) % args.grad_steps == 0:
                    graph_lr = optimizer.param_groups[0]["lr"]
                    lora_lr = optimizer.param_groups[1]["lr"]

                    accelerator.print({'Graph Lr': graph_lr, 'Lora Lr': lora_lr})
                    accelerator.print({'Accum Loss': accum_loss / args.grad_steps})
                    accelerator.log({'Graph Lr': graph_lr, 'Lora Lr': lora_lr})
                    accelerator.log({'Accum Loss': accum_loss / args.grad_steps})
                    accum_loss = 0.

                progress_bar.update(1)

            accelerator.print(f"Epoch: {epoch}|{args.epoch}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
            accelerator.log({
                'Epoch': epoch,
                'Loss': epoch_loss / len(train_loader)
                })

            # train only one epoch and save model
            if epoch == args.epoch - 1:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    accelerator.save(accelerator.unwrap_model(first_model).state_dict(), first_model_path.format(args.prefix, args.dataset, epoch, 'end'))
                    if not args.freeze_llama:
                        accelerator.save(accelerator.unwrap_model(model).state_dict(), model_path.format(args.prefix, args.dataset, epoch, 'end'))
            best_epoch = args.best_epoch
     
        accelerator.wait_for_everyone()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        accelerator.wait_for_everyone()
    else:
        first_model, model, test_loader = accelerator.prepare(first_model, model, test_loader)
        best_epoch = args.best_epoch

        # Step 5. Evaluating
        accelerator.print('Evaluating')
        with accelerator.main_process_first():
            first_model = accelerator.unwrap_model(first_model)
            first_model.load_state_dict(torch.load(first_model_path.format(args.prefix, args.dataset, best_epoch, 'end')), strict=False)
            # first_model.GT.load_state_dict(torch.load(first_model_path.format(args.prefix, args.dataset, best_epoch, 'end')))
            model = model.cuda() # transformers bug
            model = accelerator.unwrap_model(model)
            if not args.freeze_llama:
                model.load_state_dict(torch.load(model_path.format(args.prefix, args.dataset, best_epoch, 'end')))

        first_model.eval()
        model.eval()
        samples_seen = 0
        eval_output = []
        eval_label = []

        num_layers = model.config.num_hidden_layers + 1  # should be 32 for Llama-7B

        out_ds_tag = args.test_dataset if getattr(args, "task", "nc") == "nc" else f"{args.test_dataset}_{args.task}"
        global_plot_save_dir = f'./analysis/{out_ds_tag}/global_stats/'
        attention_sink_solution = f'./analysis/{out_ds_tag}/att_redis'
        activation_summary_dir = f'./analysis/{out_ds_tag}/activation_topdims'
        activation_aggregate_dir = './analysis/activation_topdims'
        activation_topk = 5

        activation_storage = None
        storage = None

        # Identify Graph Sink Tokens
        sink_hist_storage = init_sink_token_histogram_storage(num_graph_tokens=args.num_token)
        sink_records = []
        sink_dims = [1512, 1415, 2533]
        sink_threshold = 35.0

        # Identify Graph Sink Dimensions (task 1):
        # aggregate per-sample per-dim graph-token activation at the second-to-last layer
        # and average over all test samples to find the top-K sink dimensions.
        hidden_dim = model.config.hidden_size
        sink_dim_storage = init_graph_feature_mean_storage(
            hidden_dim=hidden_dim,
            layer_index=-2,
            token_reduce="mean",
            apply_rmsnorm=True,
            use_abs=True,
        )
        sink_dim_topk = 3

        # Layerwise top-dim aggregator (LLaGA-style): per-layer [L, D] sums for
        # |RMSNorm(x)|, signed RMSNorm(x), and raw x. Two views are kept:
        #   - all graph tokens
        #   - sink-only graph tokens (subset by detect_sink_tokens output)
        # Used by plot_topdims_mean_activation_curve to render the layer-specific
        # and layer-averaged curves with the top-k sink dimensions annotated.
        topdims_all_storage = init_layerwise_topdims_storage(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
        )
        topdims_sink_only_storage = init_layerwise_topdims_storage(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
        )

        # Query-to-graph attention (research question 2):
        # mean [Q, K] cross-attention matrix from post-graph query tokens to
        # graph tokens, averaged over layers + heads + samples. Query positions
        # are aligned by relative offset after the graph block.
        #q2g_q_max = args.max_text_length
        q2g_q_max = 20
        q2g_storage = init_query_to_graph_attention_storage(
            num_graph_tokens=args.num_token,
            q_max=q2g_q_max,
            num_layers=num_layers,
        )

        # Sink-token pruning (research question 3): load saved sink records and
        # prune positions per sample before `model.g_step` generation.
        pruning_records_index = None
        if args.prune_sink_tokens:
            sink_records_source_path = args.sink_record_path or os.path.join(
                global_plot_save_dir, f"{args.prefix}_sink_records.jsonl",
            )
            pruning_records_index = load_sink_records(sink_records_source_path)
            accelerator.print(
                f"Pruning enabled: mode={args.pruning_mode}, num_prune={args.num_prune}, "
                f"seed={args.seed}, records={sink_records_source_path} "
                f"({len(pruning_records_index)} samples indexed)"
            )

        # Sink re-emergence (research question 4): after pruning the top-2
        # detected sinks per sample, run an extra forward pass on the shortened
        # sequence and check whether new sinks appear among the remaining
        # graph tokens. Gated on pruning_mode=top2 + --sink_reoccur. Outputs
        # are rank-indexed (rank 3..K by baseline score among survivors), not
        # absolute K-position, to make per-sample comparisons valid.
        sink_reoccur_enabled = (
            args.prune_sink_tokens
            and args.pruning_mode == "top2"
            and args.sink_reoccur
        )
        sink_reoccur_storage = None
        reoccur_records = []
        if sink_reoccur_enabled:
            sink_reoccur_storage = init_reoccur_summary_storage(
                num_graph_tokens=args.num_token,
                num_pruned=2,
            )
            accelerator.print(
                f"Sink re-emergence analysis enabled (mode=top2, K={args.num_token}). "
                f"Per-rank outcomes indexed by baseline score (ranks 3..{args.num_token})."
            )

        # Graph-token shuffling experiment (research question 5):
        # swap num_swap sink positions with num_swap non-sink positions in the
        # graph block, then run g_step on the permuted embeds. Mutually
        # exclusive with pruning.
        reposition_enabled = args.reposition_mode != "none"
        reposition_records_index = None
        reposition_records = []
        if reposition_enabled:
            assert not args.prune_sink_tokens, (
                "--reposition_mode is mutually exclusive with --prune_sink_tokens."
            )
            reposition_source_path = args.sink_record_path or os.path.join(
                global_plot_save_dir, f"{args.prefix}_sink_records.jsonl",
            )
            reposition_records_index = load_sink_records(reposition_source_path)
            accelerator.print(
                f"Reposition enabled: mode={args.reposition_mode}, num_swap={args.num_swap}, "
                f"seed={args.reposition_seed}, records={reposition_source_path} "
                f"({len(reposition_records_index)} samples indexed)"
            )

        # Logit-lens aggregation buffers (main process collects per-sample top-1 ids/probs;
        # aggregated once after the loop into a modal-token + mean-prob heatmap).
        # Filter: only samples where sinks are exactly at graph-token positions {0, 1}.
        logit_lens_records_ids = []
        logit_lens_records_probs = []
        logit_lens_total_seen = 0

        progress_bar_test = tqdm(range(len(test_loader)))
        for step, batch in enumerate(test_loader):
            attrs_per_token = []
            input_ids = batch['input_ids']
            is_node = batch['is_node']
            attention_mask = batch['attn_mask']    # which token receives attention mask (i.e., excluding the padded tokens)
            graph = batch['graph']
            node_sets = batch['node_set']

            
            with torch.no_grad():
                embeds = first_model(     # output of graph encoder + mlp projector ==> shape: [batch, max text length, output dim] ([4, 700, 4096])
                    input_ids=input_ids,
                    is_node=is_node,
                    graph=graph
                )

                # Skip the analysis forward pass when doing a pruning or reposition eval run.
                if not args.prune_sink_tokens and not reposition_enabled:
                    outputs = model(
                        inputs_embeds=embeds,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        output_attentions=True,
                        return_dict=True
                    )

                    sink_analysis = compute_layerwise_graph_token_hidden_states(
                        hidden_states=outputs.hidden_states,
                        is_node=is_node,
                        expected_k=5
                    )

                    for b, is_valid in enumerate(sink_analysis["valid"]):
                        if not is_valid:
                            continue

                        layer_hidden = sink_analysis["layer_graph_hidden_states"][b]        # [L, 5, D]
                        graph_positions = sink_analysis["graph_token_positions"][b]         # [5]

                        sinks = detect_sink_tokens(
                            layer_graph_hidden_states=layer_hidden,
                            graph_token_positions=graph_positions,
                            sink_dims=sink_dims,
                            threshold=sink_threshold,
                        )

                        if args.logit_lens and accelerator.is_main_process:
                            logit_lens_total_seen += 1
                            # Restrict to samples where the only sink graph tokens are
                            # at positions 0 and 1, so aggregated rows have a
                            # consistent meaning: g0/g1 = sinks, g2/g3/g4 = non-sinks.
                            if set(sinks["sink_token_indices"]) == {0, 1}:
                                base_model = accelerator.unwrap_model(model)
                                ll = compute_logit_lens(
                                    layer_graph_hidden_states=layer_hidden[1:],   # drop embedding layer; keep transformer layers 0..31
                                    final_norm=base_model.model.norm,
                                    lm_head=base_model.lm_head,
                                    tokenizer=None,                                # decode after aggregation
                                )
                                logit_lens_records_ids.append(ll["top1_token_ids"])
                                logit_lens_records_probs.append(ll["top1_probs"])

                        sink_hist_storage = update_sink_storage(
                            storage=sink_hist_storage,
                            sink_token_indices=sinks["sink_token_indices"],
                        )

                        dim_summary = summarize_graph_feature_dimensions(
                            layer_graph_hidden_states=layer_hidden,
                            layer_index=-2,
                            token_reduce="mean",
                            layer_reduce="mean",
                            apply_rmsnorm=True,
                            use_abs=True,
                        )
                        sink_dim_storage = update_graph_feature_mean_storage(
                            storage=sink_dim_storage,
                            per_dim_scores=dim_summary["per_dim_scores"],
                        )

                        # All-graph-tokens layerwise topdims update.
                        topdims_all_storage = update_layerwise_topdims_storage(
                            storage=topdims_all_storage,
                            layer_graph_hidden_states=layer_hidden,
                        )
                        # Sink-only layerwise topdims update — skip samples where
                        # no graph token cleared the sink threshold.
                        if len(sinks["sink_token_indices"]) > 0:
                            sink_token_tensor = torch.tensor(
                                sinks["sink_token_indices"], dtype=torch.long,
                            )
                            topdims_sink_only_storage = update_layerwise_topdims_storage(
                                storage=topdims_sink_only_storage,
                                layer_graph_hidden_states=layer_hidden,
                                token_indices=sink_token_tensor,
                            )

                        sink_records.append({
                            "step": int(step),
                            "batch_index": int(b),
                            "sink_dims": sinks["sink_dims"],
                            "threshold": sinks["threshold"],
                            "sink_token_indices": sinks["sink_token_indices"],
                            "graph_token_positions": sinks["graph_token_positions"],
                            "top2_sink_token_indices": sinks["top2_sink_token_indices"],
                            "top2_sink_scores": sinks["top2_sink_scores"],
                            "top2_graph_token_positions": sinks["top2_graph_token_positions"],
                            # Full per-token sink-dim score (length K) needed by
                            # the rank-indexed re-emergence aggregator.
                            "layer_token_scores": sinks["layer_token_scores"].detach().cpu().tolist(),
                        })

                    update_query_to_graph_attention_storage(
                        storage=q2g_storage,
                        outputs=outputs,
                        is_node=is_node,
                        attention_mask=attention_mask,
                        expected_k=args.num_token,
                    )

                prune_positions_per_sample = None
                if args.prune_sink_tokens and pruning_records_index is not None:
                    prune_positions_per_sample = compute_prune_positions_batch(
                        records_index=pruning_records_index,
                        is_node=is_node,
                        step=step,
                        mode=args.pruning_mode,
                        num_prune=args.num_prune,
                        seed=args.seed,
                    )

                reposition_perm_per_sample = None
                if reposition_enabled and reposition_records_index is not None:
                    reposition_perm_per_sample, swap_log = compute_reposition_perm_batch(
                        records_index=reposition_records_index,
                        is_node=is_node,
                        step=step,
                        mode=args.reposition_mode,
                        num_swap=args.num_swap,
                        seed=args.reposition_seed,
                    )
                    for b, log in enumerate(swap_log):
                        reposition_records.append({
                            "step": int(step),
                            "batch_index": int(b),
                            "reposition_applied": log is not None,
                            "sink_positions": log["sink_positions"] if log else [],
                            "nonsink_positions": log["nonsink_positions"] if log else [],
                            "graph_token_indices": log["graph_token_indices"] if log else [],
                        })

                results = model.g_step(
                    in_embeds=embeds,
                    attention_mask=attention_mask,
                    prune_token_positions=prune_positions_per_sample,
                    reposition_perm=reposition_perm_per_sample,
                )

                if sink_reoccur_enabled and prune_positions_per_sample is not None:
                    # Forward the PRUNED sequence through the LLM to get hidden
                    # states at the remaining graph-token positions.
                    pruned_embeds, pruned_mask = model._prune_tokens(
                        embeds, attention_mask, prune_positions_per_sample,
                    )
                    outputs_after = model(
                        inputs_embeds=pruned_embeds,
                        attention_mask=pruned_mask,
                        output_hidden_states=True,
                        output_attentions=False,
                        return_dict=True,
                    )

                    # Build the is_node mask for the pruned buffer by dropping
                    # pruned graph positions from is_node and left-padding (to
                    # stay aligned with tokenizer.padding_side='left' and our
                    # _prune_tokens left-pad convention).
                    B_r, T_orig = is_node.shape
                    T_pruned = pruned_embeds.shape[1]
                    is_node_pruned = torch.zeros(
                        (B_r, T_pruned), dtype=torch.bool, device=is_node.device,
                    )
                    for b in range(B_r):
                        pruned_set_b = {int(p) for p in (prune_positions_per_sample[b] or [])}
                        if pruned_set_b:
                            keep_idx_b = torch.tensor(
                                [t for t in range(T_orig) if t not in pruned_set_b],
                                dtype=torch.long, device=is_node.device,
                            )
                        else:
                            keep_idx_b = torch.arange(T_orig, device=is_node.device)
                        is_node_kept_b = is_node[b].index_select(0, keep_idx_b).to(torch.bool)
                        L_b = is_node_kept_b.numel()
                        is_node_pruned[b, T_pruned - L_b:] = is_node_kept_b

                    # Extract layer-wise hidden states at the remaining graph
                    # positions, using expected_k=None so we keep all K_b tokens
                    # (which differs per sample — more sinks pruned -> smaller).
                    reoccur_analysis = compute_layerwise_graph_token_hidden_states(
                        hidden_states=outputs_after.hidden_states,
                        is_node=is_node_pruned,
                        expected_k=None,
                    )

                    for b in range(B_r):
                        if not reoccur_analysis["valid"][b]:
                            continue
                        layer_hidden_r = reoccur_analysis["layer_graph_hidden_states"][b]  # [L, K_remaining, D]

                        # Map remaining-set local indices back to original K-space.
                        orig_graph_pos = torch.nonzero(
                            is_node[b] > 0, as_tuple=False,
                        ).reshape(-1).tolist()
                        pruned_set_b = {int(p) for p in (prune_positions_per_sample[b] or [])}
                        remaining_original_idx = [
                            i for i, p in enumerate(orig_graph_pos) if int(p) not in pruned_set_b
                        ]
                        if len(remaining_original_idx) != layer_hidden_r.shape[1]:
                            # Shape mismatch — defensive skip.
                            continue

                        reoccur_result = detect_sink_tokens(
                            layer_graph_hidden_states=layer_hidden_r,
                            graph_token_positions=torch.tensor(
                                [orig_graph_pos[i] for i in remaining_original_idx],
                                dtype=torch.long,
                            ),
                            sink_dims=sink_dims,
                            threshold=sink_threshold,
                        )

                        reoccur_local = reoccur_result["sink_token_indices"]
                        reoccur_original = [remaining_original_idx[i] for i in reoccur_local]

                        # Read baseline state for this (step, batch) from the
                        # pruning records index. Baseline must include
                        # `layer_token_scores` (length K) for rank-based stats.
                        rec_b = pruning_records_index.get((int(step), int(b))) if pruning_records_index else None
                        if rec_b is None:
                            continue
                        baseline_sink_indices_k = [int(i) for i in rec_b.get("sink_token_indices", [])]
                        pruned_indices_k = [int(i) for i in rec_b.get("top2_sink_token_indices", [])]
                        scores_list = rec_b.get("layer_token_scores", None)
                        if scores_list is None or len(scores_list) != args.num_token:
                            # Missing or mis-sized scores: skip rank stats but
                            # still log a warning once on rank 0.
                            if accelerator.is_main_process and not getattr(main, "_warned_missing_scores", False):
                                accelerator.print(
                                    "[reoccur] WARNING: baseline sink_records.jsonl is missing "
                                    "`layer_token_scores`. Re-run the baseline (vanilla eval) so "
                                    "the new schema is written, then re-run reoccur."
                                )
                                main._warned_missing_scores = True
                            continue
                        baseline_scores = torch.tensor(scores_list, dtype=torch.float32)

                        sink_reoccur_storage = update_reoccur_summary(
                            storage=sink_reoccur_storage,
                            baseline_layer_token_scores=baseline_scores,
                            baseline_sink_indices=baseline_sink_indices_k,
                            pruned_indices=pruned_indices_k,
                            post_sink_indices=reoccur_original,
                        )

                        # Per-survivor metadata for the JSONL record (paper-friendly).
                        baseline_set = set(baseline_sink_indices_k)
                        post_set = set(reoccur_original)
                        order = torch.argsort(baseline_scores, descending=True).tolist()
                        rank_of = [0] * args.num_token
                        for r_, k_ in enumerate(order, start=1):
                            rank_of[k_] = r_
                        survivor_ks = [k for k in range(args.num_token) if k not in set(pruned_indices_k)]
                        reoccur_records.append({
                            "step": int(step),
                            "batch_index": int(b),
                            "baseline_sink_count": len(baseline_sink_indices_k),
                            "baseline_sink_indices": baseline_sink_indices_k,
                            "baseline_top2_indices": pruned_indices_k,
                            "remaining_indices": survivor_ks,
                            "remaining_baseline_ranks": [rank_of[k] for k in survivor_ks],
                            "remaining_was_sink": [bool(k in baseline_set) for k in survivor_ks],
                            "remaining_is_sink_after": [bool(k in post_set) for k in survivor_ks],
                            "post_sink_count": len(reoccur_original),
                            "post_sink_indices": reoccur_original,
                            "pruned_positions": sorted(pruned_set_b),
                            "sink_dims": reoccur_result["sink_dims"],
                            "threshold": reoccur_result["threshold"],
                        })

                    # Free memory — the pruned forward pass holds all hidden
                    # states and we don't need them once per-sample analysis is done.
                    del outputs_after, pruned_embeds, pruned_mask, is_node_pruned

                results = accelerator.pad_across_processes(results, dim=1, pad_index=tokenizer.pad_token_id)
                results_gathered = accelerator.gather(results).cpu().numpy()

                labels = accelerator.pad_across_processes(
                    batch["target_ids"],
                    dim=1,
                    pad_index=tokenizer.pad_token_id)
                labels_gathered = accelerator.gather(labels).cpu().numpy()

                if accelerator.num_processes > 1:
                    if step == len(test_loader) - 1:
                        results_gathered = results_gathered[
                                                    : len(test_loader.dataset) - samples_seen]
                        labels_gathered = labels_gathered[
                                                    : len(test_loader.dataset) - samples_seen]
                    else:
                        samples_seen += len(results_gathered)
                labels_gathered = np.where(labels_gathered != -100, labels_gathered, tokenizer.pad_token_id)
                accelerator.print(tokenizer.batch_decode(results_gathered, skip_special_tokens=True))
                # accelerator.print(tokenizer.batch_decode(labels_gathered, skip_special_tokens=True))

                eval_output.append(results_gathered)
                eval_label.append(labels_gathered)
            progress_bar_test.update(1)

        if args.logit_lens and accelerator.is_main_process:
            if len(logit_lens_records_ids) > 0:
                agg = aggregate_logit_lens(
                    top1_token_ids_list=logit_lens_records_ids,
                    top1_probs_list=logit_lens_records_probs,
                    tokenizer=tokenizer,
                )
                ll_dir = f'./analysis/{out_ds_tag}/logit_lens'
                plot_logit_lens_heatmap(
                    top1_strings=agg["top1_strings"],
                    top1_probs=agg["top1_probs"],
                    save_path=os.path.join(ll_dir, f'{args.prefix}_logit_lens.png'),
                    sink_indices=[0, 1],
                )
                accelerator.print(
                    f"  Logit-lens heatmap saved (kept {agg['num_samples']}/{logit_lens_total_seen} "
                    f"samples with sinks=={{0,1}}, mean agreement="
                    f"{float(agg['agreement'].mean()):.3f})"
                )
            else:
                accelerator.print(
                    f"  Logit-lens: 0/{logit_lens_total_seen} samples matched the "
                    f"sinks=={{0,1}} filter — heatmap skipped."
                )

        if activation_storage is not None:
            accelerator.wait_for_everyone()
            for layer_idx in activation_storage:
                layer_sum = activation_storage[layer_idx]["sum"].to(accelerator.device)
                layer_count = torch.tensor(
                    activation_storage[layer_idx]["count"],
                    dtype=torch.long,
                    device=accelerator.device,
                )
                activation_storage[layer_idx]["sum"] = accelerator.reduce(layer_sum, reduction="sum").cpu()
                activation_storage[layer_idx]["count"] = int(
                    accelerator.reduce(layer_count, reduction="sum").item()
                )

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                summary_path = os.path.join(
                    activation_summary_dir,
                    f"{args.prefix}_top{activation_topk}_summary.json",
                )
                save_activation_topdims_summary(
                    storage=activation_storage,
                    save_path=summary_path,
                    dataset_name=args.test_dataset,
                    topk_dims=activation_topk,
                    sort_by_abs=True,
                )

                summary_paths = sorted(
                    str(path)
                    for path in Path('./analysis').glob(
                        f"*/activation_topdims/{args.prefix}_top{activation_topk}_summary_epoch99.json"
                    )
                )
                if summary_paths:
                    plot_activation_topdims_count_aggregate(
                        summary_paths=summary_paths,
                        save_path=os.path.join(
                            activation_aggregate_dir,
                            f"{args.prefix}_top{activation_topk}_aggregate_epoch99.png",
                        ),
                    )

                accelerator.print("\nTEA-GLM sink-token summary")
                accelerator.print(f"  Valid samples: {sink_hist_storage['num_valid_samples']}")
                accelerator.print(f"  Samples with >=1 sink token: {sink_hist_storage['num_samples_with_sink']}")
                accelerator.print(f"  Saved sink records: {sink_record_path}")
                accelerator.print(f"  Saved sink histogram: {sink_hist_path}")

        # Step 6. Post-processing & Evaluating
        if args.prune_sink_tokens:
            out_suffix = pruning_output_suffix(args.pruning_mode, args.num_prune, args.seed)
        elif reposition_enabled:
            out_suffix = reposition_output_suffix(args.reposition_mode, args.num_swap, args.reposition_seed)
        elif args.append_seed_suffix:
            out_suffix = f"_seed{args.seed}"
        else:
            out_suffix = ""
        res_path = f'./results/{out_ds_tag}/{args.prefix}{out_suffix}_model_results.txt'
        label_path = f'./results/{out_ds_tag}/{args.prefix}{out_suffix}_model_labels.txt'

        # stats = finalize_head_graph_attn_stats(storage)
        # save_path = os.path.join(attention_sink_solution, "att_heads.pt")
        # torch.save(stats, save_path)

    # Start Saving (skip analysis saves in pruning / reposition mode)

        if not args.prune_sink_tokens and not reposition_enabled:
            # Plot the identified sink tokens
            sink_record_path = os.path.join(
                global_plot_save_dir,
                f"{args.prefix}{out_suffix}_sink_records.jsonl",
            )
            sink_hist_path = os.path.join(
                global_plot_save_dir,
                f"{args.prefix}{out_suffix}_sink_token_histogram.png",
            )

            # Reduce sink-token histogram counts across DDP processes so the plotted
            # distribution reflects the full test set rather than one rank's shard.
            accelerator.wait_for_everyone()
            if accelerator.num_processes > 1:
                counts_tensor = sink_hist_storage["counts"].to(accelerator.device).to(torch.long)
                meta_tensor = torch.tensor(
                    [
                        sink_hist_storage["num_samples"],
                        sink_hist_storage["num_valid_samples"],
                        sink_hist_storage["num_samples_with_sink"],
                    ],
                    dtype=torch.long,
                    device=accelerator.device,
                )
                counts_tensor = accelerator.reduce(counts_tensor, reduction="sum")
                meta_tensor = accelerator.reduce(meta_tensor, reduction="sum")
                sink_hist_storage["counts"] = counts_tensor.detach().cpu().to(torch.long)
                sink_hist_storage["num_samples"] = int(meta_tensor[0].item())
                sink_hist_storage["num_valid_samples"] = int(meta_tensor[1].item())
                sink_hist_storage["num_samples_with_sink"] = int(meta_tensor[2].item())

            if accelerator.is_main_process:
                save_sink_token_records_jsonl(
                    records=sink_records,
                    save_path=sink_record_path,
                )
                sink_distribution(
                    storage=sink_hist_storage,
                    save_path=sink_hist_path,
                )
                print(f"\nTEA-GLM sink-token summary")
                print(f"  Valid samples: {sink_hist_storage['num_valid_samples']}")
                print(f"  Samples with >=1 sink token: {sink_hist_storage['num_samples_with_sink']}")
                print(f"  Saved sink records: {sink_record_path}")
                print(f"  Saved sink histogram: {sink_hist_path}")

            # Sink-dimension identification: reduce across processes, then save + plot on main.
            accelerator.wait_for_everyone()
            if accelerator.num_processes > 1:
                sum_tensor = sink_dim_storage["sum_scores"].to(accelerator.device).to(torch.float32)
                count_tensor = torch.tensor(
                    [sink_dim_storage["num_valid_samples"]],
                    dtype=torch.long,
                    device=accelerator.device,
                )
                sum_tensor = accelerator.reduce(sum_tensor, reduction="sum")
                count_tensor = accelerator.reduce(count_tensor, reduction="sum")
                sink_dim_storage["sum_scores"] = sum_tensor.detach().cpu().to(torch.float64)
                sink_dim_storage["num_valid_samples"] = int(count_tensor.item())

            if accelerator.is_main_process:
                sink_dim_summary_path = os.path.join(
                    global_plot_save_dir,
                    f"{args.prefix}{out_suffix}_sink_dim_summary.json",
                )
                save_graph_feature_mean_summary(
                    storage=sink_dim_storage,
                    save_path=sink_dim_summary_path,
                    dataset_name=args.test_dataset,
                    topk_dims=sink_dim_topk,
                )
                # Sink Dimension Plot
                sink_dim_plot_path = os.path.join(
                    global_plot_save_dir,
                    f"{args.prefix}{out_suffix}_sink_dim_mean_activation.png",
                )
                plot_graph_feature_mean_summary(
                    storage=sink_dim_storage,
                    save_path=sink_dim_plot_path,
                    dataset_name=args.test_dataset,
                    annotate_top_n=sink_dim_topk,
                )
                top_summary = build_graph_feature_mean_summary(
                    sink_dim_storage,
                    dataset_name=args.test_dataset,
                    topk_dims=sink_dim_topk,
                )
                print(f"\nTEA-GLM sink-dimension top-{sink_dim_topk} (layer=-2, token_reduce=mean, rmsnorm=True):")
                for dim_idx, mean_val in zip(top_summary["top_dims"], top_summary["top_means"]):
                    print(f"  dim {dim_idx}: mean activation = {mean_val:.6f}")
                print(f"  n={top_summary['num_valid_samples']} valid samples")
                print(f"  Saved sink-dim summary: {sink_dim_summary_path}")
                print(f"  Saved sink-dim plot:    {sink_dim_plot_path}")

            # Layerwise topdims (all + sink-only): reduce across DDP ranks, then plot on main.
            accelerator.wait_for_everyone()
            for tname, tstorage in (
                ("all", topdims_all_storage),
                ("sink_only", topdims_sink_only_storage),
            ):
                if accelerator.num_processes > 1:
                    sum_all_t = tstorage["sum_all"].to(accelerator.device).to(torch.float32)
                    sum_signed_t = tstorage["sum_all_signed"].to(accelerator.device).to(torch.float32)
                    sum_raw_t = tstorage["sum_all_raw"].to(accelerator.device).to(torch.float32)
                    cnt_t = tstorage["count_all"].to(accelerator.device).to(torch.long)
                    n_t = torch.tensor(
                        [tstorage["num_valid_samples"]],
                        dtype=torch.long,
                        device=accelerator.device,
                    )
                    sum_all_t = accelerator.reduce(sum_all_t, reduction="sum")
                    sum_signed_t = accelerator.reduce(sum_signed_t, reduction="sum")
                    sum_raw_t = accelerator.reduce(sum_raw_t, reduction="sum")
                    cnt_t = accelerator.reduce(cnt_t, reduction="sum")
                    n_t = accelerator.reduce(n_t, reduction="sum")
                    tstorage["sum_all"] = sum_all_t.detach().cpu().to(torch.float64)
                    tstorage["sum_all_signed"] = sum_signed_t.detach().cpu().to(torch.float64)
                    tstorage["sum_all_raw"] = sum_raw_t.detach().cpu().to(torch.float64)
                    tstorage["count_all"] = cnt_t.detach().cpu().to(torch.long)
                    tstorage["num_valid_samples"] = int(n_t.item())

            if accelerator.is_main_process:
                for tname, tstorage in (
                    ("all", topdims_all_storage),
                    ("sink_only", topdims_sink_only_storage),
                ):
                    if tstorage["num_valid_samples"] == 0:
                        print(f"  Skipping topdims plot ({tname}): no valid samples.")
                        continue
                    aggregated = finalize_layerwise_topdims_storage(tstorage)
                    plot_path = os.path.join(
                        global_plot_save_dir,
                        f"{args.prefix}{out_suffix}_topdims_{tname}_activation.png",
                    )
                    layer_path, avg_path, sink_dims_layer = plot_topdims_mean_activation_curve(
                        aggregated=aggregated,
                        save_path=plot_path,
                        sink_threshold=args.sink_dim_threshold,
                        layer_index=-2,
                        use_abs=True,
                    )
                    print(
                        f"  Saved topdims ({tname}) layer-specific: {layer_path}\n"
                        f"  Saved topdims ({tname}) layer-averaged: {avg_path}\n"
                        f"  Sink dims (>{args.sink_dim_threshold}, {tname}): {sink_dims_layer} "
                        f"(samples={aggregated['num_valid_samples']})"
                    )
                    # Dump full per-layer per-dim arrays for offline replotting (utils/plot_rq*.py).
                    rq_arrays_dir = os.path.join(global_plot_save_dir, "rq_arrays")
                    os.makedirs(rq_arrays_dir, exist_ok=True)
                    np.save(
                        os.path.join(
                            rq_arrays_dir,
                            f"{args.prefix}{out_suffix}_mean_per_dim_{tname}.npy",
                        ),
                        aggregated["mean_all"].detach().cpu().to(torch.float32).numpy(),
                    )
                    np.save(
                        os.path.join(
                            rq_arrays_dir,
                            f"{args.prefix}{out_suffix}_mean_per_dim_{tname}_signed.npy",
                        ),
                        aggregated["mean_all_signed"].detach().cpu().to(torch.float32).numpy(),
                    )

            # Query-to-graph attention: reduce across DDP ranks, save + plot on main.
            accelerator.wait_for_everyone()
            if accelerator.num_processes > 1:
                sum_tensor = q2g_storage["sum_q_to_g"].to(accelerator.device).to(torch.float32)
                row_count_tensor = q2g_storage["count_q_to_g"].to(accelerator.device).to(torch.long)
                sample_count_tensor = torch.tensor(
                    [q2g_storage["num_valid_samples"]],
                    dtype=torch.long,
                    device=accelerator.device,
                )
                sum_tensor = accelerator.reduce(sum_tensor, reduction="sum")
                row_count_tensor = accelerator.reduce(row_count_tensor, reduction="sum")
                sample_count_tensor = accelerator.reduce(sample_count_tensor, reduction="sum")
                q2g_storage["sum_q_to_g"] = sum_tensor.detach().cpu().to(torch.float64)
                q2g_storage["count_q_to_g"] = row_count_tensor.detach().cpu().to(torch.long)
                q2g_storage["num_valid_samples"] = int(sample_count_tensor.item())
                # Reduce per-layer L x K accumulator (lazy-initialised; may be None
                # on a rank that saw no valid samples).
                if q2g_storage.get("sum_l_to_g") is not None:
                    sum_l_tensor = q2g_storage["sum_l_to_g"].to(accelerator.device).to(torch.float32)
                    sum_l_tensor = accelerator.reduce(sum_l_tensor, reduction="sum")
                    q2g_storage["sum_l_to_g"] = sum_l_tensor.detach().cpu().to(torch.float64)

            if accelerator.is_main_process:
                q2g_summary_path = os.path.join(
                    global_plot_save_dir,
                    f"{args.prefix}{out_suffix}_query_to_graph_attention.json",
                )
                save_query_to_graph_attention_summary(
                    storage=q2g_storage,
                    save_path=q2g_summary_path,
                    dataset_name=args.test_dataset,
                )
                q2g_plot_path = os.path.join(
                    global_plot_save_dir,
                    f"{args.prefix}{out_suffix}_query_to_graph_attention.png",
                )
                plot_query_to_graph_attention_summary(
                    storage=q2g_storage,
                    save_path=q2g_plot_path,
                    dataset_name=args.test_dataset,
                )
                print(f"\nTEA-GLM query-to-graph attention:")
                print(f"  n={q2g_storage['num_valid_samples']} valid samples")
                print(f"  Saved summary: {q2g_summary_path}")
                print(f"  Saved plot:    {q2g_plot_path}")

                # Dump query-to-graph + per-layer arrays for offline replotting.
                rq_arrays_dir = os.path.join(global_plot_save_dir, "rq_arrays")
                os.makedirs(rq_arrays_dir, exist_ok=True)
                # Sample-averaged Q x K (matches what gets written into the JSON).
                sums = q2g_storage["sum_q_to_g"].detach().cpu().to(torch.float64)
                counts = q2g_storage["count_q_to_g"].detach().cpu().to(torch.float64)
                safe = counts.clamp(min=1).unsqueeze(-1)
                mean_q_to_g = (sums / safe).to(torch.float32).numpy()
                np.save(
                    os.path.join(
                        rq_arrays_dir,
                        f"{args.prefix}{out_suffix}_query_to_graph.npy",
                    ),
                    mean_q_to_g,
                )
                np.save(
                    os.path.join(
                        rq_arrays_dir,
                        f"{args.prefix}{out_suffix}_query_to_graph_count.npy",
                    ),
                    counts.to(torch.long).numpy(),
                )
                if q2g_storage.get("sum_l_to_g") is not None and q2g_storage["num_valid_samples"] > 0:
                    n_valid = float(q2g_storage["num_valid_samples"])
                    mean_l_to_g = (q2g_storage["sum_l_to_g"] / n_valid).to(torch.float32).numpy()
                    np.save(
                        os.path.join(
                            rq_arrays_dir,
                            f"{args.prefix}{out_suffix}_layer_to_graph.npy",
                        ),
                        mean_l_to_g,
                    )

        # Sink re-emergence: reduce across DDP ranks, save JSONL + summary +
        # rank-indexed plots on main. (top-2 prune, K=args.num_token.)
        if sink_reoccur_enabled:
            accelerator.wait_for_everyone()
            if accelerator.num_processes > 1:
                # All histogram/count tensors get summed across ranks; scalar
                # counters travel as a single meta tensor.
                tensor_keys = [
                    "baseline_count_hist",
                    "post_count_hist",
                    "joint_count_hist",
                    "reoccur_outcome_hist",
                    "rank_outcome_counts",
                    "post_position_hist",
                ]
                reduced = {}
                for key in tensor_keys:
                    t = sink_reoccur_storage[key].to(accelerator.device).to(torch.long)
                    reduced[key] = accelerator.reduce(t, reduction="sum")
                meta_tensor = torch.tensor(
                    [
                        sink_reoccur_storage["num_samples"],
                        sink_reoccur_storage["num_valid_samples"],
                    ],
                    dtype=torch.long,
                    device=accelerator.device,
                )
                meta_tensor = accelerator.reduce(meta_tensor, reduction="sum")
                for key in tensor_keys:
                    sink_reoccur_storage[key] = reduced[key].detach().cpu().to(torch.long)
                sink_reoccur_storage["num_samples"] = int(meta_tensor[0].item())
                sink_reoccur_storage["num_valid_samples"] = int(meta_tensor[1].item())

            if accelerator.is_main_process:
                reoccur_record_path = os.path.join(
                    global_plot_save_dir,
                    f"{args.prefix}{out_suffix}_sink_reoccur_records.jsonl",
                )
                save_sink_token_records_jsonl(
                    records=reoccur_records,
                    save_path=reoccur_record_path,
                )

                summary_json_path = os.path.join(
                    global_plot_save_dir,
                    f"{args.prefix}{out_suffix}_sink_reoccur_summary.json",
                )
                summary_md_path = os.path.join(
                    global_plot_save_dir,
                    f"{args.prefix}{out_suffix}_sink_reoccur_summary.md",
                )
                save_reoccur_summary_table(
                    storage=sink_reoccur_storage,
                    json_path=summary_json_path,
                    md_path=summary_md_path,
                    dataset_name=args.test_dataset,
                )

                rank_plot_path = os.path.join(
                    global_plot_save_dir,
                    f"{args.prefix}{out_suffix}_sink_reoccur_promotion_by_rank.png",
                )
                plot_reoccur_promotion_by_rank(
                    storage=sink_reoccur_storage,
                    save_path=rank_plot_path,
                    title=f"{args.test_dataset}: per-rank survivor outcomes after top-2 sink prune",
                )

                joint_plot_path = os.path.join(
                    global_plot_save_dir,
                    f"{args.prefix}{out_suffix}_sink_reoccur_count_joint.png",
                )
                plot_reoccur_count_joint(
                    storage=sink_reoccur_storage,
                    save_path=joint_plot_path,
                    title=f"{args.test_dataset}: sink count baseline x post top-2 prune",
                )

                baseline_pos_hist = aggregate_baseline_position_hist(
                    pruning_records_index, args.num_token,
                )
                shift_plot_path = os.path.join(
                    global_plot_save_dir,
                    f"{args.prefix}{out_suffix}_sink_distribution_shift.png",
                )
                plot_sink_distribution_shift(
                    baseline_counts=baseline_pos_hist,
                    post_counts=sink_reoccur_storage["post_position_hist"],
                    save_path=shift_plot_path,
                    title=f"{args.test_dataset}: sink position distribution — baseline vs after top-2 prune",
                )

                n_total = sink_reoccur_storage["num_samples"]
                n_valid = sink_reoccur_storage["num_valid_samples"]
                reoccur_hist = sink_reoccur_storage["reoccur_outcome_hist"].tolist()
                any_reoccur = int(sum(reoccur_hist[1:]))
                any_reoccur_pct = (any_reoccur / n_valid * 100.0) if n_valid > 0 else 0.0
                print(f"\nTEA-GLM sink re-emergence summary (pruning_mode=top2, K={args.num_token})")
                print(f"  Total samples:   {n_total}")
                print(f"  Valid samples:   {n_valid}  (>=2 baseline sinks)")
                print(f"  Any re-emerged:  {any_reoccur} ({any_reoccur_pct:.1f}%)")
                print(f"  Saved records:   {reoccur_record_path}")
                print(f"  Saved summary:   {summary_md_path}")
                print(f"  Saved rank plot: {rank_plot_path}")
                print(f"  Saved joint plot: {joint_plot_path}")

        # Reposition: save per-sample swap records on main process.
        if reposition_enabled:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                reposition_record_path = os.path.join(
                    global_plot_save_dir,
                    f"{args.prefix}{out_suffix}_reposition_records.jsonl",
                )
                save_sink_token_records_jsonl(
                    records=reposition_records,
                    save_path=reposition_record_path,
                )
                num_applied = sum(1 for r in reposition_records if r.get("reposition_applied"))
                print(f"\nTEA-GLM reposition summary (mode={args.reposition_mode})")
                print(f"  Samples with swap applied: {num_applied} / {len(reposition_records)}")
                print(f"  Saved reposition records: {reposition_record_path}")

        if not os.path.exists(f'./results/{out_ds_tag}'):
            os.makedirs(f'./results/{out_ds_tag}')

        if accelerator.is_local_main_process:
            eval_pred, eval_decode_label = output_decode(eval_output, eval_label, tokenizer)
            with open(res_path, 'w') as f:
                json.dump(eval_pred, f)
            with open(label_path, 'w') as f:
                json.dump(eval_decode_label, f)
    

if __name__ == "__main__":

    args = parse_args()
    for exp, SEED in enumerate(range(args.exp_num)):
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        transformers.logging.set_verbosity_error()
        accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs, init_kwargs],
                                  gradient_accumulation_steps=args.grad_steps)
        if args.seed != -1:
            SEED = args.seed
        main(args, SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()