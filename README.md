# GraphSink: Graph Sink Analysis for Graph Language Models

This repository contains the experimental code for our study of **attention sinks** in Graph Language Models (GLMs). Attention sinks are graph tokens that consistently receive disproportionate attention and spike on a small set of hidden-state dimensions at the second-to-last transformer layer, largely independent of the downstream query. We probe whether these tokens are *causally* important for task performance, or whether they are interpretable artifacts of the LLM's attention machinery — and we do this in parallel across two representative GLMs so the findings are not specific to one architecture.

The two GLMs analyzed are [LLaGA](LLaGA/) (Chen et al., ICML 2024) and [TEA-GLM](TEA-GLM/) (Wang et al., NeurIPS 2024). Both wrap a frozen Vicuna-7B backbone behind a learned graph projector, but differ in the projector design and the number of graph tokens fed to the LLM (LLaGA varies by template; TEA-GLM is fixed at K=5). All experiments documented here are run on the **node-classification (NC)** task across **Cora**, **PubMed**, and **ogbn-arxiv**. For LP-task variants, see [evaluation.md](evaluation.md).

For environment setup, dataset download, model training, and upstream attribution, see [LLaGA/README.md](LLaGA/README.md) and [TEA-GLM/README.md](TEA-GLM/README.md). This top-level README only documents the four sink-analysis experiments we ran on top of those models.

## Repository layout

```
glm_sink/
├── LLaGA/
│   ├── eval/                       # eval_pretrain.py, run_graph_sink.py, eval_res_*.py
│   ├── scripts/                    # *.sh driver scripts for each experiment
│   ├── utils/                      # activation_probes, attention_probes, logit_lens, dimension_pruning, graph_remap
│   ├── analysis/                   # generated: sink_records.jsonl, heatmaps, logit-lens plots
│   └── results/                    # generated: prediction *.jsonl per experiment
│   
├── TEA-GLM/
│   ├── train_glm.py                # main inference/training entry; gates --logit_lens, --prune_sink_tokens, etc.
│   ├── analyze_attention_sinks.py  # standalone sink-identification & dimension analysis
│   ├── evaluate_regex*.py          # per-experiment aggregation scripts
│   ├── scripts/                    # test_citation*.sh driver scripts
│   ├── utils/                      # activation, attention_probe, sink_pruning, logit_lens
│   ├── analysis/                   # generated: per-dataset global_stats/, logit_lens/
│   └── results/                    # generated: prediction *.txt per experiment
└── README.md                       # this file
```

## The four sink experiments

Each subsection describes the experiment conceptually, then gives a minimal run snippet and the output locations, separately for LLaGA and TEA-GLM.

### 1. Sink token identification

**Concept.** For each test sample we extract the per-token hidden state at the second-to-last layer, apply RMSNorm, and score each graph token by its magnitude on a small set of "spike dimensions" discovered empirically from training-time activations. Tokens whose score exceeds a per-model threshold are flagged as sinks. LLaGA's NC spike dims are `[1512, 2298, 2533]`; TEA-GLM's are `[1512, 2533, 3431]` (both in a 4096-dim hidden space; threshold 35.0 for TEA-GLM).

**LLaGA.**
- Detection utilities: [LLaGA/utils/activation_probes.py](LLaGA/utils/activation_probes.py), [LLaGA/utils/dimension_pruning.py](LLaGA/utils/dimension_pruning.py) (`find_spike_dims`).
- Discovery pipeline (stages `discover_dims → detect_tokens → discover_spikes → detect_and_analyze`): [LLaGA/eval/run_graph_sink.py](LLaGA/eval/run_graph_sink.py).
- Run: `python eval/run_graph_sink.py --dataset cora --task nc --template ND --stage detect_and_analyze`.
- Outputs: `LLaGA/analysis/{dataset}_{template}/sink_records.jsonl`, `activation_topdims.json`, `sink_dim_mean_activation.png`, `llaga_topdims_counts.png`.

**TEA-GLM.**
- Detection: [TEA-GLM/utils/activation.py](TEA-GLM/utils/activation.py) (`detect_sink_tokens`, RMSNorm score on layer 30).
- Standalone analysis (post-hoc, from saved activations): [TEA-GLM/analyze_attention_sinks.py](TEA-GLM/analyze_attention_sinks.py).
- Run: `bash scripts/test_citation.sh` to populate `analysis/{dataset}/`, then `python analyze_attention_sinks.py --dataset cora` for the summary plots.
- Outputs: `TEA-GLM/analysis/{dataset}/global_stats/{prefix}_sink_records.jsonl`, `_sink_dim_summary.json`, `_sink_dim_mean_activation.png`, `_sink_token_histogram.png`.

### 2. Cross-attention between sink tokens and query tokens

**Concept.** We aggregate the per-layer query→graph attention matrix (head-averaged, then layer-averaged, then sample-averaged) and inspect the columns corresponding to sink positions. The output quantifies how much of the query's attention mass is concentrated on sinks vs. non-sink graph tokens, broken down by layer and by query-token offset relative to the end of the graph block.

**LLaGA.**
- Driver: [LLaGA/eval/eval_pretrain.py](LLaGA/eval/eval_pretrain.py) with `--attention_probe`.
- Utilities (compute + plot): [LLaGA/utils/attention_probes.py](LLaGA/utils/attention_probes.py) — `compute_layerwise_query_to_graph_attention`, `extract_sink_columns_from_query_graph_attention`, `plot_layeravg_query_sink_attention`, `analyze_and_plot_sample_attention_to_sink_layeravg`.
- Run: append `--attention_probe` to a standard NC eval command (see [LLaGA/README.md](LLaGA/README.md#step-4-evaluation) for the base command).
- Outputs: `LLaGA/analysis/{dataset}_{template}/cross_attention_layer_vs_graph_heatmap.png`, per-sample `sample_*_layer_avg.png`.

**TEA-GLM.**
- Utilities: [TEA-GLM/utils/attention_probe.py](TEA-GLM/utils/attention_probe.py) — `init_query_to_graph_attention_storage`, `update_query_to_graph_attention_storage`, `build_query_to_graph_attention_summary`. Invoked from [TEA-GLM/train_glm.py](TEA-GLM/train_glm.py)'s eval branch.
- Run: `bash scripts/test_citation.sh` (attention probing runs as part of the standard NC eval pass).
- Outputs: `TEA-GLM/analysis/{dataset}/global_stats/{prefix}_query_to_graph_attention.json` and `.png`.

### 3. Intervention experiments

We test the *causal* role of sinks via three interventions: (a) prune sinks at inference; (b) move sinks to the front of the graph block (LLaGA-only, to test whether position alone explains sink behavior); and (c) swap sinks with non-sink positions. A follow-up "re-emergence" probe checks whether new sinks appear after the original ones are removed.

#### 3a. Sink token pruning

**LLaGA.**
- Driver: [LLaGA/eval/eval_pretrain.py](LLaGA/eval/eval_pretrain.py) with `--pruning --pruning_mode {sink_token|control_nonsink_token|sink_dim|control_nonsink_dim}`, granularity `top2` or `all`, and `--sink_records_path` pointing at the JSONL from Experiment 1.
- Wrappers: [LLaGA/scripts/pruning_sink.sh](LLaGA/scripts/pruning_sink.sh), [LLaGA/scripts/pruning_sink_multi.sh](LLaGA/scripts/pruning_sink_multi.sh), [LLaGA/scripts/pruning.sh](LLaGA/scripts/pruning.sh).
- Outputs: `LLaGA/results_phc3mn/{dataset}_nc_*_predictions_prune_{sinktoken|nonsinktoken}_{top2|all}_run{N}.jsonl`.
- Aggregation: [LLaGA/eval/eval_res_pruning.py](LLaGA/eval/eval_res_pruning.py) — e.g. `python eval/eval_res_pruning.py --task nc --target sink --pruning_mode top2`.

**TEA-GLM.**
- Driver: [TEA-GLM/train_glm.py](TEA-GLM/train_glm.py) with `--prune_sink_tokens --pruning_mode {top2|all|random}`. Pruning logic in [TEA-GLM/utils/sink_pruning.py](TEA-GLM/utils/sink_pruning.py) (`compute_prune_positions_batch`).
- Wrappers: [TEA-GLM/scripts/test_citation_sinkanalysis.sh](TEA-GLM/scripts/test_citation_sinkanalysis.sh) (top-2 and all), [TEA-GLM/scripts/test_citation_sinkanalysis_top2.sh](TEA-GLM/scripts/test_citation_sinkanalysis_top2.sh), [TEA-GLM/scripts/test_citation_sinkanalysis_random.sh](TEA-GLM/scripts/test_citation_sinkanalysis_random.sh) (random non-sink control, seeds 123–137).
- Outputs: `TEA-GLM/results/{dataset}/{prefix}_prune_{top2|all|random2}_seed{S}_run{N}_*.txt`.
- Aggregation: [TEA-GLM/evaluate_regex_prune.py](TEA-GLM/evaluate_regex_prune.py) for top-2 / all, [TEA-GLM/evaluate_regex_random.py](TEA-GLM/evaluate_regex_random.py) for the random control.

#### 3b. Sink reposition to front (LLaGA-only)

**Concept.** Move the detected sink token(s) to the front of the graph block, leaving content otherwise intact. If sink behavior is a *positional* phenomenon, we expect the relocated tokens (or whichever tokens land in their old slots) to inherit the sink role; if it is a *content* phenomenon, the relocated tokens should keep theirs. We instrument the post-reposition sequence to measure both.

- Driver: [LLaGA/eval/eval_pretrain.py](LLaGA/eval/eval_pretrain.py) with `--reposition_mode {front_top2|front_all}`. Helper: [LLaGA/utils/graph_remap.py](LLaGA/utils/graph_remap.py).
- Wrapper: [LLaGA/scripts/reposition.sh](LLaGA/scripts/reposition.sh).
- Outputs: `LLaGA/results_phc3mn/{dataset}_nc_*_predictions_reposition_front_{top2|all}.jsonl`; post-reposition sink statistics in `LLaGA/analysis/{dataset}_{template}/reposition_front_all_sink_ratio.json`.
- Evaluators: [LLaGA/eval/eval_res_reposition_swap.py](LLaGA/eval/eval_res_reposition_swap.py) (accuracy), [LLaGA/eval/eval_reposition_sink_ratio.py](LLaGA/eval/eval_reposition_sink_ratio.py) (sink survival/migration).

#### 3c. Sink ↔ non-sink swap

**Concept.** Swap K sink positions with K non-sink positions inside the graph block (deterministic per seed). Compared to the front-reposition above, this disentangles "moved to position 0" from "moved anywhere new" — and the controlled non-sink targets give a paired-sample test for whether the *identity* of the swapped tokens, not just their positions, drives the accuracy delta.

**LLaGA.**
- Driver: `eval_pretrain.py --reposition_mode swap_sink_nonsink --reposition_seed S`.
- Wrapper: [LLaGA/scripts/reposition_swap_multi.sh](LLaGA/scripts/reposition_swap_multi.sh) (multi-run with varying seeds).
- Outputs: `LLaGA/results_phc3mn/{dataset}_nc_*_predictions_reposition_swap_sink_nonsink_k{K}_run{N}.jsonl`.
- Aggregation: [LLaGA/eval/eval_res_reposition_swap.py](LLaGA/eval/eval_res_reposition_swap.py).

**TEA-GLM.**
- Logic: [TEA-GLM/utils/sink_pruning.py](TEA-GLM/utils/sink_pruning.py) (`compute_reposition_perm_batch`).
- Wrapper: [TEA-GLM/scripts/test_citation_reposition.sh](TEA-GLM/scripts/test_citation_reposition.sh).
- Outputs: `TEA-GLM/results/{dataset}/{prefix}_reposition_swap_k{num_swap}_seed{S}_*.txt`; per-sample swap log `{prefix}_reposition_swap_*_reposition_records.jsonl`.
- Aggregation: [TEA-GLM/evaluate_regex_reposition.py](TEA-GLM/evaluate_regex_reposition.py) (single seed; per-K-index change ratio), [TEA-GLM/evaluate_regex_reposition_multiseed.py](TEA-GLM/evaluate_regex_reposition_multiseed.py) (multi-seed mean ± std; default seeds 42–46).

#### 3d. Sink re-emergence (follow-up to 3a-`all`)

After pruning all detected sinks, we run sink detection again on the shortened sequence to ask: do new sinks emerge among the remaining graph tokens? This tests whether sink behavior is a redundant property of a few tokens or a structural feature the model will *recreate* if you remove the carriers.

- LLaGA: [LLaGA/scripts/reoccur_sink.sh](LLaGA/scripts/reoccur_sink.sh) — gated by `--sink_reoccur --pruning --pruning_mode all`. Outputs `sink_reoccur.jsonl`, `sink_reoccur_distribution.png`, `sink_distribution_shift.png` in `LLaGA/analysis/`.
- TEA-GLM: [TEA-GLM/scripts/test_citation_reoccur.sh](TEA-GLM/scripts/test_citation_reoccur.sh) — gated by `--prune_sink_tokens --pruning_mode=all --sink_reoccur`. Outputs `{prefix}_prune_all_seed{S}_sink_reoccur_records.jsonl`, `_sink_reoccur_summary.json`, `_sink_reoccur_promotion_by_rank.png`, `_sink_distribution_shift.png`.

### 4. Logit lens analysis

**Concept.** For each graph token, at each layer, project the residual stream through the model's `final_norm + lm_head` and read off the top-1 vocabulary token (and its probability) the model would output if it stopped at that layer. Aggregating modal tokens across samples produces a `[layer × graph-token]` heatmap that reveals (i) at what depth task-relevant tokens emerge, and (ii) which graph positions — sink or not — are interpretable early vs. late.

**LLaGA.**
- Implementation: [LLaGA/utils/logit_lens.py](LLaGA/utils/logit_lens.py) (`compute_logit_lens`, `aggregate_logit_lens`, `plot_logit_lens_heatmap`).
- Driver: [LLaGA/scripts/eval_logit_lens.sh](LLaGA/scripts/eval_logit_lens.sh) — runs `eval_pretrain.py --logit_lens --use_existing_sink_records` (re-uses the JSONL from Experiment 1).
- Outputs: `LLaGA/analysis/{dataset}_{template}/logit_lens/logit_lens.png` (sink columns marked `<s>` for true sinks, `<sp>` for padded sinks).

**TEA-GLM.**
- Implementation: [TEA-GLM/utils/logit_lens.py](TEA-GLM/utils/logit_lens.py) (same API).
- Driver: `LOGIT_LENS=1 bash` [TEA-GLM/scripts/test_single.sh](TEA-GLM/scripts/test_single.sh) — routes the `--logit_lens` flag into [TEA-GLM/train_glm.py](TEA-GLM/train_glm.py)'s eval branch. Logit lens is computed on the first test sample of each dataset to keep memory bounded.
- Outputs: `TEA-GLM/analysis/{dataset}/logit_lens/{prefix}_logit_lens.png` and `.pdf`.

## Quick reproduction map (NC task)

| Experiment | LLaGA driver | LLaGA results | TEA-GLM driver | TEA-GLM results |
|---|---|---|---|---|
| 1. Sink identification | [`eval/run_graph_sink.py`](LLaGA/eval/run_graph_sink.py) | `LLaGA/analysis/{ds}_{tmpl}/sink_records.jsonl` | [`analyze_attention_sinks.py`](TEA-GLM/analyze_attention_sinks.py) + [`scripts/test_citation.sh`](TEA-GLM/scripts/test_citation.sh) | `TEA-GLM/analysis/{ds}/global_stats/*_sink_records.jsonl` |
| 2. Cross-attention | `eval_pretrain.py --attention_probe` | `.../cross_attention_layer_vs_graph_heatmap.png` | [`scripts/test_citation.sh`](TEA-GLM/scripts/test_citation.sh) | `.../global_stats/*_query_to_graph_attention.{json,png}` |
| 3a. Sink pruning | [`scripts/pruning_sink_multi.sh`](LLaGA/scripts/pruning_sink_multi.sh) | `results_phc3mn/*_predictions_prune_*.jsonl` | [`scripts/test_citation_sinkanalysis.sh`](TEA-GLM/scripts/test_citation_sinkanalysis.sh) | `results/{ds}/*_prune_*_seed{S}_run{N}_*.txt` |
| 3a-control. Random non-sink | [`scripts/pruning.sh`](LLaGA/scripts/pruning.sh) (`control_nonsink_token`) | `results_phc3mn/*_predictions_prune_nonsinktoken_*.jsonl` | [`scripts/test_citation_sinkanalysis_random.sh`](TEA-GLM/scripts/test_citation_sinkanalysis_random.sh) | `results/{ds}/*_prune_random2_seed{S}_*.txt` |
| 3b. Reposition to front | [`scripts/reposition.sh`](LLaGA/scripts/reposition.sh) | `results_phc3mn/*_predictions_reposition_front_*.jsonl` | — (LLaGA-only) | — |
| 3c. Sink↔non-sink swap | [`scripts/reposition_swap_multi.sh`](LLaGA/scripts/reposition_swap_multi.sh) | `results_phc3mn/*_predictions_reposition_swap_sink_nonsink_k*_run*.jsonl` | [`scripts/test_citation_reposition.sh`](TEA-GLM/scripts/test_citation_reposition.sh) | `results/{ds}/*_reposition_swap_k*_seed*_*.txt` |
| 3d. Sink re-emergence | [`scripts/reoccur_sink.sh`](LLaGA/scripts/reoccur_sink.sh) | `analysis/{ds}_{tmpl}/sink_reoccur.jsonl` | [`scripts/test_citation_reoccur.sh`](TEA-GLM/scripts/test_citation_reoccur.sh) | `analysis/{ds}/global_stats/*_sink_reoccur_*.{jsonl,json,png}` |
| 4. Logit lens | [`scripts/eval_logit_lens.sh`](LLaGA/scripts/eval_logit_lens.sh) | `analysis/{ds}_{tmpl}/logit_lens/logit_lens.png` | `LOGIT_LENS=1 bash` [`scripts/test_single.sh`](TEA-GLM/scripts/test_single.sh) | `analysis/{ds}/logit_lens/*_logit_lens.{png,pdf}` |

## Notes

- TEA-GLM reposition multi-seed aggregation defaults to seeds 42–46; random-pruning sweeps use seeds 123–137. LLaGA uses `--reposition_seed` / `--pruning_seed` with run indices `_run{N}` to disambiguate replicates.
- For the link-prediction (LP) variants of every experiment above, see [evaluation.md](evaluation.md) (aggregation commands) and the `lp_*.sh` scripts under [LLaGA/scripts/](LLaGA/scripts/) and [TEA-GLM/scripts/](TEA-GLM/scripts/).
