"""
Sink-token pruning helpers.

Given a sink-records JSONL produced by the baseline inference run, compute the
list of prompt positions to remove from `inputs_embeds` for each test sample.
Three pruning modes are supported:
  - top2:   remove the top-2 sink tokens per sample
  - all:    remove every detected sink token per sample
  - random: remove `num_prune` random NON-sink graph tokens (seedable)
"""
import json
import random
from typing import Any, Dict, List, Tuple

import torch


PRUNING_MODES = ("top2", "all", "random")


def load_sink_records(path: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Load sink records JSONL and index by (step, batch_index)."""
    index: Dict[Tuple[int, int], Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (int(rec["step"]), int(rec["batch_index"]))
            index[key] = rec
    return index


def _all_graph_positions(is_node_row: torch.Tensor) -> List[int]:
    return torch.nonzero(is_node_row > 0, as_tuple=False).reshape(-1).tolist()


def compute_prune_positions_batch(
    *,
    records_index: Dict[Tuple[int, int], Dict[str, Any]],
    is_node: torch.Tensor,
    step: int,
    mode: str,
    num_prune: int = 2,
    seed: int = 0,
) -> List[List[int]]:
    """
    Returns per-sample prompt positions to remove from `inputs_embeds`.

    is_node: [B, T] — graph-token mask (only needed for mode='random').
    """
    if mode not in PRUNING_MODES:
        raise ValueError(f"Unknown pruning mode: {mode}")

    batch_size = is_node.shape[0]
    out: List[List[int]] = []
    for b in range(batch_size):
        rec = records_index.get((int(step), int(b)))
        if mode == "top2":
            positions = list(rec.get("top2_graph_token_positions", [])) if rec else []
        elif mode == "all":
            positions = list(rec.get("graph_token_positions", [])) if rec else []
        else:  # random
            all_g = _all_graph_positions(is_node[b])
            sinks = {int(p) for p in (rec.get("graph_token_positions", []) if rec else [])}
            non_sinks = [p for p in all_g if p not in sinks]
            rng = random.Random(int(seed) ^ (int(step) * 1_000_003 + int(b)))
            k = min(int(num_prune), len(non_sinks))
            positions = rng.sample(non_sinks, k) if k > 0 else []
        out.append([int(p) for p in positions])
    return out


def compute_reposition_perm_batch(
    *,
    records_index: Dict[Tuple[int, int], Dict[str, Any]],
    is_node: torch.Tensor,
    step: int,
    mode: str,
    num_swap: int = 2,
    seed: int = 0,
) -> Tuple[List[Any], List[Any]]:
    """
    Per sample, build a permutation of length T that swaps `num_swap` sink
    positions with `num_swap` non-sink positions (sampled uniformly without
    replacement). Returns:
        perm_per_sample : list[B] of length-T int lists (or None when no swap
                          was performed because sinks or non-sinks were empty).
        swap_log        : list[B] of {"sink_positions":[...], "nonsink_positions":[...],
                                      "graph_token_indices":[...]} or None.

    `graph_token_indices` records the ORIGINAL K-space indices (0..K-1) that
    were swapped, so downstream aggregators can track per-index change ratios.
    """
    if mode != "swap_sink_nonsink":
        raise ValueError(f"Unknown reposition mode: {mode}")

    B, T = is_node.shape
    perm_per_sample: List[Any] = []
    swap_log: List[Any] = []

    for b in range(B):
        rec = records_index.get((int(step), int(b)))
        all_graph = torch.nonzero(is_node[b] > 0, as_tuple=False).reshape(-1).tolist()
        sinks = [int(p) for p in (rec.get("graph_token_positions", []) if rec else [])]
        sink_set = set(sinks)
        non_sinks = [int(p) for p in all_graph if int(p) not in sink_set]

        rng = random.Random(int(seed) ^ (int(step) * 1_000_003 + int(b)))
        k = min(int(num_swap), len(sinks), len(non_sinks))
        if k == 0:
            perm_per_sample.append(None)
            swap_log.append(None)
            continue

        chosen_sinks = rng.sample(sinks, k)
        chosen_nonsinks = rng.sample(non_sinks, k)

        perm = list(range(T))
        for s, ns in zip(chosen_sinks, chosen_nonsinks):
            perm[s], perm[ns] = perm[ns], perm[s]

        # Map swapped prompt positions back to K-space indices for the plot.
        graph_prompt_to_idx = {int(p): i for i, p in enumerate(all_graph)}
        involved_k_indices = sorted({
            graph_prompt_to_idx[p] for p in (chosen_sinks + chosen_nonsinks)
            if int(p) in graph_prompt_to_idx
        })

        perm_per_sample.append(perm)
        swap_log.append({
            "sink_positions": chosen_sinks,
            "nonsink_positions": chosen_nonsinks,
            "graph_token_indices": involved_k_indices,
        })
    return perm_per_sample, swap_log


def reposition_output_suffix(mode: str, num_swap: int, seed: int) -> str:
    if mode == "swap_sink_nonsink":
        return f"_reposition_swap_k{num_swap}_seed{seed}"
    raise ValueError(f"Unknown reposition mode: {mode}")


def pruning_output_suffix(mode: str, num_prune: int, seed: int) -> str:
    # Include the seed in every mode's suffix so multi-seed sweeps can share
    # uniform tooling (matched counts, mean/std aggregation) even for modes
    # whose outputs are deterministic across seeds under greedy decoding.
    if mode == "top2":
        return f"_prune_top2_seed{seed}"
    if mode == "all":
        return f"_prune_all_seed{seed}"
    if mode == "random":
        return f"_prune_random{num_prune}_seed{seed}"
    raise ValueError(f"Unknown pruning mode: {mode}")
