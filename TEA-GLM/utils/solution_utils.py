import numpy as np
import torch
import json
import random, os
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from torch_geometric.data import Batch, Data
from typing import Dict, Any, Optional, Literal, Sequence, List, Tuple, Union

"""
This solution utils file saves all the utility functions for implementing potential solutions for mitigating attention sinks
"""

############ Attention Rollout ###########

def _get_graph_and_query_indices_for_example(
    *,
    is_node: torch.Tensor,          # [T] bool or {0,1}
    attention_mask: torch.Tensor,   # [T] bool or {0,1}
    expected_k: int = 5,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns:
      key_idx:   [K] indices of graph tokens (sorted)
      query_idx: [Q] indices of text tokens AFTER last graph token (sorted)
    """
    # valid tokens only
    valid = attention_mask.bool()

    node_mask = is_node.bool() & valid
    key_idx = torch.nonzero(node_mask, as_tuple=False).squeeze(-1)

    if key_idx.numel() < expected_k:
        return None

    # If there are more than K node tokens (unlikely), keep the first K in order.
    key_idx = torch.sort(key_idx).values[:expected_k]

    last_k = int(key_idx[-1].item())
    # queries are tokens after the graph-token block
    query_mask = valid.clone()
    query_mask[: last_k + 1] = False
    query_idx = torch.nonzero(query_mask, as_tuple=False).squeeze(-1)

    if query_idx.numel() == 0:
        return None

    return key_idx, query_idx


def init_text2graph_rollout_dataset_agg(
    *,
    q_max: int,
    k_graph: int = 5,
) -> Dict[str, Any]:
    """
    Stores dataset-level sum/count for the rollout matrix slice:
      rollout[query_positions_after_graph, graph_positions]
    """
    return {
        "q_max": int(q_max),
        "k_graph": int(k_graph),
        "sum": torch.zeros(q_max, k_graph, dtype=torch.float32),  # CPU
        "count": torch.zeros(q_max, dtype=torch.long),            # CPU
        "n_samples_used": 0,
        "n_skipped": 0,
    }


@torch.no_grad()
def update_text2graph_dataset_rollout_agg(
    *,
    storage: Dict[str, Any],
    outputs,                         # HF outputs with .attentions
    attention_mask: torch.Tensor,    # [B, T]
    is_node: torch.Tensor,           # [B, T] bool
    expected_k: int = 5,
    add_identity: bool = True,
    eps: float = 1e-12,
) -> None:
    """
    Dataset aggregation for rollout(text -> graph).

    For each example b:
      - build rollout R over layers: [T,T]
      - slice rows=query after graph block, cols=graph token indices
      - accumulate into storage['sum'] (CPU float32)
      - update storage['count'] for rows that exist
    """
    attns = getattr(outputs, "attentions", None)
    if attns is None:
        raise ValueError("outputs.attentions is None. Make sure output_attentions=True.")

    B, T = attention_mask.shape
    q_max = int(storage["q_max"])
    K = int(storage["k_graph"])

    # attns: tuple length L, each [B,H,T,T]
    L = len(attns)

    for b in range(B):
        idxs = _get_graph_and_query_indices_for_example(    # Stores the graph & text indices
            is_node=is_node[b],
            attention_mask=attention_mask[b],
            expected_k=expected_k,
        )
        if idxs is None:
            storage["n_skipped"] += 1
            continue

        key_idx, query_idx = idxs
        if key_idx.numel() != K:
            storage["n_skipped"] += 1
            continue

        q_len = min(int(query_idx.numel()), q_max)
        if q_len <= 0:
            storage["n_skipped"] += 1
            continue
        query_idx = query_idx[:q_len]

        # ---- rollout init: Identity [T,T] on device ----
        device = attns[0].device
        R = torch.eye(T, device=device, dtype=torch.float32)

        # ---- multiply layer by layer ----
        for l in range(L):
            A = attns[l]  # [B,H,T,T]
            if A.dim() != 4:
                raise ValueError(f"Unexpected attention tensor dim={A.dim()} at layer={l}.")

            A_b = A[b].to(torch.float32)          # [H,T,T]
            A_mean = A_b.mean(dim=0)              # [T,T]

            if add_identity:
                A_mean = A_mean + torch.eye(T, device=device, dtype=torch.float32)

            # row-normalize
            A_mean = A_mean / (A_mean.sum(dim=-1, keepdim=True) + eps)

            # rollout update
            R = R @ A_mean

        # ---- slice text->graph from rollout ----
        sub = R.index_select(0, query_idx).index_select(1, key_idx)  # [q_len,K]

        # ---- min-max normalize to [0,1] (per-sample, on the sliced block) ----
        sub_min = sub.min()
        sub_max = sub.max()
        denom = (sub_max - sub_min).clamp_min(eps)
        sub = (sub - sub_min) / denom

        # ---- accumulate ----
        storage["sum"][:q_len, :] += sub.detach().to(torch.float32).cpu()
        storage["count"][:q_len] += 1
        storage["n_samples_used"] += 1


############# Attention Head Summation ############

def init_head_graph_attn_storage(outputs, expected_k: int = 5) -> Dict[str, Any]:
    """
    Initialize an accumulator for per-layer, per-head attention mass:
      - num_non_sink[l,h]
      - den_all_graph[l,h]
      - sink_mass[l,h]
      - n_samples_used, n_skipped
    """
    attns = getattr(outputs, "attentions", None)
    if attns is None:
        raise ValueError("outputs.attentions is None. Make sure output_attentions=True.")
    if len(attns) == 0:
        raise ValueError("outputs.attentions is empty.")

    # HuggingFace convention: tuple length L, each [B,H,T,T]
    A0 = attns[0]
    if A0.dim() != 4:
        raise ValueError(f"Expected attention tensor [B,H,T,T], got dim={A0.dim()} shape={tuple(A0.shape)}")

    _, H, _, _ = A0.shape
    L = len(attns)

    # Use float64 on CPU for stable accumulation across many batches
    storage = {
        "num_layers": L,
        "num_heads": H,
        "expected_k": int(expected_k),
        "num_non_sink": torch.zeros(L, H, dtype=torch.float64),
        "den_all_graph": torch.zeros(L, H, dtype=torch.float64),
        "sink_mass": torch.zeros(L, H, dtype=torch.float64),
        "n_samples_used": 0,
        "n_skipped": 0,
    }
    return storage


def attention_head_graph_attention_weights_update(
    *,
    storage: Dict[str, Any],
    outputs,                         # HF outputs with .attentions
    attention_mask: torch.Tensor,    # [B, T]
    is_node: torch.Tensor,           # [B, T] bool
    expected_k: int = 5,
    sink_token_pos: int = 2,         # 0-based position within the 5 graph tokens; 2 == "third graph token"
    use_only_text_after_graph: bool = True,
    eps: float = 1e-12,
) -> None:
    """
    Aggregates per-layer, per-head attention mass from text->graph.

    For each sample b:
      - key_idx: indices of the 5 graph tokens in the full sequence
      - query_idx: indices of "text tokens" (by default only those after the graph block)
      - for each layer l, head h:
          denom = sum_{q in query_idx} sum_{k in key_idx} A[l,b,h,q,k]
          num   = sum_{q in query_idx} sum_{k in non_sink_keys} A[l,b,h,q,k]
          sink  = denom - num  (equivalently attention to sink key only)

    Stores sums into:
      storage["num_non_sink"][l,h]
      storage["den_all_graph"][l,h]
      storage["sink_mass"][l,h]
    """

    attns = getattr(outputs, "attentions", None)
    if attns is None:
        raise ValueError("outputs.attentions is None. Make sure output_attentions=True.")

    L = len(attns)
    if L != storage["num_layers"]:
        raise ValueError(f"Storage num_layers={storage['num_layers']} but outputs has {L} layers.")

    B, T = attention_mask.shape
    K = int(expected_k)

    if sink_token_pos < 0 or sink_token_pos >= K:
        raise ValueError(f"sink_token_pos must be in [0,{K-1}] but got {sink_token_pos}")

    for b in range(B):
        # ---- find graph token positions ----
        # This helper is from your earlier code path.
        # Expected return: (key_idx, query_idx)
        idxs = _get_graph_and_query_indices_for_example(
            is_node=is_node[b],
            attention_mask=attention_mask[b],
            expected_k=expected_k,
        )
        if idxs is None:
            storage["n_skipped"] += 1
            continue

        key_idx, query_idx = idxs  # key_idx: [K], query_idx: [Q_all or Q_after_graph]
        if key_idx.numel() != K:
            storage["n_skipped"] += 1
            continue

        # Some pipelines may return query_idx over all non-graph tokens.
        # If you want “only tokens after the graph block”, enforce it here.
        if use_only_text_after_graph:
            last_graph = int(key_idx.max().item())
            # keep only valid, unmasked tokens strictly after last graph token
            valid = (query_idx > last_graph)
            if valid.any():
                query_idx = query_idx[valid]
            else:
                storage["n_skipped"] += 1
                continue

        if query_idx.numel() == 0:
            storage["n_skipped"] += 1
            continue

        # non-sink keys: all graph tokens except the sink position
        # sink_token_pos is relative within the K graph tokens, so map to absolute index:
        sink_abs = key_idx[sink_token_pos].view(1)  # [1]
        non_sink_mask = torch.ones(K, dtype=torch.bool, device=key_idx.device)
        non_sink_mask[sink_token_pos] = False
        non_sink_abs = key_idx[non_sink_mask]       # [K-1]

        # ---- accumulate per layer ----
        for l in range(L):
            A = attns[l]  # [B,H,T,T]
            if A.dim() != 4:
                raise ValueError(f"Unexpected attention tensor dim={A.dim()} at layer={l}.")

            # [H,T,T] in float32 for safe summation; keep on GPU for indexing speed
            A_b = A[b].to(torch.float32)

            # Gather sub-tensor: [H, Q, K]
            # rows: query positions, cols: key positions
            sub_all = A_b.index_select(1, query_idx).index_select(2, key_idx)
            # Denominator per head: sum over Q and K
            den = sub_all.sum(dim=(1, 2))  # [H]

            # Numerator: exclude sink column (K-1 cols)
            sub_non_sink = A_b.index_select(1, query_idx).index_select(2, non_sink_abs)
            num = sub_non_sink.sum(dim=(1, 2))  # [H]

            # Sink mass directly (attention to sink key only)
            # (Equivalent to A_b[..., sink_abs] summed over queries)
            sub_sink = A_b.index_select(1, query_idx).index_select(2, sink_abs)  # [H,Q,1]
            sink = sub_sink.sum(dim=(1, 2))  # [H]

            # Move to CPU float64 and accumulate
            storage["den_all_graph"][l] += den.detach().to(torch.float64).cpu()
            storage["num_non_sink"][l] += num.detach().to(torch.float64).cpu()
            storage["sink_mass"][l] += sink.detach().to(torch.float64).cpu()

        storage["n_samples_used"] += 1


def finalize_head_graph_attn_stats(storage: Dict[str, Any], eps: float = 1e-12) -> Dict[str, Any]:
    """
    Produce per-layer, per-head ratios from accumulated sums.
    """
    den = storage["den_all_graph"].clone()
    num = storage["num_non_sink"].clone()
    sink = storage["sink_mass"].clone()

    ratio_non_sink = num / (den + eps)   # [L,H]
    ratio_sink = sink / (den + eps)      # [L,H]

    return {
        "ratio_non_sink": ratio_non_sink,
        "ratio_sink": ratio_sink,
        "num_non_sink": num,
        "sink_mass": sink,
        "den_all_graph": den,
        "n_samples_used": storage["n_samples_used"],
        "n_skipped": storage["n_skipped"],
    }
            


@torch.no_grad()
def per_example_head_graph_attention_stats(
    *,
    outputs,                         # HF outputs with .attentions: tuple[L] of [B,H,T,T]
    attention_mask: torch.Tensor,    # [B, T]
    is_node: torch.Tensor,           # [B, T] bool
    expected_k: int = 5,
    sink_k: int = 2,                 # third graph token => 0-based index 2
    include_only_after_graph: bool = True,
) -> List[Dict[str, Any]]:
    """
    Returns per-example attention mass stats.
    For each example b, returns:
      - den_all_graph: [L,H]
      - sink_mass:     [L,H]
      - non_sink_mass: [L,H]
      - sink_ratio:    [L,H]  (sink_mass / den_all_graph)
      - key_idx, query_idx (for debugging)
    """

    attns = getattr(outputs, "attentions", None)
    if attns is None:
        raise ValueError("outputs.attentions is None. Make sure output_attentions=True.")

    B, T = attention_mask.shape
    L = len(attns)
    H = attns[0].shape[1]

    out: List[Dict[str, Any]] = []

    for b in range(B):
        idxs = _get_graph_and_query_indices_for_example(
            is_node=is_node[b],
            attention_mask=attention_mask[b],
            expected_k=expected_k,
        )
        if idxs is None:
            out.append({"ok": False, "reason": "no_graph_or_query"})
            continue

        key_idx, query_idx = idxs  # key_idx: [K], query_idx: [Q]
        K = int(key_idx.numel())
        if K != expected_k:
            out.append({"ok": False, "reason": f"expected_k_mismatch(K={K})"})
            continue

        # optionally only include text tokens after the graph block
        if include_only_after_graph:
            # _get_graph_and_query_indices_for_example should already return "after graph"
            # but keep this flag for clarity.
            pass

        # [L,H]
        den_all_graph = torch.zeros((L, H), dtype=torch.float32, device="cpu")
        sink_mass     = torch.zeros((L, H), dtype=torch.float32, device="cpu")
        non_sink_mass = torch.zeros((L, H), dtype=torch.float32, device="cpu")

        # split key indices into sink and non-sink
        sink_pos = key_idx[sink_k].view(1)  # [1]
        non_sink_mask = torch.ones(K, dtype=torch.bool, device=key_idx.device)
        non_sink_mask[sink_k] = False
        non_sink_pos = key_idx[non_sink_mask]  # [K-1]

        # compute per layer
        for l in range(L):
            A = attns[l]
            if A.dim() != 4:
                raise ValueError(f"Unexpected attention dim={A.dim()} at layer={l}.")

            # A[b]: [H, T, T]
            A_b = A[b].to(torch.float32)

            # select queries -> [H, Q, T]
            Aq = A_b.index_select(1, query_idx)

            # all-graph keys -> [H, Q, K]
            A_qk_all = Aq.index_select(2, key_idx)
            # sink key -> [H, Q, 1]
            A_qk_sink = Aq.index_select(2, sink_pos)
            # non-sink keys -> [H, Q, K-1]
            A_qk_non = Aq.index_select(2, non_sink_pos)

            # sum over queries and keys
            den_all_graph[l] = A_qk_all.sum(dim=(1, 2)).detach().cpu()   # [H]
            sink_mass[l]     = A_qk_sink.sum(dim=(1, 2)).detach().cpu()  # [H]
            non_sink_mass[l] = A_qk_non.sum(dim=(1, 2)).detach().cpu()   # [H]

        sink_ratio = sink_mass / (den_all_graph + 1e-12)

        out.append({
            "ok": True,
            "den_all_graph": den_all_graph,   # [L,H]
            "sink_mass": sink_mass,           # [L,H]
            "non_sink_mass": non_sink_mass,   # [L,H]
            "sink_ratio": sink_ratio,         # [L,H]
            "key_idx": key_idx.detach().cpu(),
            "query_len": int(query_idx.numel()),
        })

    return out


#### Second helper function, select graph centric heads ######
def select_graph_centric_heads(
    stats: Dict[str, Any],
    *,
    den_all_graph_threshold: float = 2.0,   # graph-centric if total mass to graph tokens is >= 2
    sink_ratio_threshold: float = 0.2,      # non-sink if sink dominance is < 0.2
) -> Dict[str, Any]:
    """
    Select heads per layer that are (1) graph-centric by absolute mass and (2) not sinky.

    Expects `stats` to contain:
      - stats["den_all_graph"]: FloatTensor [B, L, H]  # sum(attn to all graph tokens) over chosen queries
      - stats["sink_ratio"]:    FloatTensor [B, L, H]  # sink_mass / (den_all_graph + eps)

    Returns:
      - masks and selected head indices per example.
    """
    den_all_graph = stats.get("den_all_graph", None)
    sink_ratio = stats.get("sink_ratio", None)

    if den_all_graph is None or sink_ratio is None:
        raise ValueError(
            "stats must contain 'den_all_graph' and 'sink_ratio'. "
            f"Got keys={list(stats.keys())}"
        )

    if den_all_graph.dim() != 3 or sink_ratio.dim() != 3:
        raise ValueError(
            f"Expected [B,L,H] for both tensors, got "
            f"den_all_graph={tuple(den_all_graph.shape)} sink_ratio={tuple(sink_ratio.shape)}"
        )

    # Condition 1: graph-centric by absolute mass to graph tokens
    mask_graph = den_all_graph >= den_all_graph_threshold  # [B,L,H] bool

    # Condition 2: not sinky (only meaningful if graph-centric)
    mask_non_sink = sink_ratio < sink_ratio_threshold      # [B,L,H] bool

    # Combined selection
    mask_selected = mask_graph & mask_non_sink             # [B,L,H] bool

    # Package indices per example for convenience
    B, L, H = mask_selected.shape
    selected_indices = []
    for b in range(B):
        per_b = []
        for l in range(L):
            hs = torch.nonzero(mask_selected[b, l], as_tuple=False).view(-1).tolist()
            per_b.append(hs)  # list of head ids selected at layer l
        selected_indices.append(per_b)

    return {
        "mask_graph": mask_graph,
        "mask_non_sink": mask_non_sink,
        "mask_selected": mask_selected,
        "selected_indices": selected_indices,
        "thresholds": {
            "den_all_graph_threshold": float(den_all_graph_threshold),
            "sink_ratio_threshold": float(sink_ratio_threshold),
        },
    }


#### Attention Redistribution for Graph Tokens ######

class GraphAttnRedistributor:
    """
    Redistributes attention from graph sink tokens to non-sink graph tokens.
    Follows VisAttnSink's VARProcessor approach adapted for graph tokens.
    
    Attributes:
        p: redistribution factor (default 0.6). Sink tokens retain p fraction,
           (1-p) is redistributed to non-sink tokens.
        sink_token_pos: 0-based index within the 5 graph tokens (default 2 = third token)
        enabled: whether redistribution is active
    """
    
    p = 0.6
    sink_token_pos = 2  # 0-based, third graph token
    enabled = False
    selected_heads = None  # Dict[layer_idx, List[head_ids]] of heads to apply redistribution
    graph_indices = None  # Tensor of absolute positions of 5 graph tokens
    
    @classmethod
    def enable(cls, p: float = 0.6, sink_token_pos: int = 2):
        """Enable attention redistribution."""
        cls.p = p
        cls.sink_token_pos = sink_token_pos
        cls.enabled = True
    
    @classmethod
    def disable(cls):
        """Disable attention redistribution."""
        cls.enabled = False
    
    @classmethod
    def set_selected_heads(cls, selected_heads: Dict[int, List[int]]):
        """
        Set which heads to apply redistribution to.
        selected_heads: Dict[layer_idx -> List[head_indices]]
        Only heads in this dict will be modified.
        """
        cls.selected_heads = selected_heads
    
    @classmethod
    def set_graph_indices(cls, graph_indices: torch.Tensor):
        """
        Set the absolute token positions of the 5 graph tokens.
        graph_indices: Tensor of shape [5] with indices in the full sequence
        """
        cls.graph_indices = graph_indices
    
    @classmethod
    def redistribute_attention(
        cls,
        attention_weights: torch.Tensor,  # [B, H, Q, T] or [H, Q, T]
        layer_idx: int,
        batch_idx: int = 0,
        is_single_head: bool = False,
    ) -> torch.Tensor:
        """
        Apply attention redistribution from sink to non-sink graph tokens.
        
        Args:
            attention_weights: Attention matrix, shape [B, H, Q, T] or [H, Q, T]
            layer_idx: Current layer index
            batch_idx: Batch index (for [B, H, Q, T] case)
            is_single_head: If True, input is [Q, T] (single head attention)
        
        Returns:
            Modified attention weights with same shape
        """
        if not cls.enabled or cls.graph_indices is None:
            return attention_weights
        
        if cls.selected_heads is None or layer_idx not in cls.selected_heads:
            return attention_weights
        
        selected_head_ids = cls.selected_heads[layer_idx]
        if len(selected_head_ids) == 0:
            return attention_weights
        
        device = attention_weights.device
        graph_indices = cls.graph_indices.to(device)
        
        # Determine shape and extract batch/head dimensions
        if is_single_head:
            # [Q, T] -> single head per query
            K = graph_indices.numel()
            sink_pos_abs = graph_indices[cls.sink_token_pos].item()
            non_sink_mask = torch.ones(K, dtype=torch.bool, device=device)
            non_sink_mask[cls.sink_token_pos] = False
            non_sink_pos_abs = graph_indices[non_sink_mask]
            
            modified = cls._redistribute_single_head(
                attention_weights.clone(),
                sink_pos_abs,
                non_sink_pos_abs,
                cls.p
            )
            return modified
        else:
            # [B, H, Q, T] or [H, Q, T]
            if attention_weights.dim() == 4:
                B, H, Q, T = attention_weights.shape
                modified = attention_weights.clone()
            else:
                # [H, Q, T]
                H, Q, T = attention_weights.shape
                modified = attention_weights.clone()
                batch_idx = None
            
            K = graph_indices.numel()
            sink_pos_abs = graph_indices[cls.sink_token_pos].item()
            non_sink_mask = torch.ones(K, dtype=torch.bool, device=device)
            non_sink_mask[cls.sink_token_pos] = False
            non_sink_pos_abs = graph_indices[non_sink_mask]
            
            # Apply redistribution to selected heads
            for head_id in selected_head_ids:
                if attention_weights.dim() == 4:
                    modified[:, head_id, :, :] = cls._redistribute_single_head(
                        modified[:, head_id, :, :],
                        sink_pos_abs,
                        non_sink_pos_abs,
                        cls.p
                    )
                else:
                    modified[head_id, :, :] = cls._redistribute_single_head(
                        modified[head_id, :, :],
                        sink_pos_abs,
                        non_sink_pos_abs,
                        cls.p
                    )
            
            return modified
    
    @classmethod
    def _redistribute_single_head(
        cls,
        attn_map: torch.Tensor,  # [Q, T] or [B, Q, T]
        sink_pos: int,
        non_sink_pos: torch.Tensor,  # [K-1]
        p: float,
    ) -> torch.Tensor:
        """
        Redistribute attention for a single head.
        
        Args:
            attn_map: Attention weights [Q, T] or [B, Q, T]
            sink_pos: Absolute position of sink token
            non_sink_pos: Tensor of positions of non-sink graph tokens
            p: Redistribution factor
        
        Returns:
            Modified attention map with same shape
        """
        # Make a copy for manipulation
        result = attn_map.clone()
        copied_map = torch.clone(result.detach())
        
        # Step 1: Reduce sink token attention by factor p
        result[:, sink_pos] *= p
        
        # Step 2: Calculate budget from sink tokens
        budget_sink = copied_map[:, sink_pos] * (1 - p)  # [Q] or [B, Q]
        
        # Step 3: Reduce non-sink graph tokens by p
        result[:, non_sink_pos] *= p
        
        # Step 4: Calculate budget from non-sink tokens
        budget_non_sink = copied_map[:, non_sink_pos].sum(dim=-1) * (1 - p)  # [Q] or [B, Q]
        
        # Step 5: Calculate redistribution ratios for non-sink tokens
        # Zero out sink for ratio calculation
        copied_map[:, sink_pos] = 0
        
        # Get total attention to non-sink graph tokens
        non_sink_total = copied_map[:, non_sink_pos].sum(dim=-1, keepdim=True)  # [Q, 1] or [B, Q, 1]
        
        # Avoid division by zero
        ratios = copied_map[:, non_sink_pos] / (non_sink_total + 1e-12)  # [Q, K-1] or [B, Q, K-1]
        
        # Step 6: Distribute combined budget proportionally
        combined_budget = budget_sink + budget_non_sink  # [Q] or [B, Q]
        
        # Reshape for broadcasting
        if attn_map.dim() == 2:
            # [Q, T] case
            combined_budget = combined_budget.unsqueeze(-1)  # [Q, 1]
        else:
            # [B, Q, T] case
            combined_budget = combined_budget.unsqueeze(-1)  # [B, Q, 1]
        
        redistribution = combined_budget * ratios  # [Q, K-1] or [B, Q, K-1]
        
        # Add redistribution to non-sink tokens
        result[:, non_sink_pos] += redistribution
        
        return result
