"""Logit-lens probe for LLaGA graph-token residuals.

Mirrors TEA-GLM/utils/logit_lens.py: per-layer hidden states for the K
graph tokens are projected through the model's final RMSNorm + lm_head
to get a top-1 vocabulary token at each (layer, graph-token) cell. We
aggregate across samples by taking the modal token per cell and the
mean top-1 probability across the samples that picked the modal token.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence

import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@torch.no_grad()
def compute_logit_lens(
    *,
    layer_graph_hidden_states: torch.Tensor,   # [L, K, D]
    final_norm,                                # model.model.norm (LlamaRMSNorm)
    lm_head,                                   # model.lm_head (Linear V x D)
    tokenizer=None,
) -> Dict[str, Any]:
    """
    Project per-layer graph-token residuals through final_norm + lm_head and
    return the top-1 vocabulary token at each (layer, graph-token) cell.

    LLaGA's compute_layerwise_graph_token_hidden_states already excludes the
    input-embedding layer, so caller passes the full [L, K, D] tensor.
    """
    device = lm_head.weight.device
    dtype = lm_head.weight.dtype
    x = layer_graph_hidden_states.to(device=device, dtype=dtype)   # [L, K, D]
    x = final_norm(x)
    logits = lm_head(x)                                            # [L, K, V]
    probs = logits.float().softmax(dim=-1)
    top_probs, top_ids = probs.max(dim=-1)                         # [L, K]
    top_ids_cpu = top_ids.detach().cpu()
    top_probs_cpu = top_probs.detach().cpu()

    top_strings = None
    if tokenizer is not None:
        L, K = top_ids_cpu.shape
        top_strings = [
            [tokenizer.convert_ids_to_tokens([int(top_ids_cpu[l, k])])[0].lstrip("▁")
             for k in range(K)]
            for l in range(L)
        ]

    return {
        "top1_token_ids": top_ids_cpu,
        "top1_probs": top_probs_cpu,
        "top1_strings": top_strings,
    }


@torch.no_grad()
def aggregate_logit_lens(
    *,
    top1_token_ids_list,    # list of [L, K] long tensors, one per sample
    top1_probs_list,        # list of [L, K] float tensors, one per sample
    tokenizer=None,
) -> Dict[str, Any]:
    """
    Aggregate per-sample logit-lens outputs into a single (L, K) summary:
      - top1_token_ids: modal argmax token id per cell
      - top1_probs:     mean of top-1 prob across samples that picked the modal token
      - agreement:      fraction of samples whose argmax equals the modal token
    """
    ids = torch.stack(top1_token_ids_list, dim=0)     # [N, L, K]
    probs = torch.stack(top1_probs_list, dim=0)       # [N, L, K]
    N, L, K = ids.shape

    modal_ids = torch.empty(L, K, dtype=torch.long)
    modal_probs = torch.empty(L, K, dtype=torch.float32)
    agreement = torch.empty(L, K, dtype=torch.float32)

    for l in range(L):
        for k in range(K):
            cell_ids = ids[:, l, k]
            cell_probs = probs[:, l, k]
            uniq, counts = cell_ids.unique(return_counts=True)
            m = uniq[int(counts.argmax().item())]
            mask = (cell_ids == m)
            modal_ids[l, k] = m
            modal_probs[l, k] = cell_probs[mask].mean()
            agreement[l, k] = mask.float().mean()

    top_strings = None
    if tokenizer is not None:
        top_strings = [
            [tokenizer.convert_ids_to_tokens([int(modal_ids[l, k])])[0].lstrip("▁")
             for k in range(K)]
            for l in range(L)
        ]

    return {
        "top1_token_ids": modal_ids,
        "top1_probs": modal_probs,
        "top1_strings": top_strings,
        "agreement": agreement,
        "num_samples": N,
    }


def plot_logit_lens_heatmap(
    *,
    top1_strings,                       # List[List[str]] of shape [L][K]
    top1_probs,                         # tensor or array of shape [L, K]
    save_path: str,
    row_indices: Optional[Sequence[int]] = None,   # subset of K rows to display
    row_labels: Optional[Sequence[str]] = None,    # labels for the displayed rows
    sink_indices: Optional[Iterable[int]] = None,  # graph-token indices to mark as sinks
    cmap: str = "YlOrRd",
    dpi: int = 600,
    max_chars: int = 8,
    layer_stride: int = 2,
) -> str:
    """Same format as TEA-GLM's heatmap. ``row_indices`` slices the K-axis
    down to the graph tokens we want to display (e.g. 0..10 for LLaGA NC).
    ``sink_indices`` (in the original K coordinates) are marked with an
    extra ``<s>`` label on the y-axis. Output is written as a PDF.
    """
    probs = top1_probs.detach().cpu().numpy() if hasattr(top1_probs, "detach") else np.asarray(top1_probs)
    L_full, K_full = probs.shape

    if row_indices is None:
        row_indices = list(range(K_full))
    else:
        row_indices = [int(k) for k in row_indices]
    K = len(row_indices)
    sink_set = {int(s) for s in (sink_indices or [])}
    if row_labels is None:
        row_labels = [
            f"<g{k}><s>" if k in sink_set else f"<g{k}>"
            for k in row_indices
        ]
    else:
        row_labels = [
            f"{lbl}<s>" if k in sink_set else lbl
            for k, lbl in zip(row_indices, row_labels)
        ]

    # Subsample layers: every `layer_stride`-th layer; always include the final layer.
    layer_indices = list(range(0, L_full, layer_stride))
    if layer_indices[-1] != L_full - 1:
        layer_indices.append(L_full - 1)
    L = len(layer_indices)
    probs = probs[layer_indices][:, row_indices]
    strings = [[top1_strings[l][k] for k in row_indices] for l in layer_indices]

    fig, ax = plt.subplots(figsize=(max(10.0, 1.4 * L), 0.75 * K + 1.0))
    im = ax.imshow(probs.T, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    for col in range(L):
        for k in range(K):
            txt = strings[col][k]
            if max_chars and len(txt) > max_chars:
                txt = txt[:max_chars]
            ax.text(
                col, k, txt,
                ha="center", va="center", fontsize=18, color="black",
                clip_on=True,
            )

    ax.set_yticks(range(K))
    ax.set_yticklabels(row_labels, fontsize=16)
    ax.set_xticks(range(L))
    ax.set_xticklabels([f"L-{l}" for l in layer_indices], rotation=0, fontsize=16)
    ax.set_xlabel("Transformer layer", fontsize=16)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Top-1 prob", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    save_path = os.fspath(save_path)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path
