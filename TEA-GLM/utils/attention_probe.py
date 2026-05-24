import os
import json
from typing import Any, Dict, List, Optional

import torch


def extract_graph_token_positions_from_is_node(
    *,
    is_node: torch.Tensor,
    expected_k: int = 5,
) -> List[torch.Tensor]:
    """
    Return graph-token positions for each sample from is_node.

    is_node: [B, T]
    returns: list of length B, each entry is [K] token positions
    """
    out = []
    for b in range(is_node.shape[0]):
        pos = torch.nonzero(is_node[b] > 0, as_tuple=False).reshape(-1).to(torch.long)
        if expected_k is not None and pos.numel() >= expected_k:
            pos = pos[:expected_k]
        out.append(pos)
    return out

def init_query_to_graph_attention_storage(
    *,
    num_graph_tokens: int = 20,
    q_max: int = 256,
    num_layers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Storage for the mean [Q, K] cross-attention matrix from query tokens (after
    the graph block) to graph tokens, averaged over layers + heads + samples.

    Query positions are aligned by RELATIVE offset from the end of the graph
    block (offset 0 = first post-graph query token). Samples with fewer than
    q_max queries contribute only to the offsets they cover; each offset's mean
    is therefore `sum_q_to_g[q] / count_q_to_g[q]`.
    """
    return {
        "sum_q_to_g": torch.zeros(q_max, num_graph_tokens, dtype=torch.float64),
        "count_q_to_g": torch.zeros(q_max, dtype=torch.long),
        # Per-layer L x K accumulator. Always lazy-initialised on the first batch
        # from `len(outputs.attentions)`, since the caller's `num_layers` may be
        # the hidden-states count (num_hidden_layers + 1) which doesn't match the
        # attention tuple length. Counted with num_valid_samples because each
        # sample contributes one full [L, K] matrix (query-averaged).
        "sum_l_to_g": None,
        "num_valid_samples": 0,
        "num_layers": num_layers,
        "num_graph_tokens": int(num_graph_tokens),
        "q_max": int(q_max),
    }


@torch.no_grad()
def update_query_to_graph_attention_storage(
    *,
    storage: Dict[str, Any],
    outputs,
    is_node: torch.Tensor,
    attention_mask: torch.Tensor,
    expected_k: int = 20,
) -> Dict[str, Any]:
    """
    Per valid sample in the batch, compute the per-layer [Q, K] query-to-graph
    attention matrix (head-averaged), mean it over layers, and accumulate the
    first q_max rows into `storage["sum_q_to_g"]` (with per-row counts).
    """
    attentions = outputs.attentions  # tuple of L tensors, each [B, H, T, T]
    if attentions is None or len(attentions) == 0:
        return storage

    num_layers = len(attentions)
    batch_size = attentions[0].shape[0]
    q_max = int(storage["q_max"])

    graph_positions_per_sample = extract_graph_token_positions_from_is_node(
        is_node=is_node,
        expected_k=expected_k,
    )

    for b in range(batch_size):
        graph_pos = graph_positions_per_sample[b]
        if graph_pos.numel() == 0:
            continue

        # Query tokens: non-padded positions strictly after the last graph token.
        last_graph = int(graph_pos.max().item())
        valid = attention_mask[b].to(torch.bool)
        is_graph_b = is_node[b].to(torch.bool)
        query_mask = valid.clone()
        query_mask[: last_graph + 1] = False
        query_mask[is_graph_b] = False
        query_pos = torch.nonzero(query_mask, as_tuple=False).reshape(-1)
        if query_pos.numel() == 0:
            continue

        # Truncate to the aggregation window (offsets 0..q_max-1).
        query_pos = query_pos[:q_max]
        Q_b = query_pos.numel()

        dev = attentions[0].device
        graph_pos_dev = graph_pos.to(dev)
        query_pos_dev = query_pos.to(dev)

        per_layer = []
        for l in range(num_layers):
            attn = attentions[l][b]                                        # [H, T, T]
            head_avg = attn.mean(dim=0)                                    # [T, T]
            q2g = head_avg.index_select(0, query_pos_dev).index_select(1, graph_pos_dev)  # [Q_b, K]
            per_layer.append(q2g)

        stacked = torch.stack(per_layer, dim=0).to(torch.float64).cpu()    # [L, Q_b, K]
        layer_avg = stacked.mean(dim=0)                                    # [Q_b, K]

        storage["sum_q_to_g"][:Q_b] += layer_avg
        storage["count_q_to_g"][:Q_b] += 1

        # Per-layer L x K (mean over Q_b for this sample) — sample-averaged in finalize.
        l_to_g = stacked.mean(dim=1)                                       # [L, K]
        if storage.get("sum_l_to_g") is None:
            storage["sum_l_to_g"] = torch.zeros(
                l_to_g.shape, dtype=torch.float64,
            )
            storage["num_layers"] = int(l_to_g.shape[0])
        storage["sum_l_to_g"] += l_to_g

        storage["num_valid_samples"] += 1

    return storage


@torch.no_grad()
def _finalize_q2g_mean(storage: Dict[str, Any]) -> torch.Tensor:
    sums = storage["sum_q_to_g"].detach().cpu().to(torch.float64)
    counts = storage["count_q_to_g"].detach().cpu().to(torch.float64)
    safe = counts.clamp(min=1).unsqueeze(-1)
    return (sums / safe).to(torch.float32)  # [q_max, K]


@torch.no_grad()
def build_query_to_graph_attention_summary(
    storage: Dict[str, Any],
    *,
    dataset_name: str = "",
) -> Dict[str, Any]:
    n = int(storage["num_valid_samples"])
    means = _finalize_q2g_mean(storage)                                    # [q_max, K]
    counts = storage["count_q_to_g"].detach().cpu().to(torch.long)
    return {
        "dataset_name": str(dataset_name),
        "num_graph_tokens": int(storage["num_graph_tokens"]),
        "q_max": int(storage["q_max"]),
        "num_valid_samples": n,
        "mean_q_to_g": means.tolist(),       # [q_max, K]
        "count_q_to_g": counts.tolist(),     # [q_max]
    }


def save_query_to_graph_attention_summary(
    *,
    storage: Dict[str, Any],
    save_path: str,
    dataset_name: str = "",
) -> str:
    summary = build_query_to_graph_attention_summary(storage, dataset_name=dataset_name)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return save_path


def plot_query_to_graph_attention_summary(
    *,
    storage: Dict[str, Any],
    save_path: str,
    dataset_name: str = "",
    dpi: int = 180,
    min_samples_per_row: int = 1,
) -> str:
    """
    Heatmap: x = graph token index, y = query token offset after graph block,
    color = mean attention weight (head-avg, layer-avg, sample-avg).

    Rows with fewer than `min_samples_per_row` contributing samples are trimmed
    from the tail, matching LLaGA's per-sample attention_to_sink heatmap style.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

    n = int(storage["num_valid_samples"])
    means = _finalize_q2g_mean(storage).numpy()                            # [q_max, K]
    counts = storage["count_q_to_g"].detach().cpu().to(torch.long).numpy()
    num_graph_tokens = int(storage["num_graph_tokens"])

    valid_rows = counts >= int(min_samples_per_row)
    if not valid_rows.any():
        raise ValueError(
            "No query offsets have enough contributing samples to plot "
            f"(min_samples_per_row={min_samples_per_row})."
        )
    # Trim to the longest contiguous prefix that meets the threshold, so the
    # y-axis runs 0..last_valid_row without gaps.
    last_valid = int(valid_rows.nonzero()[0].max()) + 1
    means = means[:last_valid]
    q_eff = means.shape[0]

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig_h = max(5.0, min(18.0, 0.08 * q_eff + 2.5))
    fig_w = max(6.0, min(16.0, 0.5 * num_graph_tokens + 4.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    im = ax.imshow(means, aspect="auto", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attention weight (head-avg, layer-avg, sample-avg)")

    title_prefix = f"{dataset_name}: " if dataset_name else ""
    ax.set_title(
        f"{title_prefix}mean query-to-graph cross attention (n={n} samples)"
    )
    ax.set_xlabel("Graph token index")
    ax.set_ylabel("Query token offset (after graph block)")

    if num_graph_tokens <= 30:
        x_step = 1
    elif num_graph_tokens <= 60:
        x_step = 2
    else:
        x_step = 5
    ax.set_xticks(list(range(0, num_graph_tokens, x_step)))

    y_step = max(1, q_eff // 20)
    ax.set_yticks(list(range(0, q_eff, y_step)))

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path