import os
import json
from typing import Any, Dict, List, Optional, Tuple

import torch

from utils.constants import DEFAULT_GRAPH_PAD_ID


def compute_graph_token_scores(
    layer_query_to_graph: torch.Tensor,
) -> torch.Tensor:
    """
    Compute one scalar attention score per graph token.

    Supported inputs:
      - [L, Q, K]: average over layers and queries -> [K]
      - [Q, K]: average over queries -> [K]
      - [K]: already token-level scores
    """
    if layer_query_to_graph.dim() == 3:
        # layer-average + query-average
        return layer_query_to_graph.mean(dim=(0, 1))
    if layer_query_to_graph.dim() == 2:
        # query-average
        return layer_query_to_graph.mean(dim=0)
    if layer_query_to_graph.dim() == 1:
        return layer_query_to_graph

    raise ValueError(
        "layer_query_to_graph must have shape [L,Q,K], [Q,K], or [K]."
    )


def remap_graph_token_scores_to_nodes(
    *,
    graphs: torch.Tensor,
    token_scores: torch.Tensor,
    key_is_pad: Optional[torch.Tensor] = None,
    graph_pad_id: int = DEFAULT_GRAPH_PAD_ID,
    reduce: str = "mean",
) -> Dict[str, torch.Tensor]:
    """
    Remap graph-token scores to original node IDs.

    Args:
      graphs: [G, Lg] or [Lg] node IDs used by graph-token slots.
      token_scores: [K] score for each graph token slot.
      key_is_pad: optional [K] bool mask (True where token slot is padded).
      reduce: "mean", "sum", or "max" when multiple token slots map to one node.

    Returns:
      {
        "token_node_ids": [K_nonpad],
        "token_scores": [K_nonpad],
        "node_ids": [N],
        "node_scores": [N],
        "node_token_counts": [N],
      }
    """
    if graphs.dim() == 1:
        graphs = graphs.unsqueeze(0)

    token_scores = token_scores.detach().to(torch.float32).reshape(-1)
    device = token_scores.device
    flat_nodes = graphs.reshape(-1).to(device=device, dtype=torch.long)

    if key_is_pad is not None:
        key_is_pad = key_is_pad.detach().to(device=device).bool().reshape(-1)
        if key_is_pad.numel() != token_scores.numel():
            raise ValueError(
                f"key_is_pad length ({key_is_pad.numel()}) != token_scores length ({token_scores.numel()})."
            )

    # Align node IDs with token_scores.
    if flat_nodes.numel() != token_scores.numel():
        if flat_nodes.numel() < token_scores.numel():
            raise ValueError(
                "graphs has fewer token slots than token_scores. "
                "This is unsupported for remapping."
            )

        # Common case when token_scores excludes pad tokens.
        nonpad_mask_from_graph = flat_nodes != graph_pad_id
        if int(nonpad_mask_from_graph.sum().item()) != token_scores.numel():
            raise ValueError(
                "Cannot align graphs and token_scores. "
                "Provide key_is_pad or ensure token_scores matches graph-token slots."
            )
        flat_nodes = flat_nodes[nonpad_mask_from_graph]
    else:
        if key_is_pad is not None:
            flat_nodes = flat_nodes[~key_is_pad]
            token_scores = token_scores[~key_is_pad]
        else:
            nonpad_mask = flat_nodes != graph_pad_id
            flat_nodes = flat_nodes[nonpad_mask]
            token_scores = token_scores[nonpad_mask]

    if flat_nodes.numel() == 0:
        raise ValueError("No non-pad graph tokens available for remapping.")

    node_ids, inverse = torch.unique(flat_nodes, sorted=True, return_inverse=True)
    num_nodes = node_ids.numel()

    node_token_counts = torch.zeros(num_nodes, dtype=torch.long, device=token_scores.device)
    node_token_counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.long))

    if reduce == "sum" or reduce == "mean":
        node_scores = torch.zeros(num_nodes, dtype=torch.float32, device=token_scores.device)
        node_scores.scatter_add_(0, inverse, token_scores)
        if reduce == "mean":
            node_scores = node_scores / (node_token_counts.to(torch.float32) + 1e-12)
    elif reduce == "max":
        node_scores = torch.full(
            (num_nodes,),
            fill_value=torch.finfo(torch.float32).min,
            dtype=torch.float32,
            device=token_scores.device,
        )
        for i in range(token_scores.numel()):
            idx = int(inverse[i].item())
            node_scores[idx] = torch.maximum(node_scores[idx], token_scores[i])
    else:
        raise ValueError("reduce must be one of: 'mean', 'sum', 'max'.")

    return {
        "token_node_ids": flat_nodes.detach().cpu(),
        "token_scores": token_scores.detach().cpu(),
        "node_ids": node_ids.detach().cpu(),
        "node_scores": node_scores.detach().cpu(),
        "node_token_counts": node_token_counts.detach().cpu(),
    }


def _build_positions(
    node_ids: torch.Tensor,
    edge_index: Optional[torch.Tensor] = None,
    seed: int = 0,
) -> Tuple[Dict[int, Tuple[float, float]], bool]:
    """
    Return {node_id: (x,y)} and whether spring-layout (networkx) was used.
    """
    node_ids_list = [int(n.item()) for n in node_ids]

    if edge_index is not None:
        try:
            import networkx as nx

            g = nx.Graph()
            g.add_nodes_from(node_ids_list)
            edge_index = edge_index.detach().cpu().to(torch.long)
            if edge_index.dim() != 2 or edge_index.shape[0] != 2:
                raise ValueError("edge_index must have shape [2, E].")

            valid_nodes = set(node_ids_list)
            for u, v in edge_index.t().tolist():
                if u in valid_nodes and v in valid_nodes and u != v:
                    g.add_edge(int(u), int(v))

            pos = nx.spring_layout(g, seed=seed)
            pos = {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}
            return pos, True
        except Exception:
            # Fall back to simple deterministic layout below.
            pass

    n = len(node_ids_list)
    if n == 1:
        return {node_ids_list[0]: (0.0, 0.0)}, False

    # Deterministic circular fallback.
    theta = torch.linspace(0, 2 * torch.pi, steps=n + 1)[:-1]
    pos = {
        node_ids_list[i]: (float(torch.cos(theta[i]).item()), float(torch.sin(theta[i]).item()))
        for i in range(n)
    }
    return pos, False


def plot_node_attention_scores(
    *,
    node_ids: torch.Tensor,
    node_scores: torch.Tensor,
    save_path: str,
    edge_index: Optional[torch.Tensor] = None,
    title: str = "Node Attention (Layer-Avg)",
    cmap: str = "viridis",
    node_size: int = 480,
    show_node_labels: bool = True,
    seed: int = 0,
) -> str:
    """
    Plot original nodes colored by attention score.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it to save node remap plots."
        ) from e

    node_ids = node_ids.detach().cpu().to(torch.long)
    node_scores = node_scores.detach().cpu().to(torch.float32)

    if node_ids.numel() != node_scores.numel():
        raise ValueError("node_ids and node_scores must have the same length.")
    if node_ids.numel() == 0:
        raise ValueError("No nodes to plot.")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    pos, used_spring = _build_positions(node_ids=node_ids, edge_index=edge_index, seed=seed)
    x = torch.tensor([pos[int(n.item())][0] for n in node_ids], dtype=torch.float32)
    y = torch.tensor([pos[int(n.item())][1] for n in node_ids], dtype=torch.float32)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    # Draw edges first (if provided).
    if edge_index is not None:
        edge_index = edge_index.detach().cpu().to(torch.long)
        node_set = set(int(n.item()) for n in node_ids)
        for u, v in edge_index.t().tolist():
            if u in node_set and v in node_set and u != v:
                x1, y1 = pos[int(u)]
                x2, y2 = pos[int(v)]
                ax.plot([x1, x2], [y1, y2], color="#BBBBBB", linewidth=0.8, zorder=1)

    sc = ax.scatter(
        x.numpy(),
        y.numpy(),
        c=node_scores.numpy(),
        cmap=cmap,
        s=node_size,
        edgecolors="black",
        linewidths=0.8,
        zorder=2,
    )

    if show_node_labels:
        y_span = float((y.max() - y.min()).item()) if y.numel() > 1 else 1.0
        y_offset = 0.02 * max(y_span, 1.0)
        for i, nid in enumerate(node_ids.tolist()):
            ax.text(
                float(x[i].item()),
                float(y[i].item()) + y_offset,
                str(int(nid)),
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                zorder=3,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.8,
                },
            )

    layout_tag = "spring" if used_spring else "fallback"
    ax.set_title(f"{title} | layout={layout_tag}")
    ax.axis("off")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Attention score (layer-avg, query-avg)")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return save_path


def remap_and_plot_node_attention(
    *,
    layer_query_to_graph: torch.Tensor,
    graphs: torch.Tensor,
    save_path: str,
    key_is_pad: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None,
    graph_pad_id: int = DEFAULT_GRAPH_PAD_ID,
    reduce: str = "mean",
    title: str = "Node Attention (Layer-Avg)",
    cmap: str = "viridis",
    node_size: int = 480,
    show_node_labels: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    End-to-end helper:
      1) Compute graph-token scores from attention
      2) Remap token scores to original nodes
      3) Plot nodes colored by remapped scores

    Returns remap data plus saved plot path.
    """
    token_scores = compute_graph_token_scores(layer_query_to_graph)

    remap = remap_graph_token_scores_to_nodes(
        graphs=graphs,
        token_scores=token_scores,
        key_is_pad=key_is_pad,
        graph_pad_id=graph_pad_id,
        reduce=reduce,
    )

    out_path = plot_node_attention_scores(
        node_ids=remap["node_ids"],
        node_scores=remap["node_scores"],
        save_path=save_path,
        edge_index=edge_index,
        title=title,
        cmap=cmap,
        node_size=node_size,
        show_node_labels=show_node_labels,
        seed=seed,
    )

    remap["plot_path"] = out_path
    remap["token_scores_all_layers_avg"] = token_scores.detach().cpu()
    return remap


def _edge_index_to_adjacency(
    edge_index: Optional[torch.Tensor],
) -> Optional[Dict[int, set]]:
    if edge_index is None:
        return None

    e = edge_index.detach().cpu().to(torch.long)
    if e.dim() != 2 or e.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    adjacency: Dict[int, set] = {}
    for u, v in e.t().tolist():
        ui = int(u)
        vi = int(v)
        if ui == vi:
            continue
        adjacency.setdefault(ui, set()).add(vi)
        adjacency.setdefault(vi, set()).add(ui)
    return adjacency


def _classify_node_hop(
    *,
    center_node: Optional[int],
    target_node: Optional[int],
    adjacency: Optional[Dict[int, set]],
) -> str:
    if center_node is None or target_node is None:
        return "rest"

    if target_node == center_node:
        return "center"

    if adjacency is None:
        return "rest"

    one_hop = adjacency.get(center_node, set())
    if target_node in one_hop:
        return "one_hop"

    two_hop = set()
    for n1 in one_hop:
        two_hop.update(adjacency.get(n1, set()))
    two_hop.discard(center_node)

    if target_node in two_hop:
        return "two_hop"

    return "rest"


def build_highest_attention_graph_token_record(
    *,
    sample_id: Any,
    graphs: torch.Tensor,
    layer_query_to_graph: Optional[torch.Tensor] = None,
    key_idx: Optional[torch.Tensor] = None,
    key_is_pad: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None,
    graph_pad_id: int = DEFAULT_GRAPH_PAD_ID,
    remap_plot_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build one per-sample record for the highest-attention non-pad graph token.

    The attention scalar is averaged across:
      - layers
      - query tokens
    """
    if graphs.dim() == 1:
        graphs = graphs.unsqueeze(0)

    graphs_cpu = graphs.detach().cpu().to(torch.long)
    flat_nodes = graphs_cpu.reshape(-1)
    G, Lg = graphs_cpu.shape
    num_graph_tokens = int(flat_nodes.numel())
    num_nonpad_graph_tokens = int((flat_nodes != graph_pad_id).sum().item())

    record: Dict[str, Any] = {
        "sample_id": sample_id,
        "valid": False,
        "reason": None,
        "num_graph_tokens": num_graph_tokens,
        "num_nonpad_graph_tokens": num_nonpad_graph_tokens,
        "highest_token_index": None,
        "highest_prompt_position": None,
        "highest_block_index": None,
        "highest_node_id": None,
        "highest_attention": None,
        "highest_hop_category": "rest",
        "highest_node_degree": None,
        "center_node_id": None,
        "center_node_degree": None,
        "degree_gap_highest_minus_center": None,
        "remap_plot_path": remap_plot_path,
    }

    if layer_query_to_graph is None:
        record["reason"] = "missing_attention_inputs"
        return record

    token_scores = compute_graph_token_scores(
        layer_query_to_graph
    ).detach().cpu().to(torch.float32).reshape(-1)

    if key_is_pad is not None:
        key_is_pad_cpu = key_is_pad.detach().cpu().bool().reshape(-1)
        if key_is_pad_cpu.numel() != token_scores.numel():
            record["reason"] = "key_is_pad_length_mismatch"
            return record
    else:
        if flat_nodes.numel() != token_scores.numel():
            record["reason"] = "graphs_length_mismatch_without_key_is_pad"
            return record
        key_is_pad_cpu = flat_nodes == graph_pad_id

    if flat_nodes.numel() != token_scores.numel():
        record["reason"] = "graphs_and_scores_not_aligned"
        return record

    nonpad_token_indices = torch.nonzero(~key_is_pad_cpu, as_tuple=False).reshape(-1)
    if nonpad_token_indices.numel() == 0:
        record["reason"] = "no_nonpad_graph_tokens"
        return record

    local_max = int(token_scores[nonpad_token_indices].argmax().item())
    highest_token_index = int(nonpad_token_indices[local_max].item())
    highest_block_index = int(highest_token_index // Lg)
    highest_node_id = int(flat_nodes[highest_token_index].item())
    highest_attention = float(token_scores[highest_token_index].item())

    highest_prompt_position = None
    if key_idx is not None:
        key_idx_cpu = key_idx.detach().cpu().to(torch.long).reshape(-1)
        if key_idx_cpu.numel() == token_scores.numel():
            highest_prompt_position = int(key_idx_cpu[highest_token_index].item())

    center_node_id = None
    if 0 <= highest_block_index < G:
        block_nodes = graphs_cpu[highest_block_index]
        nonpad_block_nodes = block_nodes[block_nodes != graph_pad_id]
        if nonpad_block_nodes.numel() > 0:
            center_node_id = int(nonpad_block_nodes[0].item())

    adjacency = _edge_index_to_adjacency(edge_index)
    highest_hop_category = _classify_node_hop(
        center_node=center_node_id,
        target_node=highest_node_id,
        adjacency=adjacency,
    )

    highest_node_degree = None
    center_node_degree = None
    if adjacency is not None:
        highest_node_degree = len(adjacency.get(highest_node_id, set()))
        if center_node_id is not None:
            center_node_degree = len(adjacency.get(center_node_id, set()))

    record["valid"] = True
    record["reason"] = "ok"
    record["highest_token_index"] = highest_token_index
    record["highest_prompt_position"] = highest_prompt_position
    record["highest_block_index"] = highest_block_index
    record["highest_node_id"] = highest_node_id
    record["highest_attention"] = highest_attention
    record["highest_hop_category"] = highest_hop_category
    record["highest_node_degree"] = highest_node_degree
    record["center_node_id"] = center_node_id
    record["center_node_degree"] = center_node_degree
    if highest_node_degree is not None and center_node_degree is not None:
        record["degree_gap_highest_minus_center"] = (
            highest_node_degree - center_node_degree
        )
    return record


def append_highest_attention_graph_token_record(
    *,
    record: Dict[str, Any],
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def aggregate_highest_attention_graph_token_from_jsonl(
    *,
    records_path: str,
) -> Dict[str, Any]:
    if not os.path.exists(records_path):
        raise FileNotFoundError(f"records_path does not exist: {records_path}")

    hop_counts = {
        "center": 0,
        "one_hop": 0,
        "two_hop": 0,
        "rest": 0,
    }
    highest_node_degree: List[int] = []
    highest_attention: List[float] = []
    center_node_degree: List[int] = []
    degree_gap_highest_minus_center: List[int] = []
    sample_ids: List[Any] = []
    per_sample: List[Dict[str, Any]] = []
    n_samples = 0
    n_valid_samples = 0

    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_samples += 1
            rec = json.loads(line)
            per_sample.append(rec)

            if not bool(rec.get("valid", False)):
                continue

            n_valid_samples += 1
            hop = rec.get("highest_hop_category", "rest")
            if hop not in hop_counts:
                hop = "rest"
            hop_counts[hop] += 1

            hd = rec.get("highest_node_degree")
            ha = rec.get("highest_attention")
            cd = rec.get("center_node_degree")
            dg = rec.get("degree_gap_highest_minus_center")

            if hd is not None and ha is not None:
                sample_ids.append(rec.get("sample_id"))
                highest_node_degree.append(int(hd))
                highest_attention.append(float(ha))
            if cd is not None:
                center_node_degree.append(int(cd))
            if dg is not None:
                degree_gap_highest_minus_center.append(int(dg))

    def _mean(values: List[int]) -> float:
        if not values:
            return 0.0
        return float(sum(values)) / float(len(values))

    hop_percentages = {
        k: (100.0 * float(v) / float(n_valid_samples) if n_valid_samples > 0 else 0.0)
        for k, v in hop_counts.items()
    }

    return {
        "n_samples": n_samples,
        "n_valid_samples": n_valid_samples,
        "hop_counts": hop_counts,
        "hop_percentages": hop_percentages,
        "highest_node_degree": highest_node_degree,
        "highest_attention": highest_attention,
        "center_node_degree": center_node_degree,
        "degree_gap_highest_minus_center": degree_gap_highest_minus_center,
        "avg_highest_node_degree": _mean(highest_node_degree),
        "avg_center_node_degree": _mean(center_node_degree),
        "avg_degree_gap_highest_minus_center": _mean(degree_gap_highest_minus_center),
        "sample_ids": sample_ids,
        "per_sample": per_sample,
        "records_path": records_path,
    }


def plot_highest_attention_degree_vs_attention(
    *,
    aggregated: Dict[str, Any],
    save_path: str,
    dpi: int = 180,
    point_alpha: float = 0.35,
    point_size: int = 24,
) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it to save degree-attention plots."
        ) from e

    x = torch.tensor(aggregated["highest_node_degree"], dtype=torch.float32)
    y = torch.tensor(aggregated["highest_attention"], dtype=torch.float32)

    if x.numel() == 0 or y.numel() == 0:
        raise ValueError("No valid samples available to plot highest-node degree vs attention.")
    if x.numel() != y.numel():
        raise ValueError("highest_node_degree and highest_attention must have the same length.")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=dpi)
    ax.scatter(
        x.numpy(),
        y.numpy(),
        alpha=point_alpha,
        s=point_size,
        edgecolors="none",
        label="sample",
    )

    unique_x = torch.unique(x, sorted=True)
    mean_y = []
    for cur_x in unique_x:
        mask = x == cur_x
        mean_y.append(y[mask].mean())
    mean_y_t = torch.stack(mean_y, dim=0)

    ax.plot(
        unique_x.numpy(),
        mean_y_t.numpy(),
        color="crimson",
        linewidth=2.0,
        label="mean",
    )

    ax.set_title(
        "Highest-Attention Graph Token: Degree vs Attention "
        f'(valid={aggregated["n_valid_samples"]}/{aggregated["n_samples"]})'
    )
    ax.set_xlabel("Degree of highest-attention node")
    ax.set_ylabel("Attention weight of highest-attention graph token")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path
