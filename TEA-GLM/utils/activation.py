import os
import json
from typing import Any, Dict, List, Optional

import torch


@torch.no_grad()
def rmsnorm(hidden_states: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = hidden_states.to(torch.float32)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps)


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


@torch.no_grad()
def compute_layerwise_graph_token_hidden_states(
    *,
    hidden_states,
    is_node: torch.Tensor,
    expected_k: int = 5,
) -> Dict[str, Any]:
    """
    Extract raw hidden states for the graph tokens across all layers.

    hidden_states:
      tuple/list of layer tensors, each [B, T, D]

    returns per sample:
      - layer_graph_hidden_states: [L, K, D]
      - graph_token_positions: [K]
      - valid: bool
    """
    num_layers = len(hidden_states)
    batch_size = hidden_states[0].shape[0]

    graph_positions_per_sample = extract_graph_token_positions_from_is_node(
        is_node=is_node,
        expected_k=expected_k,
    )

    valid = []
    layer_graph_hidden_states_list = []
    graph_token_positions_list = []

    for b in range(batch_size):
        pos = graph_positions_per_sample[b]
        if pos.numel() == 0:
            valid.append(False)
            layer_graph_hidden_states_list.append(None)
            graph_token_positions_list.append(None)
            continue

        layer_states = []
        for layer_idx in range(num_layers):
            hs_b = hidden_states[layer_idx][b].detach().to(torch.float32)   # [T, D]
            graph_hs = hs_b.index_select(0, pos)                            # [K, D]
            layer_states.append(graph_hs)

        layer_graph_hidden_states = torch.stack(layer_states, dim=0)        # [L, K, D]

        valid.append(True)
        layer_graph_hidden_states_list.append(layer_graph_hidden_states)
        graph_token_positions_list.append(pos.detach().cpu())

    return {
        "valid": valid,
        "layer_graph_hidden_states": layer_graph_hidden_states_list,
        "graph_token_positions": graph_token_positions_list,
    }


@torch.no_grad()
def detect_sink_tokens(
    *,
    layer_graph_hidden_states: torch.Tensor,   # [L, K, D]
    graph_token_positions: Optional[torch.Tensor] = None,
    sink_dims: List[int] = [1512, 2533],
    threshold: float = 20.0,
) -> Dict[str, Any]:
    """
    TEA-GLM sink-token detection:
      1. RMSNorm over hidden dimension
      2. select second-to-last layer
      3. keep sink dims
      4. max over sink dims
      5. threshold
    """
    x = layer_graph_hidden_states.detach().to(torch.float32).cpu()   # [L, K, D]
    x = rmsnorm(x)                                                   # [L, K, D]

    num_layers, num_tokens, hidden_dim = x.shape
    sink_dims = [int(d) for d in sink_dims if 0 <= int(d) < hidden_dim]

    if len(sink_dims) == 0:
        return {
            "sink_dims": [],
            "threshold": float(threshold),
            "num_layers": int(num_layers),
            "num_graph_tokens": int(num_tokens),
            "layer_token_scores": torch.zeros(num_tokens, dtype=torch.float32),
            "sink_token_indices": [],
            "graph_token_positions": [],
        }

    if num_layers < 2:
        raise ValueError(f"Expected at least 2 layers, but got {num_layers}.")

    # Select only the second-to-last layer: [K, D]
    x_second_last = x[-2]

    # Keep only sink dims: [K, len(sink_dims)]
    sink_vals = x_second_last[:, sink_dims]

    # Max over sink dims: [K]
    layer_token_scores = sink_vals.amax(dim=-1)

    sink_mask = layer_token_scores > threshold
    sink_token_indices = torch.nonzero(sink_mask, as_tuple=False).reshape(-1)

    # Top-2 sink tokens by score among those above threshold.
    # Empty list if no tokens cross the threshold; 1-element list if only one does.
    if sink_token_indices.numel() > 0:
        sink_scores_above = layer_token_scores[sink_token_indices]
        order = torch.argsort(sink_scores_above, descending=True)
        top2_local = sink_token_indices[order][:2]
        top2_scores = sink_scores_above[order][:2].tolist()
    else:
        top2_local = torch.empty(0, dtype=torch.long)
        top2_scores = []

    if graph_token_positions is not None:
        graph_token_positions = graph_token_positions.detach().cpu().to(torch.long).reshape(-1)
        sink_graph_positions = graph_token_positions[sink_token_indices].tolist()
        top2_graph_positions = graph_token_positions[top2_local].tolist()
    else:
        sink_graph_positions = sink_token_indices.tolist()
        top2_graph_positions = top2_local.tolist()

    return {
        "sink_dims": sink_dims,
        "threshold": float(threshold),
        "num_layers": int(num_layers),
        "num_graph_tokens": int(num_tokens),
        "layer_token_scores": layer_token_scores,   # [K], score from second-to-last layer only
        "sink_token_indices": sink_token_indices.tolist(),
        "graph_token_positions": sink_graph_positions,
        "top2_sink_token_indices": top2_local.tolist(),
        "top2_sink_scores": top2_scores,
        "top2_graph_token_positions": top2_graph_positions,
    }


def _resolve_layer_index(layer_index: int, num_layers: int) -> int:
    layer_index = int(layer_index)
    if layer_index < 0:
        layer_index += num_layers
    if layer_index < 0 or layer_index >= num_layers:
        raise ValueError(
            f"layer_index={layer_index} is out of range for num_layers={num_layers}."
        )
    return layer_index


@torch.no_grad()
def summarize_graph_feature_dimensions(
    *,
    layer_graph_hidden_states: torch.Tensor,   # [L, K, D]
    threshold: float = 0.0,
    layer_index: Optional[int] = -2,
    token_reduce: str = "mean",
    layer_reduce: str = "mean",
    apply_rmsnorm: bool = True,
    use_abs: bool = False,
) -> Dict[str, Any]:
    """
    Summarize per-dimension graph-token activations for one sample.

    Returns `per_dim_scores` of shape [D] — the reduced activation per dim at
    the selected layer, after optional RMSNorm. Callers that only need the
    mean-activation aggregation can ignore `qualified_dim_mask` and `threshold`.
    """
    x = layer_graph_hidden_states.detach().to(torch.float32).cpu()
    if x.dim() != 3:
        raise ValueError(
            "layer_graph_hidden_states must have shape [L, K, D], "
            f"but got {tuple(x.shape)}."
        )

    if apply_rmsnorm:
        x = rmsnorm(x)
    if use_abs:
        x = x.abs()

    num_layers, num_tokens, hidden_dim = x.shape

    if layer_index is not None:
        selected_layer_index = _resolve_layer_index(int(layer_index), num_layers)
        x = x[selected_layer_index:selected_layer_index + 1]
        selected_layer_indices = [selected_layer_index]
    else:
        selected_layer_indices = list(range(num_layers))

    if token_reduce == "max":
        per_layer_dim_scores = x.amax(dim=1)          # [L_sel, D]
    elif token_reduce == "mean":
        per_layer_dim_scores = x.mean(dim=1)          # [L_sel, D]
    else:
        raise ValueError("token_reduce must be one of: 'max', 'mean'.")

    if layer_reduce == "max":
        per_dim_scores = per_layer_dim_scores.amax(dim=0)   # [D]
    elif layer_reduce == "mean":
        per_dim_scores = per_layer_dim_scores.mean(dim=0)   # [D]
    else:
        raise ValueError("layer_reduce must be one of: 'max', 'mean'.")

    qualified_dim_mask = per_dim_scores > float(threshold)
    qualified_dim_indices = torch.nonzero(
        qualified_dim_mask,
        as_tuple=False,
    ).reshape(-1)

    return {
        "threshold": float(threshold),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "num_graph_tokens": int(num_tokens),
        "selected_layer_indices": selected_layer_indices,
        "token_reduce": str(token_reduce),
        "layer_reduce": str(layer_reduce),
        "apply_rmsnorm": bool(apply_rmsnorm),
        "use_abs": bool(use_abs),
        "per_dim_scores": per_dim_scores,
        "qualified_dim_mask": qualified_dim_mask,
        "qualified_dim_indices": qualified_dim_indices.tolist(),
        "num_qualified_dims": int(qualified_dim_indices.numel()),
    }


def init_graph_feature_mean_storage(
    *,
    hidden_dim: int = 4096,
    layer_index: Optional[int] = -2,
    token_reduce: str = "mean",
    apply_rmsnorm: bool = True,
    use_abs: bool = False,
) -> Dict[str, Any]:
    """
    Storage for identifying sink dimensions via average activation across
    all test samples (graph tokens at one selected layer).
    """
    return {
        "sum_scores": torch.zeros(hidden_dim, dtype=torch.float64),
        "num_valid_samples": 0,
        "layer_index": layer_index,
        "token_reduce": str(token_reduce),
        "apply_rmsnorm": bool(apply_rmsnorm),
        "use_abs": bool(use_abs),
    }


def update_graph_feature_mean_storage(
    *,
    storage: Dict[str, Any],
    per_dim_scores: torch.Tensor,
) -> Dict[str, Any]:
    scores = per_dim_scores.detach().to(torch.float64).cpu().reshape(-1)
    sums = storage["sum_scores"]

    if scores.numel() != sums.numel():
        raise ValueError(
            "per_dim_scores has incompatible shape: "
            f"expected {sums.numel()} dims, got {scores.numel()}."
        )

    storage["num_valid_samples"] += 1
    storage["sum_scores"] += scores
    return storage


@torch.no_grad()
def build_graph_feature_mean_summary(
    storage: Dict[str, Any],
    *,
    dataset_name: str = "",
    topk_dims: int = 3,
) -> Dict[str, Any]:
    sums = storage["sum_scores"].detach().cpu().to(torch.float64)
    hidden_dim = int(sums.numel())
    n = int(storage["num_valid_samples"])
    means = (sums / float(max(1, n))).to(torch.float32)

    top_dims: List[int] = []
    top_means: List[float] = []
    if topk_dims > 0 and hidden_dim > 0:
        k = min(int(topk_dims), hidden_dim)
        topk = torch.topk(means, k=k, largest=True)
        top_dims = [int(d) for d in topk.indices.tolist()]
        top_means = [float(v) for v in topk.values.tolist()]

    return {
        "dataset_name": str(dataset_name),
        "hidden_dim": hidden_dim,
        "layer_index": storage.get("layer_index", None),
        "token_reduce": str(storage.get("token_reduce", "mean")),
        "apply_rmsnorm": bool(storage.get("apply_rmsnorm", True)),
        "use_abs": bool(storage.get("use_abs", False)),
        "num_valid_samples": n,
        "mean_scores": means.tolist(),
        "top_dims": top_dims,
        "top_means": top_means,
    }


def save_graph_feature_mean_summary(
    *,
    storage: Dict[str, Any],
    save_path: str,
    dataset_name: str = "",
    topk_dims: int = 3,
) -> str:
    summary = build_graph_feature_mean_summary(
        storage,
        dataset_name=dataset_name,
        topk_dims=topk_dims,
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return save_path


def load_graph_feature_mean_summary(summary_path: str) -> Dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    return {
        "dataset_name": str(summary.get("dataset_name", "")),
        "hidden_dim": int(summary["hidden_dim"]),
        "layer_index": summary.get("layer_index", None),
        "token_reduce": str(summary.get("token_reduce", "mean")),
        "apply_rmsnorm": bool(summary.get("apply_rmsnorm", True)),
        "use_abs": bool(summary.get("use_abs", False)),
        "num_valid_samples": int(summary.get("num_valid_samples", 0)),
        "mean_scores": [float(v) for v in summary.get("mean_scores", [])],
        "top_dims": [int(dim_idx) for dim_idx in summary.get("top_dims", [])],
        "top_means": [float(v) for v in summary.get("top_means", [])],
    }


def plot_graph_feature_mean_summary(
    *,
    save_path: str,
    storage: Optional[Dict[str, Any]] = None,
    summary_path: Optional[str] = None,
    dataset_name: str = "",
    highlight_dims: Optional[List[int]] = None,
    dpi: int = 180,
    annotate_top_n: int = 3,
    annotate_offset: int = 2,
) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

    if storage is None and summary_path is None:
        raise ValueError("Either storage or summary_path must be provided.")

    if summary_path is not None:
        summary = load_graph_feature_mean_summary(summary_path)
        means = torch.tensor(summary["mean_scores"], dtype=torch.float32)
        layer_index = summary.get("layer_index", None)
        num_valid_samples = int(summary["num_valid_samples"])
        if dataset_name == "":
            dataset_name = str(summary.get("dataset_name", ""))
    else:
        n = int(storage["num_valid_samples"])
        sums = storage["sum_scores"].detach().cpu().to(torch.float64)
        means = (sums / float(max(1, n))).to(torch.float32)
        layer_index = storage.get("layer_index", None)
        num_valid_samples = n

    hidden_dim = int(means.numel())
    x = torch.arange(hidden_dim, dtype=torch.long)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)
    ax.plot(x.tolist(), means.tolist(), linewidth=1.0, color="tab:blue")

    if highlight_dims:
        for dim_idx in sorted({int(dim_idx) for dim_idx in highlight_dims}):
            if 0 <= dim_idx < hidden_dim:
                ax.axvline(dim_idx, color="tab:red", linestyle="--", alpha=0.25)

    title_prefix = f"{dataset_name}: " if dataset_name else ""
    layer_label = "all layers" if layer_index is None else f"layer {layer_index}"
    ax.set_title(
        f"{title_prefix}mean graph-token activation per hidden dimension\n"
        f"{layer_label}, n={num_valid_samples}"
    )
    ax.set_xlabel("Hidden dimension index")
    ax.set_ylabel("Mean activation (averaged over all test samples)")
    ax.grid(True, axis="y", alpha=0.25)

    if annotate_top_n > 0 and hidden_dim > 0:
        k = min(int(annotate_top_n), hidden_dim)
        topk = torch.topk(means, k=k, largest=True)
        for dim_idx, val in zip(topk.indices.tolist(), topk.values.tolist()):
            ax.annotate(
                str(dim_idx),
                xy=(dim_idx, val),
                xytext=(0, annotate_offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="tab:red",
            )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


######### Layerwise top-k sink-dimension aggregator + plot
# Mirrors LLaGA's plot_topdims_mean_activation_curve. The TEA-GLM
# graph-feature aggregator above pre-reduces over layers/tokens, so it cannot
# render the two-view (single-layer + layer-averaged) curve. This aggregator
# keeps a per-layer [L, D] sum across all graph tokens of every sample.

def init_layerwise_topdims_storage(
    *,
    num_layers: int,
    hidden_dim: int,
) -> Dict[str, Any]:
    return {
        "sum_all": torch.zeros((num_layers, hidden_dim), dtype=torch.float64),          # sum of |RMSNorm(x)|
        "sum_all_signed": torch.zeros((num_layers, hidden_dim), dtype=torch.float64),   # sum of RMSNorm(x)
        "sum_all_raw": torch.zeros((num_layers, hidden_dim), dtype=torch.float64),      # sum of x
        "count_all": torch.zeros((num_layers,), dtype=torch.long),
        "num_valid_samples": 0,
    }


@torch.no_grad()
def update_layerwise_topdims_storage(
    *,
    storage: Dict[str, Any],
    layer_graph_hidden_states: torch.Tensor,   # [L, K, D]
    token_indices: Optional[torch.Tensor] = None,  # subset of K (e.g., sink-only positions)
    rmsnorm_eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    Aggregate per-sample graph-token activations into per-layer [L, D] sums in
    three views: |RMSNorm(x)|, signed RMSNorm(x), and raw x.

    If ``token_indices`` is provided, only those graph-token positions are
    aggregated (e.g., sink-only). RMSNorm is still computed per selected token
    over the hidden dim, matching LLaGA semantics. With ``token_indices=None``
    all K graph-token slots are aggregated together.
    """
    if layer_graph_hidden_states.dim() != 3:
        raise ValueError("layer_graph_hidden_states must have shape [L, K, D].")

    x = layer_graph_hidden_states.detach().to(torch.float32).cpu()
    num_layers, num_tokens, hidden_dim = x.shape

    if storage["sum_all"].shape != (num_layers, hidden_dim):
        raise ValueError(
            "layerwise_topdims_storage shape mismatch: initialise it with the "
            "model's num_layers and hidden_dim."
        )

    if token_indices is not None:
        token_indices = token_indices.detach().to(torch.long).cpu().reshape(-1)
        if token_indices.numel() == 0:
            return storage
        if int(token_indices.min().item()) < 0 or int(token_indices.max().item()) >= num_tokens:
            raise ValueError(
                f"token_indices out of range for K={num_tokens}: "
                f"min={int(token_indices.min().item())}, max={int(token_indices.max().item())}."
            )
        selected = x.index_select(1, token_indices)
    else:
        selected = x

    n_selected = selected.shape[1]
    normed = rmsnorm(selected, eps=rmsnorm_eps)
    storage["sum_all"] += normed.abs().sum(dim=1).to(torch.float64)
    storage["sum_all_signed"] += normed.sum(dim=1).to(torch.float64)
    storage["sum_all_raw"] += selected.sum(dim=1).to(torch.float64)
    storage["count_all"] += int(n_selected)
    storage["num_valid_samples"] += 1
    return storage


def finalize_layerwise_topdims_storage(
    storage: Dict[str, Any],
    *,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    cnt_f = storage["count_all"].to(torch.float64).unsqueeze(-1)
    mean_all = (storage["sum_all"] / (cnt_f + eps)).to(torch.float32)
    mean_all_signed = (storage["sum_all_signed"] / (cnt_f + eps)).to(torch.float32)
    mean_all_raw = (storage["sum_all_raw"] / (cnt_f + eps)).to(torch.float32)
    return {
        "num_valid_samples": int(storage["num_valid_samples"]),
        "mean_all": mean_all,                  # [L, D]
        "mean_all_signed": mean_all_signed,    # [L, D]
        "mean_all_raw": mean_all_raw,          # [L, D]
        "count_all": storage["count_all"].clone(),
    }


def plot_topdims_mean_activation_curve(
    *,
    aggregated: Dict[str, Any],
    save_path: str,
    dpi: int = 180,
    sink_threshold: float = 5.0,
    layer_index: int = -2,
    use_abs: bool = True,
) -> tuple:
    """
    Plot the mean RMSNorm(activation) of graph tokens averaged across tokens
    and samples, in two views:
      1) at a single transformer layer (default: second-to-last) — saved at
         ``save_path``.
      2) averaged across all transformer layers — saved at the same directory
         with ``_layeravg`` inserted before the extension.

    When ``use_abs=True`` (default), the absolute value of the RMSNorm output
    is averaged. When ``use_abs=False``, the signed RMSNorm output is averaged
    so positive/negative contributions can cancel.

    Sink dimensions are those whose curve value (or |value| in signed mode)
    exceeds ``sink_threshold``. Their indices are marked on the x-axis as
    bold-italic red labels. Returns
    ``(layer_save_path, avg_save_path, sink_dims_layer)`` where the last is
    the list of sink dims found in the layer-specific view.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

    key = "mean_all" if use_abs else "mean_all_signed"
    if key not in aggregated:
        raise KeyError(
            f"aggregated is missing '{key}'. Build it with "
            f"finalize_layerwise_topdims_storage first."
        )
    mean_all = aggregated[key].detach().cpu().to(torch.float32)   # [L, D]
    num_layers = mean_all.shape[0]
    resolved_layer = layer_index if layer_index >= 0 else num_layers + layer_index
    if not (0 <= resolved_layer < num_layers):
        raise ValueError(
            f"layer_index={layer_index} is out of range for {num_layers} layers."
        )

    mean_per_dim_layer = mean_all[resolved_layer]   # [D]
    mean_per_dim_avg = mean_all.mean(dim=0)         # [D]
    hidden_dim = mean_per_dim_layer.numel()

    base, ext = os.path.splitext(save_path)
    avg_save_path = f"{base}_layeravg{ext}"

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    y_label = "Sink Magnitude"

    def _resolve_sink_dims(curve: torch.Tensor) -> List[int]:
        rank_signal = curve if use_abs else curve.abs()
        return torch.nonzero(
            rank_signal > float(sink_threshold), as_tuple=False,
        ).reshape(-1).tolist()

    def _render(curve: torch.Tensor, out_path: str, title: str) -> List[int]:
        fig, ax = plt.subplots(figsize=(12, 4), dpi=dpi)
        ax.plot(list(range(hidden_dim)), curve.tolist(), linewidth=1.0, color="#1f77b4")
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
        if use_abs:
            ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.set_xlabel("Embedding dimension")
        ax.set_ylabel(y_label)
        ax.grid(True, axis="y", alpha=0.25)

        sink_dims_curve = _resolve_sink_dims(curve)
        for d in sink_dims_curve:
            ax.text(
                d, -0.04, str(int(d)),
                transform=ax.get_xaxis_transform(),
                color="#d62728",
                fontweight="bold",
                fontstyle="italic",
                fontsize=9,
                ha="center",
                va="top",
                clip_on=False,
            )

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return sink_dims_curve

    sink_dims_layer = _render(
        curve=mean_per_dim_layer,
        out_path=save_path,
        title="Average Graph Sink Token Activation Magnitude",
    )
    _render(
        curve=mean_per_dim_avg,
        out_path=avg_save_path,
        title="Average Graph Sink Token Activation Magnitude",
    )
    return save_path, avg_save_path, sink_dims_layer


def init_sink_token_histogram_storage(
    *,
    num_graph_tokens: int = 5,
) -> Dict[str, Any]:
    return {
        "counts": torch.zeros(num_graph_tokens, dtype=torch.long),
        "num_samples": 0,
        "num_valid_samples": 0,
        "num_samples_with_sink": 0,
    }


def update_sink_storage(
    *,
    storage: Dict[str, Any],
    sink_token_indices: List[int],
) -> Dict[str, Any]:
    storage["num_samples"] += 1
    storage["num_valid_samples"] += 1

    if len(sink_token_indices) > 0:
        storage["num_samples_with_sink"] += 1

    for idx in sink_token_indices:
        idx = int(idx)
        if 0 <= idx < storage["counts"].numel():
            storage["counts"][idx] += 1

    return storage


def save_sink_token_records_jsonl(
    *,
    records: List[Dict[str, Any]],
    save_path: str,
) -> str:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return save_path


def load_sink_token_record_index(
    *,
    records_path: str,
) -> Dict[str, Any]:
    """
    Load sink-token records and build lookup tables for inference-time pruning.

    Lookup priority should prefer per-sample identifiers when available. The
    step/batch mapping is kept as a best-effort fallback for older record files.
    """
    by_sample_idx = {}
    by_source_row_index = {}
    by_step_batch = {}
    num_records = 0

    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            graph_token_positions = [int(pos) for pos in rec.get("graph_token_positions", [])]

            sample_idx = rec.get("sample_idx", None)
            if sample_idx is not None:
                by_sample_idx[int(sample_idx)] = graph_token_positions

            source_row_index = rec.get("source_row_index", None)
            if source_row_index is not None:
                by_source_row_index[int(source_row_index)] = graph_token_positions

            step = rec.get("step", None)
            batch_index = rec.get("batch_index", None)
            if step is not None and batch_index is not None:
                by_step_batch[(int(step), int(batch_index))] = graph_token_positions

            num_records += 1

    return {
        "records_path": records_path,
        "num_records": num_records,
        "by_sample_idx": by_sample_idx,
        "by_source_row_index": by_source_row_index,
        "by_step_batch": by_step_batch,
    }


def lookup_sink_record_prune_positions(
    *,
    record_index: Dict[str, Any],
    sample_idx: Optional[int] = None,
    source_row_index: Optional[int] = None,
    step: Optional[int] = None,
    batch_index: Optional[int] = None,
) -> Optional[List[int]]:
    """
    Resolve prune positions for a single sample. Returns None when the record
    cannot be found, and an empty list when the record exists but contains no
    sink tokens to prune.
    """
    if source_row_index is not None:
        positions = record_index["by_source_row_index"].get(int(source_row_index), None)
        if positions is not None:
            return list(positions)

    if sample_idx is not None:
        positions = record_index["by_sample_idx"].get(int(sample_idx), None)
        if positions is not None:
            return list(positions)

    if step is not None and batch_index is not None:
        positions = record_index["by_step_batch"].get((int(step), int(batch_index)), None)
        if positions is not None:
            return list(positions)

    return None


def build_prune_token_positions_tensor(
    *,
    prune_positions_batch: List[Optional[List[int]]],
    device: Optional[torch.device] = None,
) -> Optional[torch.LongTensor]:
    """
    Pack a batch of variable-length prune-position lists into a padded tensor.
    Uses -1 as the sentinel for "no more positions" within a row.
    """
    if len(prune_positions_batch) == 0:
        return None

    normalized_positions = []
    max_len = 0
    any_pruned = False

    for positions in prune_positions_batch:
        row = sorted({int(pos) for pos in (positions or [])})
        normalized_positions.append(row)
        if len(row) > 0:
            any_pruned = True
            max_len = max(max_len, len(row))

    if not any_pruned:
        return None

    packed = torch.full(
        (len(normalized_positions), max_len),
        -1,
        dtype=torch.long,
        device=device,
    )

    for batch_index, row in enumerate(normalized_positions):
        if not row:
            continue
        packed[batch_index, : len(row)] = torch.tensor(
            row,
            dtype=torch.long,
            device=device,
        )

    return packed


def sink_distribution(
    *,
    storage: Dict[str, Any],
    save_path: str,
    dpi: int = 180,
    title: Optional[str] = None,
) -> str:
    """
    Plot Graph sink token distribution plot, per-dataset
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

    counts = storage["counts"].detach().cpu().to(torch.long)
    num_graph_tokens = counts.numel()
    x = torch.arange(num_graph_tokens)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=dpi)
    ax.bar(x.tolist(), counts.tolist(), width=0.75)
    ax.set_title(title if title is not None else "Distribution of detected sink graph-token indices")
    ax.set_xlabel("Graph token index")
    ax.set_ylabel("Frequency")

    if num_graph_tokens <= 30:
        tick_step = 1
    elif num_graph_tokens <= 60:
        tick_step = 2
    else:
        tick_step = 5
    ax.set_xticks(list(range(0, num_graph_tokens, tick_step)))
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_sink_distribution_shift(
    *,
    baseline_counts: torch.Tensor,
    post_counts: torch.Tensor,
    save_path: str,
    dpi: int = 180,
    title: Optional[str] = None,
) -> str:
    """
    Overlay baseline vs post-prune sink-position distributions as line curves.
    Both inputs are [K] count vectors over the same K graph-token positions.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

    base = baseline_counts.detach().cpu().to(torch.long)
    post = post_counts.detach().cpu().to(torch.long)
    if base.numel() != post.numel():
        raise ValueError(
            f"baseline_counts and post_counts must have same length, got {base.numel()} vs {post.numel()}."
        )
    K = base.numel()
    x = torch.arange(K)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=dpi)
    ax.plot(x.tolist(), base.tolist(), color="#1f77b4", marker="o", linewidth=2, label="baseline")
    ax.plot(x.tolist(), post.tolist(), color="#d62728", marker="x", linewidth=2, label="after pruning")
    ax.set_title(title if title is not None else "Sink position distribution: baseline vs post-prune")
    ax.set_xlabel("Graph token index")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")

    if K <= 30:
        tick_step = 1
    elif K <= 60:
        tick_step = 2
    else:
        tick_step = 5
    ax.set_xticks(list(range(0, K, tick_step)))
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Sink re-emergence (top-k prune) — rank-indexed aggregator + plotters.
#
# Designed for the K=5 / top-2 setting: each sample contributes
#   - a baseline sink count (0..K),
#   - a post-prune sink count (0..K),
#   - per-survivor outcome (promoted / sustained / lost / null) indexed by
#     baseline-score rank.
#
# All counts live as torch tensors so the existing DDP all-reduce pattern in
# train_glm.py applies (sum across processes).

REOCCUR_OUTCOME_NAMES = ("promoted", "sustained", "lost", "null")
REOCCUR_OUTCOME_INDEX = {name: i for i, name in enumerate(REOCCUR_OUTCOME_NAMES)}


def aggregate_baseline_position_hist(
    records_index: Dict[Any, Dict[str, Any]],
    num_graph_tokens: int,
) -> torch.Tensor:
    """Sum baseline `sink_token_indices` counts per K-position across all records."""
    counts = torch.zeros(int(num_graph_tokens), dtype=torch.long)
    K = counts.numel()
    for rec in records_index.values():
        for i in rec.get("sink_token_indices", []):
            ii = int(i)
            if 0 <= ii < K:
                counts[ii] += 1
    return counts


def init_reoccur_summary_storage(
    *,
    num_graph_tokens: int = 5,
    num_pruned: int = 2,
) -> Dict[str, Any]:
    K = int(num_graph_tokens)
    P = int(num_pruned)
    return {
        "num_graph_tokens": K,
        "num_pruned": P,
        "num_samples": 0,
        "num_valid_samples": 0,
        "baseline_count_hist": torch.zeros(K + 1, dtype=torch.long),
        "post_count_hist": torch.zeros(K + 1, dtype=torch.long),
        "joint_count_hist": torch.zeros(K + 1, K + 1, dtype=torch.long),
        "reoccur_outcome_hist": torch.zeros(K + 1, dtype=torch.long),
        "rank_outcome_counts": torch.zeros(K, len(REOCCUR_OUTCOME_NAMES), dtype=torch.long),
        "post_position_hist": torch.zeros(K, dtype=torch.long),
    }


def update_reoccur_summary(
    *,
    storage: Dict[str, Any],
    baseline_layer_token_scores: torch.Tensor,
    baseline_sink_indices: List[int],
    pruned_indices: List[int],
    post_sink_indices: List[int],
) -> Dict[str, Any]:
    """
    Update the reoccur summary storage with one sample's outcome.

    All index arguments are in original K-space (0..K-1).
    `baseline_layer_token_scores` is the [K] per-token score vector returned by
    `detect_sink_tokens` on the baseline (pre-prune) forward pass.
    """
    K = storage["num_graph_tokens"]
    P = storage["num_pruned"]

    storage["num_samples"] += 1

    baseline_count = len(baseline_sink_indices)
    storage["baseline_count_hist"][min(baseline_count, K)] += 1

    pruned_set = {int(p) for p in pruned_indices}
    if len(pruned_set) < P:
        # Not enough baseline sinks to actually prune top-P; skip from valid stats.
        return storage

    storage["num_valid_samples"] += 1

    post_count = len(post_sink_indices)
    storage["post_count_hist"][min(post_count, K)] += 1
    storage["joint_count_hist"][min(baseline_count, K), min(post_count, K)] += 1

    for i in post_sink_indices:
        ii = int(i)
        if 0 <= ii < K:
            storage["post_position_hist"][ii] += 1

    baseline_set = {int(s) for s in baseline_sink_indices}
    post_set = {int(s) for s in post_sink_indices}
    surviving_indices = [k for k in range(K) if k not in pruned_set]

    reoccur_count = sum(
        1 for k in surviving_indices
        if (k in post_set and k not in baseline_set)
    )
    storage["reoccur_outcome_hist"][min(reoccur_count, K)] += 1

    scores = baseline_layer_token_scores.detach().to(torch.float32).cpu().reshape(-1)
    if scores.numel() != K:
        raise ValueError(
            f"baseline_layer_token_scores must have length K={K}, got {scores.numel()}."
        )
    order = torch.argsort(scores, descending=True).tolist()
    rank_of = [0] * K
    for r, k in enumerate(order, start=1):
        rank_of[k] = r

    for k in surviving_indices:
        was_sink = k in baseline_set
        is_sink_now = k in post_set
        if not was_sink and is_sink_now:
            outcome = REOCCUR_OUTCOME_INDEX["promoted"]
        elif was_sink and is_sink_now:
            outcome = REOCCUR_OUTCOME_INDEX["sustained"]
        elif was_sink and not is_sink_now:
            outcome = REOCCUR_OUTCOME_INDEX["lost"]
        else:
            outcome = REOCCUR_OUTCOME_INDEX["null"]
        r = rank_of[k]
        storage["rank_outcome_counts"][r - 1, outcome] += 1

    return storage


def _hist_stats(hist: List[int]) -> Dict[str, float]:
    n = int(sum(hist))
    if n == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "ge1_pct": 0.0, "ge2_pct": 0.0}
    total = sum(c * h for c, h in enumerate(hist))
    mean = total / n
    cum = 0
    median = 0
    half = n / 2.0
    for c, h in enumerate(hist):
        cum += h
        if cum >= half:
            median = c
            break
    ge1 = sum(hist[1:]) / n * 100.0
    ge2 = sum(hist[2:]) / n * 100.0
    return {"n": n, "mean": mean, "median": float(median), "ge1_pct": ge1, "ge2_pct": ge2}


def save_reoccur_summary_table(
    *,
    storage: Dict[str, Any],
    json_path: str,
    md_path: str,
    dataset_name: Optional[str] = None,
) -> Dict[str, str]:
    K = storage["num_graph_tokens"]
    P = storage["num_pruned"]
    n_total = int(storage["num_samples"])
    n_valid = int(storage["num_valid_samples"])

    baseline_hist = storage["baseline_count_hist"].tolist()
    post_hist = storage["post_count_hist"].tolist()
    reoccur_hist = storage["reoccur_outcome_hist"].tolist()
    joint = storage["joint_count_hist"].tolist()
    rank_outcome = storage["rank_outcome_counts"].tolist()

    base_stats = _hist_stats(baseline_hist)
    post_stats = _hist_stats(post_hist)

    any_reoccur = int(sum(reoccur_hist[1:]))
    any_reoccur_pct = (any_reoccur / n_valid * 100.0) if n_valid > 0 else 0.0

    summary = {
        "num_graph_tokens": K,
        "num_pruned": P,
        "dataset": dataset_name,
        "num_samples_total": n_total,
        "num_samples_valid": n_valid,
        "block1_sink_count_shift": {
            "baseline": base_stats,
            "post_prune": post_stats,
            "delta_mean": post_stats["mean"] - base_stats["mean"],
        },
        "block2_reoccur_outcomes": {
            "outcome_hist_count": reoccur_hist,
            "any_reoccur_count": any_reoccur,
            "any_reoccur_pct": any_reoccur_pct,
        },
        "joint_count_hist": joint,
        "rank_outcome_counts": rank_outcome,
        "rank_outcome_class_order": list(REOCCUR_OUTCOME_NAMES),
    }

    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_lines = []
    title_suffix = f" — {dataset_name}" if dataset_name else ""
    md_lines.append(f"# Sink re-emergence summary (top-{P} prune, K={K}){title_suffix}")
    md_lines.append("")
    md_lines.append(f"- Total samples: {n_total}")
    md_lines.append(f"- Valid samples (>={P} baseline sinks): {n_valid}")
    md_lines.append("")
    md_lines.append("## Block 1 - Sink-count shift")
    md_lines.append(f"| Metric | Baseline | Post top-{P} prune | Delta |")
    md_lines.append("|---|---:|---:|---:|")
    md_lines.append(
        f"| Mean sinks/sample | {base_stats['mean']:.3f} | {post_stats['mean']:.3f} "
        f"| {post_stats['mean'] - base_stats['mean']:+.3f} |"
    )
    md_lines.append(
        f"| Median sinks/sample | {base_stats['median']:.0f} | {post_stats['median']:.0f} | - |"
    )
    md_lines.append(
        f"| % samples >=1 sink | {base_stats['ge1_pct']:.1f}% | {post_stats['ge1_pct']:.1f}% "
        f"| {post_stats['ge1_pct'] - base_stats['ge1_pct']:+.1f}pp |"
    )
    md_lines.append(
        f"| % samples >=2 sinks | {base_stats['ge2_pct']:.1f}% | {post_stats['ge2_pct']:.1f}% "
        f"| {post_stats['ge2_pct'] - base_stats['ge2_pct']:+.1f}pp |"
    )
    md_lines.append("")
    md_lines.append(f"## Block 2 - Re-emergence outcomes (denominator: valid samples = {n_valid})")
    md_lines.append("| #re-emerged sinks | Count | % |")
    md_lines.append("|---|---:|---:|")
    max_possible = max(0, K - P)
    for r in range(max_possible + 1):
        c = int(reoccur_hist[r])
        pct = (c / n_valid * 100.0) if n_valid > 0 else 0.0
        md_lines.append(f"| {r} | {c} | {pct:.1f}% |")
    md_lines.append(f"| **Any (>=1)** | **{any_reoccur}** | **{any_reoccur_pct:.1f}%** |")
    md_lines.append("")
    md_lines.append("## Per-rank outcomes among survivors (rank by baseline score)")
    md_lines.append("| Rank | total | promoted | sustained | lost | null |")
    md_lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in range(P + 1, K + 1):
        row = rank_outcome[r - 1]
        total = int(sum(row))
        if total == 0:
            md_lines.append(f"| {r} | 0 | 0 | 0 | 0 | 0 |")
            continue
        cells = [
            f"{int(row[REOCCUR_OUTCOME_INDEX[name]])} "
            f"({int(row[REOCCUR_OUTCOME_INDEX[name]]) / total * 100:.1f}%)"
            for name in REOCCUR_OUTCOME_NAMES
        ]
        md_lines.append(f"| {r} | {total} | " + " | ".join(cells) + " |")
    md_lines.append("")

    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return {"json_path": json_path, "md_path": md_path}


def plot_reoccur_promotion_by_rank(
    *,
    storage: Dict[str, Any],
    save_path: str,
    dpi: int = 180,
    title: Optional[str] = None,
) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

    K = storage["num_graph_tokens"]
    P = storage["num_pruned"]
    rank_outcome = storage["rank_outcome_counts"].detach().cpu().to(torch.long)

    surviving_ranks = list(range(P + 1, K + 1))
    if len(surviving_ranks) == 0:
        return save_path

    class_colors = {
        "promoted": "#d62728",
        "sustained": "#1f77b4",
        "lost": "#7f7f7f",
        "null": "#dddddd",
    }

    totals = []
    fractions = []
    for r in surviving_ranks:
        row = rank_outcome[r - 1].tolist()
        total = int(sum(row))
        totals.append(total)
        if total > 0:
            fractions.append([row[REOCCUR_OUTCOME_INDEX[c]] / total for c in REOCCUR_OUTCOME_NAMES])
        else:
            fractions.append([0.0] * len(REOCCUR_OUTCOME_NAMES))

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=dpi)

    x_positions = list(range(len(surviving_ranks)))
    bottom = [0.0] * len(surviving_ranks)
    for class_name in REOCCUR_OUTCOME_NAMES:
        ci = REOCCUR_OUTCOME_INDEX[class_name]
        heights = [fractions[i][ci] for i in range(len(surviving_ranks))]
        ax.bar(
            x_positions,
            heights,
            bottom=bottom,
            label=class_name,
            color=class_colors[class_name],
            width=0.6,
            edgecolor="white",
            linewidth=0.5,
        )
        if class_name in ("promoted", "sustained"):
            for i, h in enumerate(heights):
                if h > 0.02:
                    ax.text(
                        x_positions[i],
                        bottom[i] + h / 2.0,
                        f"{h * 100:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white",
                        fontweight="bold",
                    )
        bottom = [bottom[i] + heights[i] for i in range(len(surviving_ranks))]

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"rank {r}\n(n={totals[i]})" for i, r in enumerate(surviving_ranks)])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of valid samples")
    ax.set_title(title if title is not None else f"Per-rank survivor outcomes after top-{P} sink prune")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_reoccur_count_joint(
    *,
    storage: Dict[str, Any],
    save_path: str,
    dpi: int = 180,
    title: Optional[str] = None,
) -> str:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

    K = storage["num_graph_tokens"]
    P = storage["num_pruned"]
    joint = storage["joint_count_hist"].detach().cpu().to(torch.long).numpy()

    base_range = list(range(P, K + 1))
    post_range = list(range(0, K - P + 1))
    if len(base_range) == 0 or len(post_range) == 0:
        return save_path

    M = joint[base_range][:, post_range]
    M_max = int(M.max()) if M.size > 0 else 0

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=dpi)
    im = ax.imshow(M, cmap="Blues", aspect="auto")

    ax.set_xticks(list(range(len(post_range))))
    ax.set_xticklabels([str(c) for c in post_range])
    ax.set_yticks(list(range(len(base_range))))
    ax.set_yticklabels([str(c) for c in base_range])
    ax.set_xlabel(f"Post top-{P}-prune sink count")
    ax.set_ylabel("Baseline sink count")
    ax.set_title(title if title is not None else f"Joint sink count: baseline x post top-{P} prune")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = int(M[i, j])
            if v > 0:
                ax.text(
                    j, i, str(v),
                    ha="center", va="center",
                    color="white" if (M_max > 0 and v > M_max / 2) else "black",
                    fontsize=9,
                )

    for i, b in enumerate(base_range):
        diag_post = b - P
        if diag_post in post_range:
            j = post_range.index(diag_post)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=1.5))

    fig.colorbar(im, ax=ax, label="Sample count")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path
