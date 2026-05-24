import os
import json
import torch
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

from utils.constants import DEFAULT_GRAPH_PAD_ID, GRAPH_TOKEN_INDEX


def _get_prompt_step_attentions(generate_outputs) -> Tuple[torch.Tensor, ...]:
    """
    Extract step-0 (prompt) attentions from HF generate outputs.

    Expected structure:
      - tuple over generation steps
        - each item is tuple over layers
          - each tensor is [B, H, Q, K]

    For decoder-only generation with cache, step 0 is prompt attention and Q == K.
    """
    attns = getattr(generate_outputs, "attentions", None)
    if attns is None:
        raise ValueError(
            "generate_outputs.attentions is None. Set output_attentions=True and return_dict_in_generate=True."
        )

    step0 = attns[0]
    if not isinstance(step0, (tuple, list)) or len(step0) == 0:
        raise ValueError("Unexpected attentions structure from generate().")

    t = step0[0]
    if t.dim() != 4 or t.shape[-1] != t.shape[-2]:
        raise ValueError(
            f"Step-0 attentions are not square. Got shape {tuple(t.shape)}. "
            "This usually means you are not looking at the prompt step."
        )

    return tuple(step0)


def get_expanded_graph_key_query_indices(
    *,
    input_ids_1d: torch.Tensor,
    attention_mask_1d: torch.Tensor,
    graphs: torch.Tensor,
    prompt_len: int,
    keep_pad_tokens: bool = True,
    mm_use_graph_special_token: bool = False,
    use_hop: Optional[int] = None,
    sample_neighbor_size: Optional[int] = None,
    graph_pad_id: int = DEFAULT_GRAPH_PAD_ID,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Compute expanded graph key indices and query indices in the prompt sequence.

    Returns:
      key_idx: graph key positions to analyze (includes pads if keep_pad_tokens=True)
      query_idx: query positions (text tokens after the last graph block)
      key_idx_all: all graph key positions in expanded prompt (always includes pads)
      key_is_pad: bool mask aligned to key_idx, True where the key token is pad
    """
    del sample_neighbor_size  # kept for signature parity with existing analysis functions

    if input_ids_1d.dim() != 1 or attention_mask_1d.dim() != 1:
        raise ValueError("input_ids_1d and attention_mask_1d must be 1D tensors.")

    valid = attention_mask_1d.bool()
    placeholder_pos = torch.nonzero(
        (input_ids_1d == GRAPH_TOKEN_INDEX) & valid, as_tuple=False
    ).squeeze(-1)
    if placeholder_pos.numel() == 0:
        return None

    if graphs is None:
        raise ValueError("graphs must be provided to compute expanded graph indices.")
    if graphs.dim() == 1:
        graphs = graphs.unsqueeze(0)

    placeholder_pos = torch.sort(placeholder_pos).values
    num_graph_blocks = graphs.shape[0]
    if placeholder_pos.numel() > num_graph_blocks:
        raise ValueError(
            f"Found {placeholder_pos.numel()} <graph> placeholders but only {num_graph_blocks} graph blocks."
        )

    selected_key_positions: List[torch.Tensor] = []
    selected_key_is_pad: List[torch.Tensor] = []
    all_key_positions: List[torch.Tensor] = []
    offset = 0

    for cur_graph_idx, p in enumerate(placeholder_pos.tolist()):
        start = p + offset
        g = graphs[cur_graph_idx]
        L_base = int(g.numel())

        if mm_use_graph_special_token:
            if use_hop is None:
                raise ValueError(
                    "use_hop is required when mm_use_graph_special_token=True."
                )
            L = L_base + (use_hop + 2)
            all_pos = torch.arange(start, start + L, device=input_ids_1d.device)
            key_pos = all_pos
            # No reliable pad mapping once separators are injected.
            key_is_pad = torch.zeros_like(key_pos, dtype=torch.bool)
        else:
            L = L_base
            all_pos = torch.arange(start, start + L, device=input_ids_1d.device)
            pad_mask = g == graph_pad_id
            if keep_pad_tokens:
                key_pos = all_pos
                key_is_pad = pad_mask
            else:
                keep = ~pad_mask
                key_pos = all_pos[keep]
                key_is_pad = torch.zeros_like(key_pos, dtype=torch.bool)

        selected_key_positions.append(key_pos)
        selected_key_is_pad.append(key_is_pad)
        all_key_positions.append(all_pos)
        offset += (L - 1)

    if len(selected_key_positions) == 0:
        return None

    key_idx = torch.cat(selected_key_positions, dim=0)
    key_is_pad = torch.cat(selected_key_is_pad, dim=0)
    key_idx_all = torch.cat(all_key_positions, dim=0)

    in_bounds = (key_idx >= 0) & (key_idx < prompt_len)
    key_idx = key_idx[in_bounds]
    key_is_pad = key_is_pad[in_bounds]
    key_idx_all = key_idx_all[(key_idx_all >= 0) & (key_idx_all < prompt_len)]

    if key_idx.numel() == 0 or key_idx_all.numel() == 0:
        return None

    query_start = int(key_idx_all.max().item()) + 1
    if query_start >= prompt_len:
        return None

    query_idx = torch.arange(query_start, prompt_len, device=input_ids_1d.device, dtype=torch.long)
    return key_idx, query_idx, key_idx_all, key_is_pad


def _compute_pad_nonpad_stats(
    layer_query_to_graph: torch.Tensor,
    key_is_pad: torch.Tensor,
) -> Dict[str, Any]:
    """
    Compute per-layer pad/non-pad means for one sample.
    """
    key_is_pad = key_is_pad.bool()
    num_pad = int(key_is_pad.sum().item())
    num_nonpad = int((~key_is_pad).sum().item())

    if num_pad == 0 or num_nonpad == 0:
        return {
            "num_pad_keys": num_pad,
            "num_nonpad_keys": num_nonpad,
            "layer_pad_mean": None,
            "layer_nonpad_mean": None,
            "layer_pad_nonpad_ratio": None,
        }

    pad_scores = layer_query_to_graph[:, :, key_is_pad]       # [L, Q, K_pad]
    nonpad_scores = layer_query_to_graph[:, :, ~key_is_pad]   # [L, Q, K_nonpad]

    layer_pad_mean = pad_scores.mean(dim=(1, 2))
    layer_nonpad_mean = nonpad_scores.mean(dim=(1, 2))

    return {
        "num_pad_keys": num_pad,
        "num_nonpad_keys": num_nonpad,
        "layer_pad_mean": layer_pad_mean,
        "layer_nonpad_mean": layer_nonpad_mean,
        "layer_pad_nonpad_ratio": layer_pad_mean / (layer_nonpad_mean + 1e-12),
    }


@torch.no_grad()
def compute_layerwise_query_to_graph_attention(
    *,
    generate_outputs,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    graphs: torch.Tensor,
    keep_pad_tokens: bool = True,
    mm_use_graph_special_token: bool = False,
    use_hop: Optional[int] = None,
    sample_neighbor_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Per sample, compute per-layer query->graph attention matrices averaged over heads.

    Output entries per sample:
      - layer_query_to_graph: [num_layers, Q, K]
      - key_idx: [K] expanded graph key indices in prompt
      - key_idx_all: all expanded graph key indices (with pads)
      - key_is_pad: [K] bool, True where key token is pad
      - query_idx: [Q] expanded query indices in prompt
      - pad_nonpad_stats: per-layer pad vs non-pad means (per sample)

    No cross-sample aggregation is performed.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    attns0 = _get_prompt_step_attentions(generate_outputs)
    num_layers = len(attns0)
    batch_size = attns0[0].shape[0]
    prompt_len = attns0[0].shape[-1]

    if graphs is not None and graphs.dim() == 3:
        graphs_batched = graphs
    else:
        graphs_batched = graphs.unsqueeze(0).expand(batch_size, -1, -1)

    valid: List[bool] = []
    layer_query_to_graph_list: List[Optional[torch.Tensor]] = []
    key_idx_list: List[Optional[torch.Tensor]] = []
    key_idx_all_list: List[Optional[torch.Tensor]] = []
    key_is_pad_list: List[Optional[torch.Tensor]] = []
    query_idx_list: List[Optional[torch.Tensor]] = []
    pad_nonpad_stats_list: List[Optional[Dict[str, Any]]] = []

    for b in range(batch_size):
        idxs = get_expanded_graph_key_query_indices(
            input_ids_1d=input_ids[b],
            attention_mask_1d=attention_mask[b],
            graphs=graphs_batched[b],
            prompt_len=prompt_len,
            keep_pad_tokens=keep_pad_tokens,
            mm_use_graph_special_token=mm_use_graph_special_token,
            use_hop=use_hop,
            sample_neighbor_size=sample_neighbor_size,
        )

        if idxs is None:
            valid.append(False)
            layer_query_to_graph_list.append(None)
            key_idx_list.append(None)
            key_idx_all_list.append(None)
            key_is_pad_list.append(None)
            query_idx_list.append(None)
            pad_nonpad_stats_list.append(None)
            continue

        key_idx, query_idx, key_idx_all, key_is_pad = idxs # if choosing to remove padded tokens, then key_idx_all will be different than key_idx

        layer_mats: List[torch.Tensor] = []
        for layer_id in range(num_layers):
            # [B, H, T, T] -> [H, T, T] for sample b
            layer_attn_b = attns0[layer_id][b].to(torch.float32)
            # [H, Q, K]
            sub = layer_attn_b.index_select(1, query_idx).index_select(2, key_idx)
            # head-average -> [Q, K]
            sub_head_avg = sub.mean(dim=0)
            layer_mats.append(sub_head_avg)

        layer_query_to_graph = torch.stack(layer_mats, dim=0)  # [L, Q, K]
        pad_nonpad_stats = _compute_pad_nonpad_stats(layer_query_to_graph, key_is_pad)

        valid.append(True)
        layer_query_to_graph_list.append(layer_query_to_graph)
        key_idx_list.append(key_idx)
        key_idx_all_list.append(key_idx_all)
        key_is_pad_list.append(key_is_pad)
        query_idx_list.append(query_idx)
        pad_nonpad_stats_list.append(pad_nonpad_stats)

    return {
        "valid": valid,
        "prompt_len": prompt_len,
        "layer_query_to_graph": layer_query_to_graph_list,
        "key_idx": key_idx_list,
        "key_idx_all": key_idx_all_list,
        "key_is_pad": key_is_pad_list,
        "query_idx": query_idx_list,
        "pad_nonpad_stats": pad_nonpad_stats_list,
    }


def _choose_ticks(length: int, max_ticks: int) -> List[int]:
    if length <= 0:
        return []
    if length <= max_ticks:
        return list(range(length))
    step = max(1, length // (max_ticks - 1))
    ticks = list(range(0, length, step))
    if ticks[-1] != length - 1:
        ticks.append(length - 1)
    return ticks


def _build_key_tick_labels(
    x_ticks: List[int],
    k_len: int,
    key_idx: Optional[torch.Tensor],
    key_is_pad: Optional[torch.Tensor],
    sink_prompt_positions: Optional[List[int]] = None,
    top2_sink_prompt_positions: Optional[List[int]] = None,
) -> List[str]:
    labels: List[str] = []
    has_pad_mask = key_is_pad is not None and key_is_pad.numel() == k_len
    sink_prompt_position_set = (
        {int(pos) for pos in sink_prompt_positions}
        if sink_prompt_positions is not None
        else set()
    )
    top2_sink_prompt_position_set = (
        {int(pos) for pos in top2_sink_prompt_positions}
        if top2_sink_prompt_positions is not None
        else set()
    )

    for t in x_ticks:
        if key_idx is not None and key_idx.numel() == k_len:
            prompt_pos = int(key_idx[t])
        else:
            prompt_pos = int(t)

        base = str(prompt_pos)
        is_pad = has_pad_mask and bool(key_is_pad[t])

        if prompt_pos in top2_sink_prompt_position_set:
            base += "[T2P]" if is_pad else "[T2]"
        elif prompt_pos in sink_prompt_position_set:
            base += "[SP]" if is_pad else "[S]"
        elif is_pad:
            base += "(P)"
        labels.append(base)

    return labels


def plot_layerwise_query_graph_attention(
    *,
    layer_query_to_graph: torch.Tensor,
    sample_id: str,
    save_dir: str,
    query_idx: Optional[torch.Tensor] = None,
    key_idx: Optional[torch.Tensor] = None,
    key_is_pad: Optional[torch.Tensor] = None,
    sink_prompt_positions: Optional[List[int]] = None,
    top2_sink_prompt_positions: Optional[List[int]] = None,
    cmap: str = "viridis",
    dpi: int = 180,
    max_xticks: int = 16,
    max_yticks: int = 16,
) -> List[str]:
    """
    Save one heatmap per layer.

    y-axis: query tokens
    x-axis: graph key tokens
    color: head-averaged attention weight

    X tick labels can also annotate sink identities:
      - "(P)" for padded graph keys
      - "[S]" / "[SP]" for sink tokens
      - "[T2]" / "[T2P]" for top-2 sink tokens
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it to save attention heatmaps."
        ) from e

    if layer_query_to_graph.dim() != 3:
        raise ValueError("layer_query_to_graph must have shape [num_layers, Q, K].")

    os.makedirs(save_dir, exist_ok=True)

    layer_query_to_graph = layer_query_to_graph.detach().cpu()
    num_layers, q_len, k_len = layer_query_to_graph.shape

    if q_len == 0 or k_len == 0:
        return []

    if query_idx is not None:
        query_idx = query_idx.detach().cpu()
    if key_idx is not None:
        key_idx = key_idx.detach().cpu()
    if key_is_pad is not None:
        key_is_pad = key_is_pad.detach().cpu().bool()

    global_vmin = float(layer_query_to_graph.min().item())
    global_vmax = float(layer_query_to_graph.max().item())

    saved_paths: List[str] = []

    for layer_id in range(num_layers):
        mat = layer_query_to_graph[layer_id]

        fig_w = max(6.0, 0.14 * k_len)
        fig_h = max(4.0, 0.14 * q_len)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

        im = ax.imshow(
            mat.numpy(),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=global_vmin,
            vmax=global_vmax,
        )

        ax.set_title(f"Sample {sample_id} | Layer {layer_id}")

        ax.set_xlabel("Graph key tokens")
        ax.set_ylabel("Query tokens")

        # Show every graph token index on x-axis.
        x_ticks = list(range(k_len))
        y_ticks = _choose_ticks(q_len, max_yticks)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        x_labels = _build_key_tick_labels(
            x_ticks,
            k_len,
            key_idx,
            key_is_pad,
            sink_prompt_positions=sink_prompt_positions,
            top2_sink_prompt_positions=top2_sink_prompt_positions,
        )
        ax.set_xticklabels(x_labels, rotation=90, ha="center")

        if query_idx is not None and query_idx.numel() == q_len:
            ax.set_yticklabels([str(int(query_idx[t])) for t in y_ticks])
        else:
            ax.set_yticklabels([str(t) for t in y_ticks])

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Attention weight (head-avg)")

        fig.tight_layout()
        out_path = os.path.join(save_dir, f"sample_{sample_id}_layer_{layer_id:02d}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        saved_paths.append(out_path)

    return saved_paths


def plot_layeravg_query_graph_attention(
    *,
    layer_query_to_graph: torch.Tensor,
    sample_id: str,
    save_dir: Optional[str] = None,
    save_path: Optional[str] = None,
    query_idx: Optional[torch.Tensor] = None,
    key_idx: Optional[torch.Tensor] = None,
    key_is_pad: Optional[torch.Tensor] = None,
    sink_prompt_positions: Optional[List[int]] = None,
    top2_sink_prompt_positions: Optional[List[int]] = None,
    cmap: str = "viridis",
    dpi: int = 180,
    max_xticks: int = 16,
    max_yticks: int = 16,
) -> Optional[str]:
    """
    Save one heatmap after averaging attention across layers.

    y-axis: query tokens
    x-axis: graph key tokens
    color: head-averaged attention weight (then layer-averaged)

    X tick labels can also annotate sink identities:
      - "(P)" for padded graph keys
      - "[S]" / "[SP]" for sink tokens
      - "[T2]" / "[T2P]" for top-2 sink tokens
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it to save attention heatmaps."
        ) from e

    if layer_query_to_graph.dim() != 3:
        raise ValueError("layer_query_to_graph must have shape [num_layers, Q, K].")

    if save_path is None and save_dir is None:
        raise ValueError("Provide save_dir or save_path when saving a layer-averaged attention plot.")

    if save_path is not None:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(save_dir, exist_ok=True)

    layer_query_to_graph = layer_query_to_graph.detach().cpu()
    avg_mat = layer_query_to_graph.mean(dim=0)  # [Q, K]
    q_len, k_len = avg_mat.shape

    if q_len == 0 or k_len == 0:
        return None

    if query_idx is not None:
        query_idx = query_idx.detach().cpu()
    if key_idx is not None:
        key_idx = key_idx.detach().cpu()
    if key_is_pad is not None:
        key_is_pad = key_is_pad.detach().cpu().bool()

    fig_w = max(6.0, 0.14 * k_len)
    fig_h = max(4.0, 0.14 * q_len)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    im = ax.imshow(
        avg_mat.numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
    )

    ax.set_title(f"Sample {sample_id} | Attention To All Graph Tokens | Layer Avg")

    ax.set_xlabel("All graph tokens")
    ax.set_ylabel("Query tokens")

    # Show every graph token index on x-axis.
    x_ticks = list(range(k_len))
    y_ticks = _choose_ticks(q_len, max_yticks)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    x_labels = _build_key_tick_labels(
        x_ticks,
        k_len,
        key_idx,
        key_is_pad,
        sink_prompt_positions=sink_prompt_positions,
        top2_sink_prompt_positions=top2_sink_prompt_positions,
    )
    ax.set_xticklabels(x_labels, rotation=90, ha="center")

    if query_idx is not None and query_idx.numel() == q_len:
        ax.set_yticklabels([str(int(query_idx[t])) for t in y_ticks])
    else:
        ax.set_yticklabels([str(t) for t in y_ticks])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attention weight (head-avg, layer-avg)")

    fig.tight_layout()
    out_path = save_path
    if out_path is None:
        out_path = os.path.join(save_dir, f"sample_{sample_id}_layer_avg.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return out_path


def analyze_and_plot_sample_attention(
    *,
    generate_outputs,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    graphs: torch.Tensor,
    sample_id: str,
    save_dir: str,
    keep_pad_tokens: bool = True,
    mm_use_graph_special_token: bool = False,
    use_hop: Optional[int] = None,
    sample_neighbor_size: Optional[int] = None,
    sink_prompt_positions: Optional[List[int]] = None,
    top2_sink_prompt_positions: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper for single-sample usage.

    Returns analysis dict plus:
      - saved_paths: list of heatmap files for this sample
    """
    analysis = compute_layerwise_query_to_graph_attention(
        generate_outputs=generate_outputs,
        input_ids=input_ids,
        attention_mask=attention_mask,
        graphs=graphs,
        keep_pad_tokens=keep_pad_tokens,
        mm_use_graph_special_token=mm_use_graph_special_token,
        use_hop=use_hop,
        sample_neighbor_size=sample_neighbor_size,
    )

    saved_paths: List[str] = []

    # Primary use case in eval_pretrain is batch size 1, but this handles B>1.
    for b, is_valid in enumerate(analysis["valid"]):
        if not is_valid:
            continue

        layer_qk = analysis["layer_query_to_graph"][b]
        query_idx = analysis["query_idx"][b]
        key_idx = analysis["key_idx"][b]
        key_is_pad = analysis["key_is_pad"][b]

        suffix = sample_id if len(analysis["valid"]) == 1 else f"{sample_id}_b{b}"
        paths = plot_layerwise_query_graph_attention(
            layer_query_to_graph=layer_qk,
            sample_id=suffix,
            save_dir=save_dir,
            query_idx=query_idx,
            key_idx=key_idx,
            key_is_pad=key_is_pad,
            sink_prompt_positions=sink_prompt_positions,
            top2_sink_prompt_positions=top2_sink_prompt_positions,
        )
        saved_paths.extend(paths)

    analysis["saved_paths"] = saved_paths
    return analysis


def analyze_and_plot_sample_attention_layeravg(
    *,
    generate_outputs,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    graphs: torch.Tensor,
    sample_id: str,
    save_dir: Optional[str] = None,
    save_path: Optional[str] = None,
    keep_pad_tokens: bool = True,
    mm_use_graph_special_token: bool = False,
    use_hop: Optional[int] = None,
    sample_neighbor_size: Optional[int] = None,
    plotting: bool = False,
    sink_prompt_positions: Optional[List[int]] = None,
    top2_sink_prompt_positions: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper for single-sample usage with one layer-averaged plot per sample.

    Returns analysis dict plus:
      - saved_paths_layeravg: list of layer-averaged heatmap files
      - optional x-axis markers for sink and top-2 sink prompt positions
    """
    analysis = compute_layerwise_query_to_graph_attention(
        generate_outputs=generate_outputs,
        input_ids=input_ids,
        attention_mask=attention_mask,
        graphs=graphs,
        keep_pad_tokens=keep_pad_tokens,
        mm_use_graph_special_token=mm_use_graph_special_token,
        use_hop=use_hop,
        sample_neighbor_size=sample_neighbor_size,
    )

    saved_paths_layeravg: List[str] = []
    if plotting:
        for b, is_valid in enumerate(analysis["valid"]):
            if not is_valid:
                continue

            layer_qk = analysis["layer_query_to_graph"][b]
            query_idx = analysis["query_idx"][b]
            key_idx = analysis["key_idx"][b]
            key_is_pad = analysis["key_is_pad"][b]

            suffix = sample_id if len(analysis["valid"]) == 1 else f"{sample_id}_b{b}"
            if save_path is not None and len(analysis["valid"]) != 1:
                raise ValueError("save_path is only supported when plotting a single sample.")

            out_path = plot_layeravg_query_graph_attention(
                layer_query_to_graph=layer_qk,
                sample_id=suffix,
                save_dir=save_dir,
                save_path=save_path,
                query_idx=query_idx,
                key_idx=key_idx,
                key_is_pad=key_is_pad,
                sink_prompt_positions=sink_prompt_positions,
                top2_sink_prompt_positions=top2_sink_prompt_positions,
            )
            if out_path is not None:
                saved_paths_layeravg.append(out_path)

    analysis["saved_paths_layeravg"] = saved_paths_layeravg
    return analysis


def extract_sink_columns_from_query_graph_attention(
    *,
    layer_query_to_graph: torch.Tensor,
    key_idx: torch.Tensor,
    sink_prompt_positions: List[int],
    key_is_pad: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Select sink-token columns from one sample's [L, Q, K] query->graph attention.

    Sink tokens are identified by their expanded prompt positions.
    """
    if layer_query_to_graph.dim() != 3:
        raise ValueError("layer_query_to_graph must have shape [num_layers, Q, K].")

    key_idx = key_idx.reshape(-1)
    if key_idx.numel() != layer_query_to_graph.shape[-1]:
        raise ValueError("key_idx must align with the last dimension of layer_query_to_graph.")

    requested_positions = sorted({int(pos) for pos in sink_prompt_positions})
    if len(requested_positions) == 0:
        return {
            "valid": False,
            "reason": "no_sink_prompt_positions",
            "layer_query_to_sink": None,
            "sink_key_idx": None,
            "sink_key_is_pad": None,
            "sink_column_indices": None,
            "missing_sink_prompt_positions": [],
        }

    has_pad_mask = key_is_pad is not None and key_is_pad.numel() == key_idx.numel()
    if has_pad_mask:
        key_is_pad = key_is_pad.reshape(-1).bool()

    col_indices: List[int] = []
    found_positions: List[int] = []
    found_pad_flags: List[bool] = []

    for pos in requested_positions:
        matches = torch.nonzero(key_idx == int(pos), as_tuple=False).reshape(-1)
        if matches.numel() == 0:
            continue

        col = int(matches[0].item())
        col_indices.append(col)
        found_positions.append(int(key_idx[col].item()))
        if has_pad_mask:
            found_pad_flags.append(bool(key_is_pad[col].item()))

    missing_positions = [pos for pos in requested_positions if pos not in set(found_positions)]
    if len(col_indices) == 0:
        return {
            "valid": False,
            "reason": "sink_prompt_positions_not_found_in_key_idx",
            "layer_query_to_sink": None,
            "sink_key_idx": None,
            "sink_key_is_pad": None,
            "sink_column_indices": None,
            "missing_sink_prompt_positions": missing_positions,
        }

    col_idx_attn = torch.tensor(col_indices, dtype=torch.long, device=layer_query_to_graph.device)
    layer_query_to_sink = layer_query_to_graph.index_select(2, col_idx_attn)

    sink_key_idx = torch.tensor(found_positions, dtype=key_idx.dtype, device=key_idx.device)
    sink_key_is_pad = None
    if has_pad_mask:
        sink_key_is_pad = torch.tensor(found_pad_flags, dtype=torch.bool, device=key_idx.device)

    return {
        "valid": True,
        "reason": None,
        "layer_query_to_sink": layer_query_to_sink,
        "sink_key_idx": sink_key_idx,
        "sink_key_is_pad": sink_key_is_pad,
        "sink_column_indices": torch.tensor(col_indices, dtype=torch.long, device=key_idx.device),
        "missing_sink_prompt_positions": missing_positions,
    }


def plot_layeravg_query_sink_attention(
    *,
    layer_query_to_sink: torch.Tensor,
    sample_id: str,
    save_path: str,
    query_idx: Optional[torch.Tensor] = None,
    sink_key_idx: Optional[torch.Tensor] = None,
    sink_key_is_pad: Optional[torch.Tensor] = None,
    cmap: str = "viridis",
    dpi: int = 180,
    max_yticks: int = 16,
) -> Optional[str]:
    """
    Save one heatmap after averaging attention across layers for sink-token columns.

    y-axis: query tokens after the graph block
    x-axis: sink graph tokens
    color: head-averaged attention weight (then layer-averaged)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it to save attention heatmaps."
        ) from e

    if layer_query_to_sink.dim() != 3:
        raise ValueError("layer_query_to_sink must have shape [num_layers, Q, S].")

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    layer_query_to_sink = layer_query_to_sink.detach().cpu()
    avg_mat = layer_query_to_sink.mean(dim=0)  # [Q, S]
    q_len, s_len = avg_mat.shape

    if q_len == 0 or s_len == 0:
        return None

    if query_idx is not None:
        query_idx = query_idx.detach().cpu()
    if sink_key_idx is not None:
        sink_key_idx = sink_key_idx.detach().cpu()
    if sink_key_is_pad is not None:
        sink_key_is_pad = sink_key_is_pad.detach().cpu().bool()

    fig_w = max(6.0, 0.7 * s_len)
    fig_h = max(4.0, 0.14 * q_len)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    im = ax.imshow(
        avg_mat.numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
    )

    ax.set_title(f"Sample {sample_id} | Attention To Sink Tokens | Layer Avg")
    ax.set_xlabel("Sink tokens")
    ax.set_ylabel("Query tokens")

    x_ticks = list(range(s_len))
    y_ticks = _choose_ticks(q_len, max_yticks)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    x_labels = _build_key_tick_labels(x_ticks, s_len, sink_key_idx, sink_key_is_pad)
    ax.set_xticklabels(x_labels, rotation=90, ha="center")

    if query_idx is not None and query_idx.numel() == q_len:
        ax.set_yticklabels([str(int(query_idx[t])) for t in y_ticks])
    else:
        ax.set_yticklabels([str(t) for t in y_ticks])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attention weight (head-avg, layer-avg)")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return save_path


def analyze_and_plot_sample_attention_to_sink_layeravg(
    *,
    generate_outputs,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    graphs: torch.Tensor,
    sink_prompt_positions: List[int],
    sample_id: str,
    save_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    keep_pad_tokens: bool = True,
    mm_use_graph_special_token: bool = False,
    use_hop: Optional[int] = None,
    sample_neighbor_size: Optional[int] = None,
    plotting: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper for per-sample prompt attention restricted to sink tokens.

    Returns analysis dict plus:
      - layer_query_to_sink: list of [L, Q, S] tensors or None
      - sink_key_idx: sink prompt positions that were found in key_idx
      - saved_paths_attention_to_sink: list of saved .jpg heatmaps
    """
    if plotting and save_path is None and save_dir is None:
        raise ValueError("Provide save_path or save_dir when plotting=True.")

    analysis = compute_layerwise_query_to_graph_attention(
        generate_outputs=generate_outputs,
        input_ids=input_ids,
        attention_mask=attention_mask,
        graphs=graphs,
        keep_pad_tokens=keep_pad_tokens,
        mm_use_graph_special_token=mm_use_graph_special_token,
        use_hop=use_hop,
        sample_neighbor_size=sample_neighbor_size,
    )

    sink_valid: List[bool] = []
    layer_query_to_sink_list: List[Optional[torch.Tensor]] = []
    sink_key_idx_list: List[Optional[torch.Tensor]] = []
    sink_key_is_pad_list: List[Optional[torch.Tensor]] = []
    sink_column_indices_list: List[Optional[torch.Tensor]] = []
    missing_sink_prompt_positions_list: List[List[int]] = []
    saved_paths_attention_to_sink: List[str] = []

    for b, is_valid in enumerate(analysis["valid"]):
        if not is_valid:
            sink_valid.append(False)
            layer_query_to_sink_list.append(None)
            sink_key_idx_list.append(None)
            sink_key_is_pad_list.append(None)
            sink_column_indices_list.append(None)
            missing_sink_prompt_positions_list.append([])
            continue

        selected = extract_sink_columns_from_query_graph_attention(
            layer_query_to_graph=analysis["layer_query_to_graph"][b],
            key_idx=analysis["key_idx"][b],
            sink_prompt_positions=sink_prompt_positions,
            key_is_pad=analysis["key_is_pad"][b],
        )

        sink_valid.append(bool(selected["valid"]))
        layer_query_to_sink_list.append(selected["layer_query_to_sink"])
        sink_key_idx_list.append(selected["sink_key_idx"])
        sink_key_is_pad_list.append(selected["sink_key_is_pad"])
        sink_column_indices_list.append(selected["sink_column_indices"])
        missing_sink_prompt_positions_list.append(selected["missing_sink_prompt_positions"])

        if plotting and selected["valid"]:
            suffix = sample_id if len(analysis["valid"]) == 1 else f"{sample_id}_b{b}"
            if save_path is not None and len(analysis["valid"]) != 1:
                raise ValueError("save_path is only supported when plotting a single sample.")

            cur_save_path = save_path
            if cur_save_path is None:
                cur_save_path = os.path.join(save_dir, f"{suffix}.jpg")

            out_path = plot_layeravg_query_sink_attention(
                layer_query_to_sink=selected["layer_query_to_sink"],
                sample_id=suffix,
                save_path=cur_save_path,
                query_idx=analysis["query_idx"][b],
                sink_key_idx=selected["sink_key_idx"],
                sink_key_is_pad=selected["sink_key_is_pad"],
            )
            if out_path is not None:
                saved_paths_attention_to_sink.append(out_path)

    analysis["sink_valid"] = sink_valid
    analysis["layer_query_to_sink"] = layer_query_to_sink_list
    analysis["sink_key_idx"] = sink_key_idx_list
    analysis["sink_key_is_pad"] = sink_key_is_pad_list
    analysis["sink_column_indices"] = sink_column_indices_list
    analysis["missing_sink_prompt_positions"] = missing_sink_prompt_positions_list
    analysis["saved_paths_attention_to_sink"] = saved_paths_attention_to_sink
    return analysis


def _token_scores_from_layer_query_to_graph(layer_query_to_graph: torch.Tensor) -> torch.Tensor:
    """Return one score per graph token by averaging layers and queries."""
    if layer_query_to_graph.dim() == 3:
        return layer_query_to_graph.mean(dim=(0, 1))
    if layer_query_to_graph.dim() == 2:
        return layer_query_to_graph.mean(dim=0)
    if layer_query_to_graph.dim() == 1:
        return layer_query_to_graph
    raise ValueError("layer_query_to_graph must have shape [L,Q,K], [Q,K], or [K].")


def _find_first_nonpad_after_first_pad(key_is_pad: torch.Tensor) -> Optional[int]:
    """
    Find the token index j such that:
      - key_is_pad contains a first pad run starting at i
      - j is the first non-pad index after that first run
    """
    key_is_pad = key_is_pad.bool().reshape(-1)
    pad_pos = torch.nonzero(key_is_pad, as_tuple=False).reshape(-1)
    if pad_pos.numel() == 0:
        return None

    j = int(pad_pos[0].item())
    n = key_is_pad.numel()
    while j < n and bool(key_is_pad[j]):
        j += 1

    if j >= n:
        return None
    return j


def compute_first_postpad_center_cosine_similarity(
    *,
    graphs: torch.Tensor,
    graph_emb: torch.Tensor,
    key_is_pad: Optional[torch.Tensor] = None,
    graph_pad_id: int = DEFAULT_GRAPH_PAD_ID,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    Compute cosine similarity between:
      - the first non-pad graph token after the first pad run
      - the center graph token in the same graph block

    The center token is defined as the first non-pad token in that block, which is
    consistent with the existing center-node logic used elsewhere in this file.
    """
    if graphs.dim() == 1:
        graphs = graphs.unsqueeze(0)
    graphs = graphs.detach().cpu().to(torch.long)
    flat_nodes = graphs.reshape(-1)
    G, Lg = graphs.shape
    del G

    if key_is_pad is not None:
        key_is_pad = key_is_pad.detach().cpu().bool().reshape(-1)
    else:
        key_is_pad = flat_nodes == graph_pad_id

    if key_is_pad.numel() != flat_nodes.numel():
        return {
            "valid": False,
            "reason": "key_is_pad_length_mismatch",
            "target_token_index": None,
            "center_token_index": None,
            "target_node_id": None,
            "center_node_id": None,
            "cosine_similarity": None,
        }

    emb = graph_emb.detach().cpu().to(torch.float32)
    if emb.dim() == 3:
        flat_emb = emb.reshape(-1, emb.shape[-1])
    elif emb.dim() == 2:
        flat_emb = emb
    else:
        return {
            "valid": False,
            "reason": f"unexpected_graph_emb_dim_{emb.dim()}",
            "target_token_index": None,
            "center_token_index": None,
            "target_node_id": None,
            "center_node_id": None,
            "cosine_similarity": None,
        }

    if flat_emb.shape[0] != flat_nodes.numel():
        return {
            "valid": False,
            "reason": "graph_emb_and_graphs_not_aligned",
            "target_token_index": None,
            "center_token_index": None,
            "target_node_id": None,
            "center_node_id": None,
            "cosine_similarity": None,
        }

    target_idx = _find_first_nonpad_after_first_pad(key_is_pad)
    if target_idx is None:
        return {
            "valid": False,
            "reason": "no_postpad_nonpad_token",
            "target_token_index": None,
            "center_token_index": None,
            "target_node_id": None,
            "center_node_id": None,
            "cosine_similarity": None,
        }

    if bool(key_is_pad[target_idx]):
        return {
            "valid": False,
            "reason": "target_is_still_pad",
            "target_token_index": int(target_idx),
            "center_token_index": None,
            "target_node_id": None,
            "center_node_id": None,
            "cosine_similarity": None,
        }

    block_id = int(target_idx // Lg)
    block_start = block_id * Lg
    block_pad = key_is_pad[block_start : block_start + Lg]
    center_local = torch.nonzero(~block_pad, as_tuple=False).reshape(-1)
    if center_local.numel() == 0:
        return {
            "valid": False,
            "reason": "no_nonpad_center_token_in_block",
            "target_token_index": int(target_idx),
            "center_token_index": None,
            "target_node_id": int(flat_nodes[target_idx].item()),
            "center_node_id": None,
            "cosine_similarity": None,
        }

    center_idx = int(block_start + int(center_local[0].item()))
    target_vec = flat_emb[target_idx].reshape(1, -1)
    center_vec = flat_emb[center_idx].reshape(1, -1)
    cosine_similarity = float(
        torch.nn.functional.cosine_similarity(
            target_vec,
            center_vec,
            dim=-1,
            eps=eps,
        )[0].item()
    )

    return {
        "valid": True,
        "reason": "ok",
        "target_token_index": int(target_idx),
        "center_token_index": int(center_idx),
        "target_node_id": int(flat_nodes[target_idx].item()),
        "center_node_id": int(flat_nodes[center_idx].item()),
        "cosine_similarity": cosine_similarity,
    }


def _edge_index_to_adjacency(edge_index: Optional[torch.Tensor]) -> Optional[Dict[int, set]]:
    if edge_index is None:
        return None

    e = edge_index.detach().cpu().to(torch.long)
    if e.dim() != 2 or e.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    adj: Dict[int, set] = {}
    for u, v in e.t().tolist():
        ui = int(u)
        vi = int(v)
        if ui == vi:
            continue
        adj.setdefault(ui, set()).add(vi)
        adj.setdefault(vi, set()).add(ui)
    return adj


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


def check_first_postpad_token_highest_attention(
    *,
    layer_query_to_graph: torch.Tensor,
    graphs: torch.Tensor,
    key_is_pad: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None,
    center_node_ids: Optional[torch.Tensor] = None,
    graph_pad_id: int = DEFAULT_GRAPH_PAD_ID,
    eps: float = 1e-8,
    adjacency: Optional[Dict[int, set]] = None,
) -> Dict[str, Any]:
    """
    Per sample check:
      1) Find the first real graph token after the first pad-token group.
      2) Check whether this token has the highest attention among non-pad graph tokens.
      3) Classify this token as center / one_hop / two_hop / rest.
      4) Check whether the 12th graph token is highest among all graph tokens.
      5) Check whether the 12th graph token is the same as the post-pad target token.

    Notes:
      - Uses layer+query averaged attention score per graph token.
      - Assumes graphs are token-node IDs in flatten order [G, Lg] -> [K].
    """
    scores = _token_scores_from_layer_query_to_graph(layer_query_to_graph).detach().to(torch.float32).cpu().reshape(-1)

    if graphs.dim() == 1:
        graphs = graphs.unsqueeze(0)
    graphs = graphs.detach().cpu().to(torch.long)
    flat_nodes = graphs.reshape(-1)
    G, Lg = graphs.shape

    if key_is_pad is not None:
        key_is_pad = key_is_pad.detach().cpu().bool().reshape(-1)
        if key_is_pad.numel() != scores.numel():
            return {
                "valid": False,
                "reason": "key_is_pad_length_mismatch",
                "target_token_index": None,
                "target_node_id": None,
                "target_score": None,
                "max_nonpad_score": None,
                "target_is_highest": False,
                "hop_category": "rest",
                "twelfth_token_exists": bool(scores.numel() >= 12),
                "twelfth_token_index": 11 if scores.numel() >= 12 else None,
                "twelfth_token_score": float(scores[11].item()) if scores.numel() >= 12 else None,
                "max_all_score": float(scores.max().item()) if scores.numel() > 0 else None,
                "twelfth_token_is_highest": bool(scores[11].item() >= (scores.max().item() - eps)) if scores.numel() >= 12 else False,
                "twelfth_token_matches_target": False,
            }
    else:
        if flat_nodes.numel() != scores.numel():
            return {
                "valid": False,
                "reason": "graphs_length_mismatch_without_key_is_pad",
                "target_token_index": None,
                "target_node_id": None,
                "target_score": None,
                "max_nonpad_score": None,
                "target_is_highest": False,
                "hop_category": "rest",
                "twelfth_token_exists": bool(scores.numel() >= 12),
                "twelfth_token_index": 11 if scores.numel() >= 12 else None,
                "twelfth_token_score": float(scores[11].item()) if scores.numel() >= 12 else None,
                "max_all_score": float(scores.max().item()) if scores.numel() > 0 else None,
                "twelfth_token_is_highest": bool(scores[11].item() >= (scores.max().item() - eps)) if scores.numel() >= 12 else False,
                "twelfth_token_matches_target": False,
            }
        key_is_pad = flat_nodes == graph_pad_id

    twelfth_token_exists = bool(scores.numel() >= 12)
    twelfth_token_index = 11 if twelfth_token_exists else None
    max_all_score = float(scores.max().item()) if scores.numel() > 0 else None
    twelfth_token_score = float(scores[11].item()) if twelfth_token_exists else None
    twelfth_token_is_highest = (
        bool(scores[11].item() >= (max_all_score - eps)) if twelfth_token_exists else False
    )

    if flat_nodes.numel() != scores.numel():
        return {
            "valid": False,
            "reason": "graphs_and_scores_not_aligned",
            "target_token_index": None,
            "target_node_id": None,
            "target_score": None,
            "max_nonpad_score": None,
            "target_is_highest": False,
            "hop_category": "rest",
            "twelfth_token_exists": twelfth_token_exists,
            "twelfth_token_index": twelfth_token_index,
            "twelfth_token_score": twelfth_token_score,
            "max_all_score": max_all_score,
            "twelfth_token_is_highest": twelfth_token_is_highest,
            "twelfth_token_matches_target": False,
        }
    # the target graph token that comes right after first sets of padded tokens
    target_idx = _find_first_nonpad_after_first_pad(key_is_pad)
    if target_idx is None:
        return {
            "valid": False,
            "reason": "no_postpad_nonpad_token",
            "target_token_index": None,
            "target_node_id": None,
            "target_score": None,
            "max_nonpad_score": None,
            "target_is_highest": False,
            "hop_category": "rest",
            "twelfth_token_exists": twelfth_token_exists,
            "twelfth_token_index": twelfth_token_index,
            "twelfth_token_score": twelfth_token_score,
            "max_all_score": max_all_score,
            "twelfth_token_is_highest": twelfth_token_is_highest,
            "twelfth_token_matches_target": False,
        }

    nonpad_mask = ~key_is_pad
    if int(nonpad_mask.sum().item()) == 0:
        return {
            "valid": False,
            "reason": "no_nonpad_tokens",
            "target_token_index": None,
            "target_node_id": None,
            "target_score": None,
            "max_nonpad_score": None,
            "target_is_highest": False,
            "hop_category": "rest",
            "twelfth_token_exists": twelfth_token_exists,
            "twelfth_token_index": twelfth_token_index,
            "twelfth_token_score": twelfth_token_score,
            "max_all_score": max_all_score,
            "twelfth_token_is_highest": twelfth_token_is_highest,
            "twelfth_token_matches_target": False,
        }

    if bool(key_is_pad[target_idx]):
        return {
            "valid": False,
            "reason": "target_is_still_pad",
            "target_token_index": int(target_idx),
            "target_node_id": None,
            "target_score": None,
            "max_nonpad_score": None,
            "target_is_highest": False,
            "hop_category": "rest",
            "twelfth_token_exists": twelfth_token_exists,
            "twelfth_token_index": twelfth_token_index,
            "twelfth_token_score": twelfth_token_score,
            "max_all_score": max_all_score,
            "twelfth_token_is_highest": twelfth_token_is_highest,
            "twelfth_token_matches_target": bool(twelfth_token_exists and twelfth_token_index == int(target_idx)),
        }

    target_score = float(scores[target_idx].item())
    max_nonpad_score = float(scores[nonpad_mask].max().item())
    target_is_highest = bool(target_score >= (max_nonpad_score - eps))

    target_node_id = int(flat_nodes[target_idx].item())

    block_id = int(target_idx // Lg)
    center_node: Optional[int] = None

    if center_node_ids is not None:
        c = center_node_ids.detach().cpu().to(torch.long).reshape(-1)
        if block_id < c.numel():
            center_node = int(c[block_id].item())

    if center_node is None:
        block_nodes = graphs[block_id]
        nonpad_block = block_nodes[block_nodes != graph_pad_id]
        if nonpad_block.numel() > 0:
            center_node = int(nonpad_block[0].item())

    if adjacency is None:
        adjacency = _edge_index_to_adjacency(edge_index)

    hop_category = _classify_node_hop(
        center_node=center_node,
        target_node=target_node_id,
        adjacency=adjacency,
    )

    return {
        "valid": True,
        "reason": "ok",
        "target_token_index": int(target_idx),
        "target_node_id": int(target_node_id),
        "target_score": target_score,
        "max_nonpad_score": max_nonpad_score,
        "target_is_highest": target_is_highest,
        "hop_category": hop_category,
        "twelfth_token_exists": twelfth_token_exists,
        "twelfth_token_index": twelfth_token_index,
        "twelfth_token_score": twelfth_token_score,
        "max_all_score": max_all_score,
        "twelfth_token_is_highest": twelfth_token_is_highest,
        "twelfth_token_matches_target": bool(twelfth_token_exists and twelfth_token_index == int(target_idx)),
    }


def summarize_first_postpad_highest_attention(
    *,
    sample_layer_query_to_graph: List[torch.Tensor],
    sample_graphs: List[torch.Tensor],
    sample_key_is_pad: Optional[List[Optional[torch.Tensor]]] = None,
    sample_graph_embs: Optional[List[Optional[torch.Tensor]]] = None,
    edge_index: Optional[torch.Tensor] = None,
    sample_center_node_ids: Optional[List[Optional[torch.Tensor]]] = None,
    graph_pad_id: int = DEFAULT_GRAPH_PAD_ID,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    Dataset-level statistics for the first non-pad token after the first pad group.

    Returns:
      - pct_target_is_highest_over_all_samples
      - pct_target_is_highest_over_valid_samples
      - hop-category percentages among highest-attention hits:
          center / one_hop / two_hop / rest
      - per_sample results for debugging
    """
    n = len(sample_layer_query_to_graph)
    if len(sample_graphs) != n:
        raise ValueError("sample_graphs length must match sample_layer_query_to_graph length.")

    if sample_key_is_pad is None:
        sample_key_is_pad = [None] * n
    elif len(sample_key_is_pad) != n:
        raise ValueError("sample_key_is_pad length must match sample_layer_query_to_graph length.")

    if sample_center_node_ids is None:
        sample_center_node_ids = [None] * n
    elif len(sample_center_node_ids) != n:
        raise ValueError("sample_center_node_ids length must match sample_layer_query_to_graph length.")

    if sample_graph_embs is None:
        sample_graph_embs = [None] * n
    elif len(sample_graph_embs) != n:
        raise ValueError("sample_graph_embs length must match sample_layer_query_to_graph length.")

    adjacency = _edge_index_to_adjacency(edge_index)

    per_sample: List[Dict[str, Any]] = []
    num_valid = 0
    num_highest = 0
    num_samples_with_twelfth_token = 0
    num_twelfth_highest = 0
    num_twelfth_matches_target = 0
    num_cosine_valid = 0
    sum_cosine = 0.0

    hop_counts = {
        "center": 0,
        "one_hop": 0,
        "two_hop": 0,
        "rest": 0,
    }

    for i in range(n):
        res = check_first_postpad_token_highest_attention(
            layer_query_to_graph=sample_layer_query_to_graph[i],
            graphs=sample_graphs[i],
            key_is_pad=sample_key_is_pad[i],
            edge_index=edge_index,
            center_node_ids=sample_center_node_ids[i],
            graph_pad_id=graph_pad_id,
            eps=eps,
            adjacency=adjacency,
        )

        graph_emb = sample_graph_embs[i]
        if graph_emb is not None:
            cosine_res = compute_first_postpad_center_cosine_similarity(
                graphs=sample_graphs[i],
                graph_emb=graph_emb,
                key_is_pad=sample_key_is_pad[i],
                graph_pad_id=graph_pad_id,
                eps=eps,
            )
            res["postpad_center_cosine_valid"] = cosine_res["valid"]
            res["postpad_center_cosine_reason"] = cosine_res["reason"]
            res["postpad_center_cosine_similarity"] = cosine_res["cosine_similarity"]
            res["center_token_index"] = cosine_res["center_token_index"]
            res["center_node_id"] = cosine_res["center_node_id"]
            if cosine_res["valid"]:
                num_cosine_valid += 1
                sum_cosine += float(cosine_res["cosine_similarity"])
        else:
            res["postpad_center_cosine_valid"] = False
            res["postpad_center_cosine_reason"] = "graph_emb_not_provided"
            res["postpad_center_cosine_similarity"] = None

        per_sample.append(res)

        if res["twelfth_token_exists"]:
            num_samples_with_twelfth_token += 1
            if res["twelfth_token_is_highest"]:
                num_twelfth_highest += 1
            if res["twelfth_token_matches_target"]:
                num_twelfth_matches_target += 1

        if not res["valid"]:
            continue

        num_valid += 1
        if res["target_is_highest"]:
            num_highest += 1
            hop = res["hop_category"]
            if hop not in hop_counts:
                hop = "rest"
            hop_counts[hop] += 1

    def pct(num: int, den: int) -> float:
        return 100.0 * float(num) / float(den) if den > 0 else 0.0

    hop_pcts_among_highest = {
        k: pct(v, num_highest) for k, v in hop_counts.items()
    }

    return {
        "num_samples": n,
        "num_valid_samples": num_valid,
        "num_highest_cases": num_highest,
        "num_samples_with_twelfth_token": num_samples_with_twelfth_token,
        "num_twelfth_highest_cases": num_twelfth_highest,
        "num_twelfth_matches_target_cases": num_twelfth_matches_target,
        "pct_target_is_highest_over_all_samples": pct(num_highest, n),
        "pct_target_is_highest_over_valid_samples": pct(num_highest, num_valid),
        "pct_twelfth_token_is_highest_over_all_samples": pct(num_twelfth_highest, n),
        "pct_twelfth_token_is_highest_over_samples_with_twelfth_token": pct(
            num_twelfth_highest, num_samples_with_twelfth_token
        ),
        "pct_twelfth_token_matches_target_over_all_samples": pct(
            num_twelfth_matches_target, n
        ),
        "pct_twelfth_token_matches_target_over_samples_with_twelfth_token": pct(
            num_twelfth_matches_target, num_samples_with_twelfth_token
        ),
        "num_postpad_center_cosine_valid_samples": num_cosine_valid,
        "avg_postpad_center_cosine_similarity": (
            float(sum_cosine) / float(num_cosine_valid) if num_cosine_valid > 0 else 0.0
        ),
        "hop_counts_among_highest": hop_counts,
        "hop_percentages_among_highest": hop_pcts_among_highest,
        "per_sample": per_sample,
    }


def build_first_postpad_sample_record(
    *,
    sample_id: Any,
    layer_query_to_graph: torch.Tensor,
    graphs: torch.Tensor,
    key_is_pad: Optional[torch.Tensor] = None,
    center_node_ids: Optional[torch.Tensor] = None,
    graph_emb: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Build a serializable per-sample record for post-hoc first-post-pad analysis.

    Stores token-level scores (already averaged over layers and queries), not full [L,Q,K].
    If `graph_emb` is provided, also stores the cosine similarity between the first
    post-pad non-pad token and the center token in the same graph block.
    """
    token_scores = _token_scores_from_layer_query_to_graph(
        layer_query_to_graph
    ).detach().cpu().to(torch.float32)

    if graphs.dim() == 1:
        graphs = graphs.unsqueeze(0)
    graphs_cpu = graphs.detach().cpu().to(torch.long)

    rec: Dict[str, Any] = {
        "sample_id": sample_id,
        "token_scores": token_scores.tolist(),
        "graphs": graphs_cpu.tolist(),
    }

    if key_is_pad is not None:
        rec["key_is_pad"] = key_is_pad.detach().cpu().bool().reshape(-1).tolist()
    if center_node_ids is not None:
        rec["center_node_ids"] = (
            center_node_ids.detach().cpu().to(torch.long).reshape(-1).tolist()
        )

    if graph_emb is not None:
        cosine_res = compute_first_postpad_center_cosine_similarity(
            graphs=graphs_cpu,
            graph_emb=graph_emb,
            key_is_pad=key_is_pad,
        )
        rec["postpad_center_cosine_valid"] = bool(cosine_res["valid"])
        rec["postpad_center_cosine_reason"] = cosine_res["reason"]
        rec["postpad_center_cosine_similarity"] = cosine_res["cosine_similarity"]
        rec["center_token_index"] = cosine_res["center_token_index"]
        rec["center_node_id"] = cosine_res["center_node_id"]

    return rec


def append_first_postpad_sample_record(
    *,
    record: Dict[str, Any],
    save_path: str,
) -> None:
    """
    Append one per-sample record as a JSONL line.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def summarize_jsonl(
    *,
    records_path: str,
    edge_index: Optional[torch.Tensor] = None,
    graph_pad_id: int = DEFAULT_GRAPH_PAD_ID,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    Post-hoc summary directly from JSONL records produced by
    build_first_postpad_sample_record + append_first_postpad_sample_record.
    """
    if not os.path.exists(records_path):
        raise FileNotFoundError(f"records_path does not exist: {records_path}")

    adjacency = _edge_index_to_adjacency(edge_index)

    per_sample: List[Dict[str, Any]] = []
    num_samples = 0
    num_valid = 0
    num_highest = 0
    num_samples_with_twelfth_token = 0
    num_twelfth_highest = 0
    num_twelfth_matches_target = 0
    num_cosine_valid = 0
    sum_cosine = 0.0

    hop_counts = {
        "center": 0,
        "one_hop": 0,
        "two_hop": 0,
        "rest": 0,
    }

    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num_samples += 1
            rec = json.loads(line)

            token_scores = torch.tensor(
                rec["token_scores"], dtype=torch.float32
            )  # [K]
            graphs = torch.tensor(rec["graphs"], dtype=torch.long)  # [G, Lg]

            key_is_pad = None
            if "key_is_pad" in rec and rec["key_is_pad"] is not None:
                key_is_pad = torch.tensor(rec["key_is_pad"], dtype=torch.bool)

            center_node_ids = None
            if "center_node_ids" in rec and rec["center_node_ids"] is not None:
                center_node_ids = torch.tensor(rec["center_node_ids"], dtype=torch.long)

            res = check_first_postpad_token_highest_attention(
                layer_query_to_graph=token_scores,  # [K] supported
                graphs=graphs,
                key_is_pad=key_is_pad,
                edge_index=edge_index,
                center_node_ids=center_node_ids,
                graph_pad_id=graph_pad_id,
                eps=eps,
                adjacency=adjacency,
            )
            if "sample_id" in rec:
                res["sample_id"] = rec["sample_id"]

            if "postpad_center_cosine_valid" in rec:
                res["postpad_center_cosine_valid"] = bool(rec["postpad_center_cosine_valid"])
                res["postpad_center_cosine_reason"] = rec.get(
                    "postpad_center_cosine_reason", "ok"
                )
                res["postpad_center_cosine_similarity"] = rec.get(
                    "postpad_center_cosine_similarity"
                )
                res["center_token_index"] = rec.get("center_token_index")
                res["center_node_id"] = rec.get("center_node_id")
                if (
                    res["postpad_center_cosine_valid"]
                    and res["postpad_center_cosine_similarity"] is not None
                ):
                    num_cosine_valid += 1
                    sum_cosine += float(res["postpad_center_cosine_similarity"])
            per_sample.append(res)

            if res["twelfth_token_exists"]:
                num_samples_with_twelfth_token += 1
                if res["twelfth_token_is_highest"]:
                    num_twelfth_highest += 1
                if res["twelfth_token_matches_target"]:
                    num_twelfth_matches_target += 1

            if not res["valid"]:
                continue

            num_valid += 1
            if res["target_is_highest"]:
                num_highest += 1
                hop = res["hop_category"]
                if hop not in hop_counts:
                    hop = "rest"
                hop_counts[hop] += 1

    def pct(num: int, den: int) -> float:
        return 100.0 * float(num) / float(den) if den > 0 else 0.0

    hop_pcts_among_highest = {
        k: pct(v, num_highest) for k, v in hop_counts.items()
    }

    return {
        "num_samples": num_samples,
        "num_valid_samples": num_valid,
        "num_highest_cases": num_highest,
        "num_samples_with_twelfth_token": num_samples_with_twelfth_token,
        "num_twelfth_highest_cases": num_twelfth_highest,
        "num_twelfth_matches_target_cases": num_twelfth_matches_target,
        "pct_target_is_highest_over_all_samples": pct(num_highest, num_samples),
        "pct_target_is_highest_over_valid_samples": pct(num_highest, num_valid),
        "pct_twelfth_token_is_highest_over_all_samples": pct(
            num_twelfth_highest, num_samples
        ),
        "pct_twelfth_token_is_highest_over_samples_with_twelfth_token": pct(
            num_twelfth_highest, num_samples_with_twelfth_token
        ),
        "pct_twelfth_token_matches_target_over_all_samples": pct(
            num_twelfth_matches_target, num_samples
        ),
        "pct_twelfth_token_matches_target_over_samples_with_twelfth_token": pct(
            num_twelfth_matches_target, num_samples_with_twelfth_token
        ),
        "num_postpad_center_cosine_valid_samples": num_cosine_valid,
        "avg_postpad_center_cosine_similarity": (
            float(sum_cosine) / float(num_cosine_valid) if num_cosine_valid > 0 else 0.0
        ),
        "hop_counts_among_highest": hop_counts,
        "hop_percentages_among_highest": hop_pcts_among_highest,
        "per_sample": per_sample,
    }


def _relative_bins(length: int, n_bins: int) -> torch.Tensor:
    """
    Map positions 0..length-1 to relative bins 0..n_bins-1.
    """
    if length <= 0:
        return torch.zeros(0, dtype=torch.long)
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0.")
    if length == 1:
        return torch.zeros(1, dtype=torch.long)

    rel = torch.arange(length, dtype=torch.float32) / float(length - 1)
    bins = torch.clamp((rel * (n_bins - 1)).round().to(torch.long), 0, n_bins - 1)
    return bins


def aggregate_token_scores_by_relative_position(
    *,
    sample_token_scores: List[torch.Tensor],
    sample_graphs: List[torch.Tensor],
    sample_key_is_pad: Optional[List[Optional[torch.Tensor]]] = None,
    graph_pad_id: int = DEFAULT_GRAPH_PAD_ID,
    n_bins: int = 64,
) -> Dict[str, Any]:
    """
    Aggregate per-sample token scores into dataset-level relative-position bins.

    This is robust to different graph-token lengths and pad positions across samples.
    """
    n = len(sample_token_scores)
    if len(sample_graphs) != n:
        raise ValueError("sample_graphs length must match sample_token_scores length.")

    if sample_key_is_pad is None:
        sample_key_is_pad = [None] * n
    elif len(sample_key_is_pad) != n:
        raise ValueError("sample_key_is_pad length must match sample_token_scores length.")

    sum_all = torch.zeros(n_bins, dtype=torch.float32)
    cnt_all = torch.zeros(n_bins, dtype=torch.long)
    sum_nonpad = torch.zeros(n_bins, dtype=torch.float32)
    cnt_nonpad = torch.zeros(n_bins, dtype=torch.long)
    sum_pad = torch.zeros(n_bins, dtype=torch.float32)
    cnt_pad = torch.zeros(n_bins, dtype=torch.long)

    num_valid_samples = 0
    per_sample_meta: List[Dict[str, Any]] = []

    for i in range(n):
        scores = sample_token_scores[i].detach().cpu().to(torch.float32).reshape(-1)

        g = sample_graphs[i]
        if g.dim() == 1:
            g = g.unsqueeze(0)
        g = g.detach().cpu().to(torch.long)
        flat_nodes = g.reshape(-1)

        key_is_pad = sample_key_is_pad[i]
        if key_is_pad is not None:
            key_is_pad = key_is_pad.detach().cpu().bool().reshape(-1)
            if key_is_pad.numel() != scores.numel():
                per_sample_meta.append(
                    {
                        "sample_index": i,
                        "valid": False,
                        "reason": "key_is_pad_length_mismatch",
                    }
                )
                continue
        else:
            if flat_nodes.numel() != scores.numel():
                per_sample_meta.append(
                    {
                        "sample_index": i,
                        "valid": False,
                        "reason": "graphs_length_mismatch_without_key_is_pad",
                    }
                )
                continue
            key_is_pad = flat_nodes == graph_pad_id

        if flat_nodes.numel() != scores.numel():
            per_sample_meta.append(
                {
                    "sample_index": i,
                    "valid": False,
                    "reason": "graphs_and_scores_not_aligned",
                }
            )
            continue

        bins = _relative_bins(scores.numel(), n_bins=n_bins)
        one = torch.ones_like(bins, dtype=torch.long)

        sum_all.scatter_add_(0, bins, scores)
        cnt_all.scatter_add_(0, bins, one)

        nonpad_mask = ~key_is_pad
        if bool(nonpad_mask.any()):
            bins_nonpad = bins[nonpad_mask]
            scores_nonpad = scores[nonpad_mask]
            sum_nonpad.scatter_add_(0, bins_nonpad, scores_nonpad)
            cnt_nonpad.scatter_add_(0, bins_nonpad, torch.ones_like(bins_nonpad, dtype=torch.long))

        if bool(key_is_pad.any()):
            bins_pad = bins[key_is_pad]
            scores_pad = scores[key_is_pad]
            sum_pad.scatter_add_(0, bins_pad, scores_pad)
            cnt_pad.scatter_add_(0, bins_pad, torch.ones_like(bins_pad, dtype=torch.long))

        num_valid_samples += 1
        per_sample_meta.append(
            {
                "sample_index": i,
                "valid": True,
                "num_tokens": int(scores.numel()),
                "num_pad_tokens": int(key_is_pad.sum().item()),
                "num_nonpad_tokens": int((~key_is_pad).sum().item()),
            }
        )

    eps = 1e-12
    mean_all = sum_all / (cnt_all.to(torch.float32) + eps)
    mean_nonpad = sum_nonpad / (cnt_nonpad.to(torch.float32) + eps)
    mean_pad = sum_pad / (cnt_pad.to(torch.float32) + eps)

    return {
        "n_samples": n,
        "n_valid_samples": num_valid_samples,
        "n_bins": n_bins,
        "bin_centers": torch.linspace(0.0, 1.0, steps=n_bins),
        "all_mean": mean_all,
        "all_count": cnt_all,
        "nonpad_mean": mean_nonpad,
        "nonpad_count": cnt_nonpad,
        "pad_mean": mean_pad,
        "pad_count": cnt_pad,
        "per_sample_meta": per_sample_meta,
    }


def aggregate_token_scores_by_relative_position_from_jsonl(
    *,
    records_path: str,
    graph_pad_id: int = DEFAULT_GRAPH_PAD_ID,
    n_bins: int = 64,
) -> Dict[str, Any]:
    """
    Aggregate relative-position token scores from JSONL records created by
    build_first_postpad_sample_record + append_first_postpad_sample_record.
    """
    if not os.path.exists(records_path):
        raise FileNotFoundError(f"records_path does not exist: {records_path}")

    sample_token_scores: List[torch.Tensor] = []
    sample_graphs: List[torch.Tensor] = []
    sample_key_is_pad: List[Optional[torch.Tensor]] = []
    sample_ids: List[Any] = []

    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sample_ids.append(rec.get("sample_id"))
            sample_token_scores.append(torch.tensor(rec["token_scores"], dtype=torch.float32))
            sample_graphs.append(torch.tensor(rec["graphs"], dtype=torch.long))

            if "key_is_pad" in rec and rec["key_is_pad"] is not None:
                sample_key_is_pad.append(torch.tensor(rec["key_is_pad"], dtype=torch.bool))
            else:
                sample_key_is_pad.append(None)

    agg = aggregate_token_scores_by_relative_position(
        sample_token_scores=sample_token_scores,
        sample_graphs=sample_graphs,
        sample_key_is_pad=sample_key_is_pad,
        graph_pad_id=graph_pad_id,
        n_bins=n_bins,
    )
    agg["sample_ids"] = sample_ids
    agg["records_path"] = records_path
    return agg


def plot_dataset_relative_position_attention(
    *,
    aggregated: Dict[str, Any],
    save_path: str,
    include_splits: bool = True,
    cmap: str = "viridis",
    dpi: int = 180,
) -> str:
    """
    Plot dataset-level attention vs relative key position.

    If include_splits=True, plots 3 rows:
      1) all tokens
      2) non-pad tokens
      3) pad tokens
    Else plots only all tokens.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it to save aggregated heatmaps."
        ) from e

    all_mean = aggregated["all_mean"].detach().cpu().to(torch.float32)
    n_bins = int(aggregated["n_bins"])
    if all_mean.numel() != n_bins:
        raise ValueError("aggregated['all_mean'] size does not match n_bins.")

    if include_splits:
        mat = torch.stack(
            [
                aggregated["all_mean"].detach().cpu().to(torch.float32),
                aggregated["nonpad_mean"].detach().cpu().to(torch.float32),
                aggregated["pad_mean"].detach().cpu().to(torch.float32),
            ],
            dim=0,
        )  # [3, n_bins]
        y_labels = ["all", "nonpad", "pad"]
    else:
        mat = all_mean.unsqueeze(0)  # [1, n_bins]
        y_labels = ["all"]

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig_w = max(8.0, 0.14 * n_bins)
    fig_h = max(2.5, 1.3 * mat.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    im = ax.imshow(
        mat.numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
    )

    ax.set_title(
        f"Dataset Relative-Position Attention (valid={aggregated['n_valid_samples']}/{aggregated['n_samples']})"
    )
    ax.set_xlabel("Relative key position bin (0=start, 1=end)")
    ax.set_ylabel("Token group")

    x_ticks = _choose_ticks(n_bins, max_ticks=16)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x/(n_bins-1):.2f}" if n_bins > 1 else "0.00" for x in x_ticks])
    ax.set_yticks(list(range(len(y_labels))))
    ax.set_yticklabels(y_labels)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean attention score")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def aggregate_dataset_average_cross_attention(
    *,
    sample_layer_query_to_graph: List[torch.Tensor],
) -> Dict[str, Any]:
    """
    Aggregate per-sample query->graph attention heatmaps into one dataset-level heatmap.

    Each sample may have a different number of query tokens (Q) and graph key tokens
    (K). We first average each sample over layers to get one [Q, K] matrix, then
    align matrices at the top-left corner and compute an elementwise mean over the
    samples that contain each position.
    """
    per_sample_meta: List[Dict[str, Any]] = []
    sample_avg_mats: List[torch.Tensor] = []
    max_q = 0
    max_k = 0

    for i, layer_query_to_graph in enumerate(sample_layer_query_to_graph):
        if layer_query_to_graph is None:
            per_sample_meta.append(
                {
                    "sample_index": i,
                    "valid": False,
                    "reason": "none_sample",
                }
            )
            continue

        mat = layer_query_to_graph.detach().cpu().to(torch.float32)
        if mat.dim() == 3:
            mat = mat.mean(dim=0)
        elif mat.dim() != 2:
            per_sample_meta.append(
                {
                    "sample_index": i,
                    "valid": False,
                    "reason": f"unexpected_dim_{mat.dim()}",
                }
            )
            continue

        q_len, k_len = mat.shape
        if q_len == 0 or k_len == 0:
            per_sample_meta.append(
                {
                    "sample_index": i,
                    "valid": False,
                    "reason": "empty_heatmap",
                }
            )
            continue

        sample_avg_mats.append(mat)
        max_q = max(max_q, q_len)
        max_k = max(max_k, k_len)
        per_sample_meta.append(
            {
                "sample_index": i,
                "valid": True,
                "q_len": int(q_len),
                "k_len": int(k_len),
            }
        )

    if not sample_avg_mats:
        raise ValueError("No valid sample attention heatmaps were provided.")

    sum_mat = torch.zeros((max_q, max_k), dtype=torch.float32)
    count_mat = torch.zeros((max_q, max_k), dtype=torch.long)

    for mat in sample_avg_mats:
        q_len, k_len = mat.shape
        sum_mat[:q_len, :k_len] += mat
        count_mat[:q_len, :k_len] += 1

    mean_mat = sum_mat / count_mat.clamp_min(1).to(torch.float32)

    return {
        "n_samples": len(sample_layer_query_to_graph),
        "n_valid_samples": len(sample_avg_mats),
        "avg_query_to_graph": mean_mat,
        "count_query_to_graph": count_mat,
        "per_sample_meta": per_sample_meta,
    }


def plot_dataset_average_cross_attention_heatmap(
    *,
    aggregated: Dict[str, Any],
    save_path: str,
    cmap: str = "viridis",
    dpi: int = 180,
    max_yticks: int = 16,
) -> str:
    """
    Plot the dataset-level average query->graph attention heatmap.

    This uses the same visual layout as the per-sample layer-averaged plot:
      - y-axis: query token indices
      - x-axis: graph key token indices
      - color: average attention weight
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it to save attention heatmaps."
        ) from e

    avg_mat = aggregated["avg_query_to_graph"].detach().cpu().to(torch.float32)
    if avg_mat.dim() != 2:
        raise ValueError("aggregated['avg_query_to_graph'] must have shape [Q, K].")

    q_len, k_len = avg_mat.shape
    if q_len == 0 or k_len == 0:
        raise ValueError("aggregated['avg_query_to_graph'] is empty.")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig_w = max(6.0, 0.14 * k_len)
    fig_h = max(4.0, 0.14 * q_len)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    im = ax.imshow(
        avg_mat.numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
    )

    ax.set_title(
        f"Dataset Avg Cross Attention (valid={aggregated['n_valid_samples']}/{aggregated['n_samples']})"
    )
    ax.set_xlabel("Graph key tokens")
    ax.set_ylabel("Query tokens")

    x_ticks = list(range(k_len))
    y_ticks = _choose_ticks(q_len, max_yticks)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks], rotation=90, ha="center")
    ax.set_yticklabels([str(y) for y in y_ticks])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attention weight (head-avg, layer-avg, sample-avg)")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_final_average_cross_attention_for_dataset(
    *,
    sample_layer_query_to_graph: List[torch.Tensor],
    save_path: str,
    cmap: str = "viridis",
    dpi: int = 180,
    max_yticks: int = 16,
) -> Dict[str, Any]:
    """
    One-call helper to generate the final dataset-level average cross-attention heatmap.

    Unlike the relative-position summary, this keeps the same [query, graph-token]
    heatmap structure as the per-sample layer-averaged plots and averages those
    heatmaps across samples.
    """
    aggregated = aggregate_dataset_average_cross_attention(
        sample_layer_query_to_graph=sample_layer_query_to_graph,
    )
    plot_path = plot_dataset_average_cross_attention_heatmap(
        aggregated=aggregated,
        save_path=save_path,
        cmap=cmap,
        dpi=dpi,
        max_yticks=max_yticks,
    )
    aggregated["plot_path"] = plot_path
    return aggregated


def aggregate_layer_vs_graph_attention(
    sample_layer_to_graph: List[torch.Tensor],
) -> Dict[str, Any]:
    """
    Aggregate per-sample [L, K] attention (already query-averaged) into a [L, K]
    matrix averaged over samples. Top-left aligned to handle variable K.
    """
    if len(sample_layer_to_graph) == 0:
        raise ValueError("sample_layer_to_graph is empty.")
    L = int(sample_layer_to_graph[0].shape[0])
    K_max = int(max(t.shape[1] for t in sample_layer_to_graph))
    sum_lk = torch.zeros((L, K_max), dtype=torch.float32)
    cnt_k = torch.zeros((K_max,), dtype=torch.long)
    for t in sample_layer_to_graph:
        t = t.detach().to(torch.float32).cpu()
        K = int(t.shape[1])
        sum_lk[:, :K] += t
        cnt_k[:K] += 1
    cnt_safe = cnt_k.clamp(min=1).to(torch.float32)
    mean_lk = sum_lk / cnt_safe.unsqueeze(0)
    return {
        "mean_layer_to_graph": mean_lk,   # [L, K]
        "column_count": cnt_k,            # [K]
        "n_samples": int(len(sample_layer_to_graph)),
    }


def plot_layer_vs_graph_attention_heatmap(
    *,
    aggregated: Dict[str, Any],
    save_path: str,
    title: Optional[str] = None,
    cmap: str = "viridis",
    dpi: int = 180,
) -> str:
    """
    Plot a [L, K] heatmap. y-axis: transformer layers (L1..Lnum_layers).
    x-axis: graph token indices. Color: head-avg, query-avg, sample-avg attention.
    """
    mean_lk = aggregated["mean_layer_to_graph"].detach().cpu().numpy()
    L, K = mean_lk.shape
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig_w = max(8.0, min(0.18 * K + 4.0, 24.0))
    fig_h = max(4.0, min(0.22 * L + 2.0, 12.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    im = ax.imshow(mean_lk, aspect="auto", cmap=cmap, origin="upper")
    ax.set_xlabel("Graph token index")
    ax.set_ylabel("Transformer layer")
    ax.set_xticks(range(K))
    ax.set_xticklabels([str(i) for i in range(K)], fontsize=7)
    ax.set_yticks(range(L))
    ax.set_yticklabels([f"L{i+1}" for i in range(L)], fontsize=7)
    if title is None:
        title = (
            f"per-layer mean attention to graph tokens "
            f"(n={aggregated['n_samples']} samples, head-avg, query-avg, sample-avg)"
        )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


############# Average Attention Probing on Generated Texts ###################

def _get_generated_step_attentions(generate_outputs) -> Tuple[Tuple[torch.Tensor, ...], ...]:
    """
    Extract decoding-step attentions from HF generate outputs, excluding step-0 prompt attention.

    Expected structure:
      - tuple over generation steps
        - each item is tuple over layers
          - each tensor is [B, H, Q, K]

    In your current setup, attns[0] is the prompt step, and attns[1:] are the
    generated-token decoding steps. Each decoding step usually has Q == 1.
    """
    attns = getattr(generate_outputs, "attentions", None)
    if attns is None:
        raise ValueError(
            "generate_outputs.attentions is None. Set output_attentions=True and return_dict_in_generate=True."
        )

    if not isinstance(attns, (tuple, list)) or len(attns) == 0:
        raise ValueError("Unexpected attentions structure from generate().")

    if len(attns) <= 1:
        return tuple()

    gen_steps = attns[1:]
    if not isinstance(gen_steps[0], (tuple, list)) or len(gen_steps[0]) == 0:
        raise ValueError("Unexpected generated-step attentions structure from generate().")

    return tuple(gen_steps)


@torch.no_grad()
def compute_layerwise_generated_query_to_graph_attention(
    *,
    generate_outputs,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    graphs: torch.Tensor,
    keep_pad_tokens: bool = True,
    mm_use_graph_special_token: bool = False,
    use_hop: Optional[int] = None,
    sample_neighbor_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Per sample, compute per-layer generated-token -> graph attention matrices averaged over heads.

    This uses only generated decoding steps, not the prompt step.

    Output entries per sample:
      - layer_generated_query_to_graph: [num_layers, S, K]
          S = number of generated tokens
          K = number of graph key tokens
      - layer_generated_to_graph_avg: [num_layers, K]
          averaged across generated tokens within the sample
      - key_idx: [K]
      - key_idx_all: all expanded graph key indices (with pads)
      - key_is_pad: [K] bool
      - num_generated_tokens: int

    No cross-sample aggregation is performed.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    # We still use the prompt step only to recover prompt length / graph positions.
    attns0 = _get_prompt_step_attentions(generate_outputs)
    gen_attns = _get_generated_step_attentions(generate_outputs)

    num_layers = len(attns0)
    batch_size = attns0[0].shape[0]
    prompt_len = attns0[0].shape[-1]
    num_generated_steps = len(gen_attns)

    if graphs is not None and graphs.dim() == 3:
        graphs_batched = graphs
    else:
        graphs_batched = graphs.unsqueeze(0).expand(batch_size, -1, -1)

    valid: List[bool] = []
    layer_generated_query_to_graph_list: List[Optional[torch.Tensor]] = []
    layer_generated_to_graph_avg_list: List[Optional[torch.Tensor]] = []
    key_idx_list: List[Optional[torch.Tensor]] = []
    key_idx_all_list: List[Optional[torch.Tensor]] = []
    key_is_pad_list: List[Optional[torch.Tensor]] = []
    num_generated_tokens_list: List[int] = []

    for b in range(batch_size):
        idxs = get_expanded_graph_key_query_indices(
            input_ids_1d=input_ids[b],
            attention_mask_1d=attention_mask[b],
            graphs=graphs_batched[b],
            prompt_len=prompt_len,
            keep_pad_tokens=keep_pad_tokens,
            mm_use_graph_special_token=mm_use_graph_special_token,
            use_hop=use_hop,
            sample_neighbor_size=sample_neighbor_size,
        )

        if idxs is None:
            valid.append(False)
            layer_generated_query_to_graph_list.append(None)
            layer_generated_to_graph_avg_list.append(None)
            key_idx_list.append(None)
            key_idx_all_list.append(None)
            key_is_pad_list.append(None)
            num_generated_tokens_list.append(0)
            continue

        key_idx, _, key_idx_all, key_is_pad = idxs

        if num_generated_steps == 0:
            valid.append(False)
            layer_generated_query_to_graph_list.append(None)
            layer_generated_to_graph_avg_list.append(None)
            key_idx_list.append(key_idx)
            key_idx_all_list.append(key_idx_all)
            key_is_pad_list.append(key_is_pad)
            num_generated_tokens_list.append(0)
            continue

        per_layer_steps: List[torch.Tensor] = []
        valid_sample = True

        for layer_id in range(num_layers):
            step_vecs: List[torch.Tensor] = []

            for step_id in range(num_generated_steps):
                # [B, H, Q, K_total] -> sample b: [H, Q, K_total]
                layer_attn_b = gen_attns[step_id][layer_id][b].to(torch.float32)   # gen_attns: tuple of five generated tokens (each with shape )

                if layer_attn_b.dim() != 3:
                    valid_sample = False
                    break

                # Use the newest generated token as the query.
                # With cache this is usually Q == 1, but taking the last query is safer.
                q_last = layer_attn_b[:, -1, :]  # [H, K_total]

                if key_idx.numel() == 0 or int(key_idx.max().item()) >= q_last.shape[-1]:
                    valid_sample = False
                    break

                # Keep only graph-token keys: [H, K]
                q_to_graph = q_last.index_select(1, key_idx)

                # Head average -> [K]
                q_to_graph_head_avg = q_to_graph.mean(dim=0)
                step_vecs.append(q_to_graph_head_avg)

            if not valid_sample:
                break

            # [S, K]
            layer_step_mat = torch.stack(step_vecs, dim=0)
            per_layer_steps.append(layer_step_mat)

        if not valid_sample or len(per_layer_steps) == 0:
            valid.append(False)
            layer_generated_query_to_graph_list.append(None)
            layer_generated_to_graph_avg_list.append(None)
            key_idx_list.append(key_idx)
            key_idx_all_list.append(key_idx_all)
            key_is_pad_list.append(key_is_pad)
            num_generated_tokens_list.append(0)
            continue

        # [L, S, K]
        layer_generated_query_to_graph = torch.stack(per_layer_steps, dim=0)

        # average across generated tokens -> [L, K]
        layer_generated_to_graph_avg = layer_generated_query_to_graph.mean(dim=1)

        valid.append(True)
        layer_generated_query_to_graph_list.append(layer_generated_query_to_graph)
        layer_generated_to_graph_avg_list.append(layer_generated_to_graph_avg)
        key_idx_list.append(key_idx)
        key_idx_all_list.append(key_idx_all)
        key_is_pad_list.append(key_is_pad)
        num_generated_tokens_list.append(int(layer_generated_query_to_graph.shape[1]))

    return {
        "valid": valid,
        "prompt_len": prompt_len,
        "num_generated_steps_total": num_generated_steps,
        "layer_generated_query_to_graph": layer_generated_query_to_graph_list,   # [L,S,K]
        "layer_generated_to_graph_avg": layer_generated_to_graph_avg_list,       # [L,K]
        "key_idx": key_idx_list,
        "key_idx_all": key_idx_all_list,
        "key_is_pad": key_is_pad_list,
        "num_generated_tokens": num_generated_tokens_list,
    }


def aggregate_generated_attention_by_graph_position(
    *,
    sample_layer_generated_query_to_graph: List[torch.Tensor],
    sample_key_is_pad: Optional[List[Optional[torch.Tensor]]] = None,
) -> Dict[str, Any]:
    """
    Aggregate generated-token -> graph attention into one dataset-level vector over graph key positions.

    Each sample tensor is expected to have shape [L, S, K]:
      L = number of layers
      S = number of generated tokens
      K = number of graph key tokens

    For each sample, this function averages over:
      - layers
      - generated tokens

    Then it averages across samples position-wise.

    It also tracks how often each graph-key position is padded across samples.
    """
    n = len(sample_layer_generated_query_to_graph)
    if n == 0:
        raise ValueError("sample_layer_generated_query_to_graph must be non-empty.")

    if sample_key_is_pad is None:
        sample_key_is_pad = [None] * n
    elif len(sample_key_is_pad) != n:
        raise ValueError(
            "sample_key_is_pad length must match sample_layer_generated_query_to_graph length."
        )

    max_k = 0
    num_valid_samples = 0
    per_sample_meta: List[Dict[str, Any]] = []

    sample_vecs: List[torch.Tensor] = []
    sample_pad_masks: List[Optional[torch.Tensor]] = []

    for i, x in enumerate(sample_layer_generated_query_to_graph):
        if x is None:
            per_sample_meta.append(
                {"sample_index": i, "valid": False, "reason": "none_sample"}
            )
            continue

        x = x.detach().cpu().to(torch.float32)
        if x.dim() != 3:
            per_sample_meta.append(
                {"sample_index": i, "valid": False, "reason": f"unexpected_dim_{x.dim()}"}
            )
            continue

        if x.shape[1] == 0 or x.shape[2] == 0:
            per_sample_meta.append(
                {"sample_index": i, "valid": False, "reason": "empty_generated_or_key_axis"}
            )
            continue

        # Average over layers and generated tokens -> [K]
        vec = x.mean(dim=(0, 1))
        k_len = int(vec.shape[0])

        pad_mask = sample_key_is_pad[i]
        if pad_mask is not None:
            pad_mask = pad_mask.detach().cpu().bool().reshape(-1)
            if pad_mask.numel() != k_len:
                per_sample_meta.append(
                    {"sample_index": i, "valid": False, "reason": "key_is_pad_length_mismatch"}
                )
                continue

        sample_vecs.append(vec)
        sample_pad_masks.append(pad_mask)
        max_k = max(max_k, k_len)
        num_valid_samples += 1
        per_sample_meta.append(
            {"sample_index": i, "valid": True, "k_len": k_len}
        )

    if len(sample_vecs) == 0:
        raise ValueError("No valid generated-attention samples were provided.")

    sum_vec = torch.zeros(max_k, dtype=torch.float32)
    cnt_vec = torch.zeros(max_k, dtype=torch.long)

    pad_count = torch.zeros(max_k, dtype=torch.long)
    pad_known_count = torch.zeros(max_k, dtype=torch.long)

    for vec, pad_mask in zip(sample_vecs, sample_pad_masks):
        k_len = vec.shape[0]
        sum_vec[:k_len] += vec
        cnt_vec[:k_len] += 1

        if pad_mask is not None:
            pad_count[:k_len] += pad_mask.to(torch.long)
            pad_known_count[:k_len] += 1

    mean_vec = sum_vec / cnt_vec.clamp_min(1).to(torch.float32)

    # Fraction of samples in which this position is padded, among samples where pad info is known.
    pad_fraction = pad_count.to(torch.float32) / pad_known_count.clamp_min(1).to(torch.float32)

    return {
        "n_samples": n,
        "n_valid_samples": num_valid_samples,
        "avg_attention_by_key_position": mean_vec,   # [K_max]
        "count_by_key_position": cnt_vec,            # [K_max]
        "pad_fraction_by_key_position": pad_fraction,  # [K_max]
        "pad_count_by_key_position": pad_count,
        "pad_known_count_by_key_position": pad_known_count,
        "per_sample_meta": per_sample_meta,
    }


def plot_generated_attention_histogram(
    *,
    aggregated: Dict[str, Any],
    save_path: str,
    pad_threshold: float = 0.5,
    dpi: int = 180,
) -> str:
    """
    Plot a dataset-level bar plot over graph key positions.

    x-axis: graph key token position
    y-axis: average attention weight, already averaged across:
      - heads
      - layers
      - generated tokens
      - samples

    Tick labels append 'P' when the position is padded in at least `pad_threshold`
    fraction of valid samples with known pad masks.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it to save histogram plots."
        ) from e

    avg_vec = aggregated["avg_attention_by_key_position"].detach().cpu().to(torch.float32)
    pad_frac = aggregated["pad_fraction_by_key_position"].detach().cpu().to(torch.float32)
    cnt_vec = aggregated["count_by_key_position"].detach().cpu().to(torch.long)

    if avg_vec.dim() != 1:
        raise ValueError("aggregated['avg_attention_by_key_position'] must have shape [K].")

    k_len = int(avg_vec.numel())
    if k_len == 0:
        raise ValueError("aggregated['avg_attention_by_key_position'] is empty.")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    x = torch.arange(k_len, dtype=torch.long)
    x_np = x.numpy()
    y_np = avg_vec.numpy()

    fig_w = max(10.0, 0.28 * k_len)
    fig_h = 4.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    ax.bar(x_np, y_np)

    ax.set_title(
        f"Dataset Avg Generated->Graph Attention "
        f"(valid={aggregated['n_valid_samples']}/{aggregated['n_samples']})"
    )
    ax.set_xlabel("Graph key token position")
    ax.set_ylabel("Average attention weight")

    tick_positions = list(range(k_len))
    tick_labels = []
    for i in tick_positions:
        label = str(i)
        if float(pad_frac[i].item()) >= pad_threshold:
            label += "P"
        tick_labels.append(label)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, ha="center")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def build_top2_sink_attention_nonpad_record(
    *,
    sample_id: Any,
    graphs: torch.Tensor,
    top2_sink_prompt_positions: List[int],
    layer_query_to_graph: Optional[torch.Tensor] = None,
    key_idx: Optional[torch.Tensor] = None,
    key_is_pad: Optional[torch.Tensor] = None,
    graph_pad_id: int = DEFAULT_GRAPH_PAD_ID,
) -> Dict[str, Any]:
    '''
    Build one per-sample record for:
      - x: percentage of non-padded graph tokens
      - y: average attention weight of the top-2 sink tokens

    The attention scalar is averaged across:
      - layers
      - query tokens
      - the found top-2 sink-token columns
    '''
    if graphs.dim() == 1:
        graphs = graphs.unsqueeze(0)

    graphs_cpu = graphs.detach().cpu().to(torch.long)
    num_graph_tokens = int(graphs_cpu.numel())
    num_nonpad_graph_tokens = int((graphs_cpu != graph_pad_id).sum().item())
    nonpad_fraction = (
        float(num_nonpad_graph_tokens) / float(num_graph_tokens)
        if num_graph_tokens > 0
        else None
    )
    nonpad_percentage = 100.0 * nonpad_fraction if nonpad_fraction is not None else None

    requested_positions = sorted({int(pos) for pos in top2_sink_prompt_positions})
    record: Dict[str, Any] = {
        'sample_id': sample_id,
        'valid': False,
        'reason': None,
        'num_graph_tokens': num_graph_tokens,
        'num_nonpad_graph_tokens': num_nonpad_graph_tokens,
        'nonpad_fraction': nonpad_fraction,
        'nonpad_percentage': nonpad_percentage,
        'top2_sink_prompt_positions_requested': requested_positions,
        'top2_sink_prompt_positions_found': [],
        'missing_top2_sink_prompt_positions': requested_positions,
        'num_top2_sink_tokens_found': 0,
        'top2_sink_is_pad': None,
        'top2_sink_token_mean_attentions': None,
        'top2_sink_avg_attention': None,
    }

    if len(requested_positions) == 0:
        record['reason'] = 'no_top2_sink_prompt_positions'
        return record

    if layer_query_to_graph is None or key_idx is None:
        record['reason'] = 'missing_attention_inputs'
        return record

    selected = extract_sink_columns_from_query_graph_attention(
        layer_query_to_graph=layer_query_to_graph,
        key_idx=key_idx,
        sink_prompt_positions=requested_positions,
        key_is_pad=key_is_pad,
    )

    found_positions = []
    if selected['sink_key_idx'] is not None:
        found_positions = (
            selected['sink_key_idx'].detach().cpu().to(torch.long).reshape(-1).tolist()
        )

    record['top2_sink_prompt_positions_found'] = found_positions
    record['missing_top2_sink_prompt_positions'] = selected['missing_sink_prompt_positions']
    record['num_top2_sink_tokens_found'] = len(found_positions)

    if not bool(selected['valid']):
        record['reason'] = selected['reason']
        return record

    layer_query_to_top2 = selected['layer_query_to_sink'].detach().cpu().to(torch.float32)
    top2_token_mean_attentions = layer_query_to_top2.mean(dim=(0, 1))

    top2_sink_is_pad = None
    if selected['sink_key_is_pad'] is not None:
        top2_sink_is_pad = (
            selected['sink_key_is_pad'].detach().cpu().bool().reshape(-1).tolist()
        )

    record['valid'] = True
    record['reason'] = 'ok'
    record['top2_sink_is_pad'] = top2_sink_is_pad
    record['top2_sink_token_mean_attentions'] = top2_token_mean_attentions.tolist()
    record['top2_sink_avg_attention'] = float(top2_token_mean_attentions.mean().item())
    return record


def append_top2_sink_attention_nonpad_record(
    *,
    record: Dict[str, Any],
    save_path: str,
) -> None:
    '''Append one top-2-sink attention vs non-pad-fraction record as a JSONL line.'''
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')


def aggregate_top2_sink_attention_nonpad_from_jsonl(
    *,
    records_path: str,
) -> Dict[str, Any]:
    '''Aggregate per-sample top-2 sink attention vs non-pad percentage records.'''
    if not os.path.exists(records_path):
        raise FileNotFoundError(f'records_path does not exist: {records_path}')

    nonpad_percentage: List[float] = []
    top2_sink_avg_attention: List[float] = []
    sample_ids: List[Any] = []
    per_sample: List[Dict[str, Any]] = []
    n_samples = 0

    with open(records_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_samples += 1
            rec = json.loads(line)
            per_sample.append(rec)

            if not bool(rec.get('valid', False)):
                continue
            if rec.get('nonpad_percentage') is None or rec.get('top2_sink_avg_attention') is None:
                continue

            sample_ids.append(rec.get('sample_id'))
            nonpad_percentage.append(float(rec['nonpad_percentage']))
            top2_sink_avg_attention.append(float(rec['top2_sink_avg_attention']))

    return {
        'n_samples': n_samples,
        'n_valid_samples': len(nonpad_percentage),
        'nonpad_percentage': torch.tensor(nonpad_percentage, dtype=torch.float32),
        'top2_sink_avg_attention': torch.tensor(top2_sink_avg_attention, dtype=torch.float32),
        'sample_ids': sample_ids,
        'per_sample': per_sample,
        'records_path': records_path,
    }


def plot_top2_sink_attention_vs_nonpad_percentage(
    *,
    aggregated: Dict[str, Any],
    save_path: str,
    dpi: int = 180,
    point_alpha: float = 0.35,
    point_size: int = 24,
) -> str:
    '''
    Plot dataset-level relationship between:
      - x: percentage of non-padded graph tokens
      - y: average attention weight of the top-2 sink tokens
    '''
    x = aggregated['nonpad_percentage'].detach().cpu().to(torch.float32)
    y = aggregated['top2_sink_avg_attention'].detach().cpu().to(torch.float32)

    if x.numel() == 0 or y.numel() == 0:
        raise ValueError('No valid samples available to plot top-2 sink attention vs non-pad percentage.')
    if x.numel() != y.numel():
        raise ValueError('nonpad_percentage and top2_sink_avg_attention must have the same length.')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=dpi)
    ax.scatter(
        x.numpy(),
        y.numpy(),
        alpha=point_alpha,
        s=point_size,
        edgecolors='none',
        label='sample',
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
        color='crimson',
        linewidth=2.0,
        label='mean',
    )

    ax.set_title(
        'Average Attention: Top2 Sink Tokens on non-pad %'
        f'(valid={aggregated["n_valid_samples"]}/{aggregated["n_samples"]})'
    )
    ax.set_xlabel('Non-padded graph tokens (%)')
    ax.set_ylabel('Average attention of top-2 sink tokens')
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return save_path
