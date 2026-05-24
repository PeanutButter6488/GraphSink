import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from dataclasses import dataclass

from utils.constants import DEFAULT_GRAPH_PAD_ID, GRAPH_TOKEN_INDEX

@torch.no_grad()
def rmsnorm(
    hidden_states: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Apply RMS normalization over the last dimension.

    hidden_states:
      - [T, D] or [K, D] or [L, K, D]
    """
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps)


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


def _get_prompt_step_hidden_states(generate_outputs) -> Tuple[torch.Tensor, ...]:
    """
    Extract step-0 hidden states from HF generate outputs.

    Expected structure:
      - tuple over generation steps
        - each item is tuple over hidden-state stages
          - stage 0 is input embeddings
          - stages 1..L are transformer layer outputs
          - each tensor is [B, T, D]

    Returns only transformer layer outputs, excluding input embeddings.
    """
    hidden_states = getattr(generate_outputs, "hidden_states", None)
    if hidden_states is None:
        raise ValueError(
            "generate_outputs.hidden_states is None. Set output_hidden_states=True and return_dict_in_generate=True."
        )

    step0 = hidden_states[0]
    if not isinstance(step0, (tuple, list)) or len(step0) < 2:
        raise ValueError("Unexpected hidden_states structure from generate().")

    first = step0[0]
    if first.dim() != 3:
        raise ValueError(
            f"Prompt-step hidden states must have shape [B,T,D]. Got {tuple(first.shape)}."
        )

    return tuple(step0[1:])


@torch.no_grad()
@torch.no_grad()
def compute_layerwise_graph_token_hidden_states(
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
    Per sample, extract hidden states for graph tokens at every transformer layer.

    Returns per sample:
      - layer_graph_hidden_states: [num_layers, K, D]
      - layer_graph_activation_max_abs: [num_layers, K]
      - key_idx: [K] expanded prompt positions of graph tokens
      - key_is_pad: [K] bool mask aligned to graph tokens
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    hs0 = _get_prompt_step_hidden_states(generate_outputs)  # Acquire hidden states: shape [num_layers, batch, seq (230), 4096]
    num_layers = len(hs0)
    batch_size = hs0[0].shape[0]
    prompt_len = hs0[0].shape[1]

    if graphs is not None and graphs.dim() == 3:
        graphs_batched = graphs
    else:
        graphs_batched = graphs.unsqueeze(0).expand(batch_size, -1, -1)

    valid: List[bool] = []
    layer_graph_hidden_states_list: List[Optional[torch.Tensor]] = []
    layer_graph_activation_max_abs_list: List[Optional[torch.Tensor]] = []
    key_idx_list: List[Optional[torch.Tensor]] = []
    key_idx_all_list: List[Optional[torch.Tensor]] = []
    key_is_pad_list: List[Optional[torch.Tensor]] = []

    for b in range(batch_size):
        idxs = get_expanded_graph_key_query_indices(   # get graph key indices (absolute position)
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
            layer_graph_hidden_states_list.append(None)
            layer_graph_activation_max_abs_list.append(None)
            key_idx_list.append(None)
            key_idx_all_list.append(None)
            key_is_pad_list.append(None)
            continue

        key_idx, _, key_idx_all, key_is_pad = idxs

        layer_states: List[torch.Tensor] = []
        layer_max_abs: List[torch.Tensor] = []
        for layer_id in range(num_layers):
            layer_hidden_b = hs0[layer_id][b].to(torch.float32)  # [T, D]
            graph_hidden = layer_hidden_b.index_select(0, key_idx)  # [K, D]
            layer_states.append(graph_hidden)
            layer_max_abs.append(graph_hidden.abs().amax(dim=-1))  # [K]

        layer_graph_hidden_states = torch.stack(layer_states, dim=0)  # [L, K, D]
        layer_graph_activation_max_abs = torch.stack(layer_max_abs, dim=0)  # [L, K]

        valid.append(True)
        layer_graph_hidden_states_list.append(layer_graph_hidden_states)
        layer_graph_activation_max_abs_list.append(layer_graph_activation_max_abs)
        key_idx_list.append(key_idx)
        key_idx_all_list.append(key_idx_all)
        key_is_pad_list.append(key_is_pad)

    return {
        "valid": valid,
        "prompt_len": prompt_len,
        "layer_graph_hidden_states": layer_graph_hidden_states_list,
        "layer_graph_activation_max_abs": layer_graph_activation_max_abs_list,
        "key_idx": key_idx_list,
        "key_idx_all": key_idx_all_list,
        "key_is_pad": key_is_pad_list,
    }


def _find_first_nonpad_after_first_pad(key_is_pad: torch.Tensor) -> Optional[int]:
    """
    Find the first non-pad graph token that comes immediately after the first pad run.
    """
    key_is_pad = key_is_pad.detach().bool().reshape(-1).cpu()
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


@dataclass
class ActivationAggState:
    sum_nonpad: torch.Tensor   # [L, D]
    cnt_nonpad: torch.Tensor   # [L]
    sum_pad: torch.Tensor      # [L, D]
    cnt_pad: torch.Tensor      # [L]
    num_valid_samples: int


def init_activation_agg_state(
    *,
    num_layers: int,
    hidden_dim: int,
    device: torch.device = torch.device("cpu"),
) -> ActivationAggState:
    return ActivationAggState(
        sum_nonpad=torch.zeros((num_layers, hidden_dim), dtype=torch.float32, device=device),
        cnt_nonpad=torch.zeros((num_layers,), dtype=torch.long, device=device),
        sum_pad=torch.zeros((num_layers, hidden_dim), dtype=torch.float32, device=device),
        cnt_pad=torch.zeros((num_layers,), dtype=torch.long, device=device),
        num_valid_samples=0,
    )


def update_activation_agg_state(
    *,
    state: ActivationAggState,
    layer_graph_hidden_states: torch.Tensor,   # [L, K, D]
    key_is_pad: torch.Tensor,                  # [K]
) -> ActivationAggState:
    """
    Aggregate hidden activations across samples, split into non-pad vs pad graph tokens.
    """
    if layer_graph_hidden_states.dim() != 3:
        raise ValueError("layer_graph_hidden_states must have shape [L, K, D].")

    layer_graph_hidden_states = layer_graph_hidden_states.detach().to(torch.float32).cpu()
    key_is_pad = key_is_pad.detach().bool().reshape(-1).cpu()

    num_layers, num_tokens, hidden_dim = layer_graph_hidden_states.shape
    if key_is_pad.numel() != num_tokens:
        raise ValueError("key_is_pad length must match number of graph tokens.")

    if state.sum_nonpad.shape != (num_layers, hidden_dim):
        raise ValueError(
            "ActivationAggState shape mismatch. Initialize it with the model's num_layers and hidden_dim."
        )

    nonpad_mask = ~key_is_pad
    if bool(nonpad_mask.any()):
        nonpad_hidden = layer_graph_hidden_states[:, nonpad_mask, :]   # [L, K_nonpad, D]
        state.sum_nonpad += nonpad_hidden.sum(dim=1)
        state.cnt_nonpad += nonpad_mask.sum().to(torch.long)

    if bool(key_is_pad.any()):
        pad_hidden = layer_graph_hidden_states[:, key_is_pad, :]       # [L, K_pad, D]
        state.sum_pad += pad_hidden.sum(dim=1)
        state.cnt_pad += key_is_pad.sum().to(torch.long)

    state.num_valid_samples += 1
    return state


def finalize_activation_agg_state(
    state: ActivationAggState,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    cnt_nonpad_f = state.cnt_nonpad.to(torch.float32).unsqueeze(-1)
    cnt_pad_f = state.cnt_pad.to(torch.float32).unsqueeze(-1)

    mean_nonpad = state.sum_nonpad / (cnt_nonpad_f + eps)
    mean_pad = state.sum_pad / (cnt_pad_f + eps)

    return {
        "num_valid_samples": state.num_valid_samples,
        "mean_nonpad": mean_nonpad,
        "count_nonpad": state.cnt_nonpad.clone(),
        "mean_pad": mean_pad,
        "count_pad": state.cnt_pad.clone(),
    }


@dataclass
class SinkActivationAggState:
    sum_sink: torch.Tensor        # [L, D]
    cnt_sink: torch.Tensor        # [L]
    sum_rest_graph: torch.Tensor  # [L, D]
    cnt_rest_graph: torch.Tensor  # [L]
    sum_pad: torch.Tensor         # [L, D]
    cnt_pad: torch.Tensor         # [L]
    num_valid_samples: int


def init_sink_activation_agg_state(
    *,
    num_layers: int,
    hidden_dim: int,
    device: torch.device = torch.device("cpu"),
) -> SinkActivationAggState:
    return SinkActivationAggState(
        sum_sink=torch.zeros((num_layers, hidden_dim), dtype=torch.float32, device=device),
        cnt_sink=torch.zeros((num_layers,), dtype=torch.long, device=device),
        sum_rest_graph=torch.zeros((num_layers, hidden_dim), dtype=torch.float32, device=device),
        cnt_rest_graph=torch.zeros((num_layers,), dtype=torch.long, device=device),
        sum_pad=torch.zeros((num_layers, hidden_dim), dtype=torch.float32, device=device),
        cnt_pad=torch.zeros((num_layers,), dtype=torch.long, device=device),
        num_valid_samples=0,
    )


def update_sink_activation_agg_state(
    *,
    state: SinkActivationAggState,
    layer_graph_hidden_states: torch.Tensor,   # [L, K, D]
    key_is_pad: torch.Tensor,                  # [K]
) -> SinkActivationAggState:
    """
    Aggregate hidden activations for:
      - the sink token itself
      - the remaining non-pad graph tokens
      - the pad tokens

    The sink token is defined as the first non-pad token after the first pad run.
    """
    if layer_graph_hidden_states.dim() != 3:
        raise ValueError("layer_graph_hidden_states must have shape [L, K, D].")

    layer_graph_hidden_states = layer_graph_hidden_states.detach().to(torch.float32).cpu()
    key_is_pad = key_is_pad.detach().bool().reshape(-1).cpu()

    num_layers, num_tokens, hidden_dim = layer_graph_hidden_states.shape
    if key_is_pad.numel() != num_tokens:
        raise ValueError("key_is_pad length must match number of graph tokens.")

    if state.sum_sink.shape != (num_layers, hidden_dim):
        raise ValueError(
            "SinkActivationAggState shape mismatch. Initialize it with the model's num_layers and hidden_dim."
        )

    sink_idx = _find_first_nonpad_after_first_pad(key_is_pad)
    if sink_idx is None:
        return state

    sink_hidden = layer_graph_hidden_states[:, sink_idx, :]  # [L, D]
    state.sum_sink += sink_hidden
    state.cnt_sink += 1

    rest_graph_mask = ~key_is_pad
    rest_graph_mask[sink_idx] = False
    if bool(rest_graph_mask.any()):
        rest_graph_hidden = layer_graph_hidden_states[:, rest_graph_mask, :]  # [L, K_rest_graph, D]
        state.sum_rest_graph += rest_graph_hidden.sum(dim=1)
        state.cnt_rest_graph += int(rest_graph_mask.sum().item())

    pad_mask = key_is_pad
    if bool(pad_mask.any()):
        pad_hidden = layer_graph_hidden_states[:, pad_mask, :]  # [L, K_pad, D]
        state.sum_pad += pad_hidden.sum(dim=1)
        state.cnt_pad += int(pad_mask.sum().item())

    state.num_valid_samples += 1
    return state


def finalize_sink_activation_agg_state(
    state: SinkActivationAggState,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    cnt_sink_f = state.cnt_sink.to(torch.float32).unsqueeze(-1)
    cnt_rest_graph_f = state.cnt_rest_graph.to(torch.float32).unsqueeze(-1)
    cnt_pad_f = state.cnt_pad.to(torch.float32).unsqueeze(-1)

    mean_sink = state.sum_sink / (cnt_sink_f + eps)
    mean_rest_graph = state.sum_rest_graph / (cnt_rest_graph_f + eps)
    mean_pad = state.sum_pad / (cnt_pad_f + eps)

    return {
        "num_valid_samples": state.num_valid_samples,
        "mean_sink": mean_sink,
        "count_sink": state.cnt_sink.clone(),
        "mean_rest_graph": mean_rest_graph,
        "count_rest_graph": state.cnt_rest_graph.clone(),
        "mean_pad": mean_pad,
        "count_pad": state.cnt_pad.clone(),
    }


@dataclass
class ActivationTopDimsAggState:
    sum_all: torch.Tensor          # [L, D] — sum of |RMSNorm(x)|
    sum_all_signed: torch.Tensor   # [L, D] — sum of RMSNorm(x) (signed)
    sum_all_raw: torch.Tensor      # [L, D] — sum of x (raw, no RMSNorm, signed)
    cnt_all: torch.Tensor          # [L]
    num_valid_samples: int


def init_activation_topdims_agg_state(
    *,
    num_layers: int,
    hidden_dim: int,
    device: torch.device = torch.device("cpu"),
) -> ActivationTopDimsAggState:
    return ActivationTopDimsAggState(
        sum_all=torch.zeros((num_layers, hidden_dim), dtype=torch.float32, device=device),
        sum_all_signed=torch.zeros((num_layers, hidden_dim), dtype=torch.float32, device=device),
        sum_all_raw=torch.zeros((num_layers, hidden_dim), dtype=torch.float32, device=device),
        cnt_all=torch.zeros((num_layers,), dtype=torch.long, device=device),
        num_valid_samples=0,
    )


def update_activation_topdims_agg_state(
    *,
    state: ActivationTopDimsAggState,
    layer_graph_hidden_states: torch.Tensor,   # [L, K, D]
    token_indices: Optional[torch.Tensor] = None,  # subset of tokens (indices into K)
    rmsnorm_eps: float = 1e-6,
) -> ActivationTopDimsAggState:
    """
    Aggregate hidden activations across graph tokens, in three views:
      - sum_all:        sum of ``|RMSNorm(x)|``
      - sum_all_signed: sum of ``RMSNorm(x)`` (signed)
      - sum_all_raw:    sum of ``x`` (raw, no RMSNorm, signed)

    If ``token_indices`` is provided, only those graph-token positions are
    aggregated (e.g., previously identified sink tokens). RMSNorm is still
    computed per-selected-token over the hidden dim, matching the all-token
    behaviour.

    Downstream plots pick whichever view they want without re-running
    aggregation.

    With ``token_indices=None`` this treats all graph-token slots together:
      - neighborhood view: pad + non-pad tokens are combined
      - hop view: the three aggregated graph tokens are combined as-is
    """
    if layer_graph_hidden_states.dim() != 3:
        raise ValueError("layer_graph_hidden_states must have shape [L, K, D].")

    layer_graph_hidden_states = layer_graph_hidden_states.detach().to(torch.float32).cpu()
    num_layers, num_tokens, hidden_dim = layer_graph_hidden_states.shape

    if state.sum_all.shape != (num_layers, hidden_dim):
        raise ValueError(
            "ActivationTopDimsAggState shape mismatch. Initialize it with the model's num_layers and hidden_dim."
        )

    if token_indices is not None:
        token_indices = token_indices.detach().to(torch.long).cpu().reshape(-1)
        if token_indices.numel() == 0:
            return state
        if int(token_indices.min().item()) < 0 or int(token_indices.max().item()) >= num_tokens:
            raise ValueError(
                f"token_indices out of range for K={num_tokens}: "
                f"min={int(token_indices.min().item())}, max={int(token_indices.max().item())}."
            )
        selected = layer_graph_hidden_states.index_select(1, token_indices)
    else:
        selected = layer_graph_hidden_states

    n_selected = selected.shape[1]
    normed = rmsnorm(selected, eps=rmsnorm_eps)
    state.sum_all += normed.abs().sum(dim=1)
    state.sum_all_signed += normed.sum(dim=1)
    state.sum_all_raw += selected.sum(dim=1)
    state.cnt_all += n_selected
    state.num_valid_samples += 1
    return state


def finalize_activation_topdims_agg_state(
    state: ActivationTopDimsAggState,
    *,
    topk: int = 5,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Compute per-layer mean activation values and keep only the top-k dimension IDs.
    """
    cnt_all_f = state.cnt_all.to(torch.float32).unsqueeze(-1)
    mean_all = state.sum_all / (cnt_all_f + eps)
    mean_all_signed = state.sum_all_signed / (cnt_all_f + eps)
    mean_all_raw = state.sum_all_raw / (cnt_all_f + eps)
    num_layers, hidden_dim = mean_all.shape

    top_dims_by_layer: List[List[int]] = []
    top_values_by_layer: List[List[float]] = []
    top_dim_counts = torch.zeros((hidden_dim,), dtype=torch.long)

    k = min(int(topk), hidden_dim)
    for layer_idx in range(num_layers):
        values, indices = torch.topk(mean_all[layer_idx], k=k)
        dims = [int(v.item()) for v in indices]
        vals = [float(v.item()) for v in values]
        top_dims_by_layer.append(dims)
        top_values_by_layer.append(vals)
        top_dim_counts[indices] += 1

    return {
        "num_valid_samples": state.num_valid_samples,
        "topk": int(k),
        "mean_all": mean_all,
        "mean_all_signed": mean_all_signed,
        "mean_all_raw": mean_all_raw,
        "count_all": state.cnt_all.clone(),
        "top_dims_by_layer": top_dims_by_layer,
        "top_values_by_layer": top_values_by_layer,
        "top_dim_counts": top_dim_counts,
    }


def save_activation_topdims_summary(
    *,
    aggregated: Dict[str, Any],
    save_path: str,
    dataset_name: Optional[str] = None,
    view_name: Optional[str] = None,
) -> str:
    """
    Save only the per-layer top-k activation dimension IDs and lightweight metadata.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    mean_all = aggregated["mean_all"].detach().cpu()
    payload: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "view_name": view_name,
        "num_valid_samples": int(aggregated["num_valid_samples"]),
        "topk": int(aggregated["topk"]),
        "num_layers": int(mean_all.shape[0]),
        "hidden_dim": int(mean_all.shape[1]),
        "count_all": aggregated["count_all"].detach().cpu().to(torch.long).tolist(),
        "top_dims_by_layer": aggregated["top_dims_by_layer"],
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return save_path


def load_activation_topdims_summary(
    *,
    summary_path: str,
) -> Dict[str, Any]:
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary_path does not exist: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_activation_topdims_count_aggregate(
    *,
    summary_paths: List[str],
    save_path: str,
    dpi: int = 180,
    annotate_top_n: int = 15,
    annotate_offset: int = 2,
) -> str:
    """
    Aggregate across ALL provided summary files and plot how often each hidden
    dimension appears among the per-layer top-k activation dimensions.

    Changes vs your original:
      1) No per-dataset lines: counts are summed across all summary_paths.
      2) Annotate dimension numbers for the top-N most frequent dimensions.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

    if len(summary_paths) == 0:
        raise ValueError("summary_paths must be non-empty.")

    # Load all summaries
    summaries = [load_activation_topdims_summary(summary_path=p) for p in summary_paths]
    hidden_dim = int(max(s["hidden_dim"] for s in summaries))

    # Aggregate counts across all summaries
    counts = torch.zeros((hidden_dim,), dtype=torch.long)
    for summary in summaries:
        for dims in summary["top_dims_by_layer"]:
            for dim_idx in dims:
                d = int(dim_idx)
                if 0 <= d < hidden_dim:
                    counts[d] += 1

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)
    x = torch.arange(hidden_dim, dtype=torch.long)

    ax.plot(x.tolist(), counts.tolist(), linewidth=1.2)

    ax.set_title("Top activation-dimension counts (aggregated)")
    ax.set_xlabel("Dimension number")
    ax.set_ylabel("Count among per-layer top-k dimensions")
    ax.grid(True, axis="y", alpha=0.25)

    # Annotate the top-N dimensions with their indices
    if annotate_top_n and annotate_top_n > 0:
        k = min(int(annotate_top_n), hidden_dim)
        topk = torch.topk(counts, k=k, largest=True)

        for d, c in zip(topk.indices.tolist(), topk.values.tolist()):
            # Place label slightly above the point
            ax.annotate(
                str(d),
                xy=(d, c),
                xytext=(0, annotate_offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_topdims_mean_activation_curve(
    *,
    aggregated: Dict[str, Any],
    save_path: str,
    dpi: int = 180,
    sink_threshold: float = 5.0,
    layer_index: int = -2,
    use_abs: bool = True,
) -> Tuple[str, str, List[int]]:
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
            f"aggregated is missing '{key}'. Re-run aggregation with the "
            f"updated update_activation_topdims_agg_state."
        )
    mean_all = aggregated[key].detach().cpu().to(torch.float32)  # [L, D]
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


def plot_topdims_mean_activation_curve_original(
    *,
    aggregated: Dict[str, Any],
    save_path: str,
    dpi: int = 180,
    topk_dims: int = 3,
) -> Tuple[str, List[int]]:
    """
    Replicate the pre-modification plot exactly: layer- and token-averaged
    *raw signed* activations (no RMSNorm, no absolute value), per feature
    dimension. Top-k dims are picked by largest signed value, matching the
    original behaviour.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

    if "mean_all_raw" not in aggregated:
        raise KeyError(
            "aggregated is missing 'mean_all_raw'. Re-run aggregation with the "
            "updated update_activation_topdims_agg_state."
        )

    mean_all = aggregated["mean_all_raw"].detach().cpu().to(torch.float32)  # [L, D]
    mean_per_dim = mean_all.mean(dim=0)                                      # [D]
    hidden_dim = mean_per_dim.numel()
    k = min(int(topk_dims), hidden_dim)
    _, top_idx = torch.topk(mean_per_dim, k=k)
    top_dims = sorted(int(v.item()) for v in top_idx)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=dpi)
    ax.plot(list(range(hidden_dim)), mean_per_dim.tolist(), linewidth=1.0, color="#1f77b4")
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
    for d in top_dims:
        v = float(mean_per_dim[d].item())
        ax.axvline(d, color="#d62728", linewidth=0.9, alpha=0.75, linestyle="--")
        ax.scatter([d], [v], color="#d62728", s=18, zorder=3)
        ax.text(d, v, f"{d}", color="#d62728", fontsize=8, ha="left", va="bottom")
    ax.set_title(
        f"Layer- and token-averaged graph-token activation | top-{k} dims: {top_dims} | "
        f"samples={int(aggregated['num_valid_samples'])}"
    )
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Sink Magnitude")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path, top_dims


def plot_aggregated_activation_curves(
    *,
    aggregated: Dict[str, Any],
    save_dir: str,
    dpi: int = 180,
) -> List[str]:
    """
    For every layer, save two plots:
      - average activation over all non-pad graph tokens
      - average activation over all pad graph tokens
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it to save aggregated activation plots."
        ) from e

    mean_nonpad = aggregated["mean_nonpad"].detach().cpu().to(torch.float32)   # [L, D]
    mean_pad = aggregated["mean_pad"].detach().cpu().to(torch.float32)         # [L, D]
    count_nonpad = aggregated["count_nonpad"].detach().cpu().to(torch.long)
    count_pad = aggregated["count_pad"].detach().cpu().to(torch.long)

    num_layers, hidden_dim = mean_nonpad.shape
    x = list(range(hidden_dim))
    os.makedirs(save_dir, exist_ok=True)

    saved_paths: List[str] = []
    for layer_idx in range(num_layers):
        plots = [
            ("graph", mean_nonpad[layer_idx], int(count_nonpad[layer_idx].item()), "#1f77b4"),
            ("pad", mean_pad[layer_idx], int(count_pad[layer_idx].item()), "#666666"),
        ]

        for name, y, count, color in plots:
            topk = min(2, hidden_dim)
            _, top_idx = torch.topk(y, k=topk)
            top_dims = sorted(int(v.item()) for v in top_idx)
            fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
            ax.plot(x, y.tolist(), linewidth=1.0, color=color)
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
            for rank, dim_idx in enumerate(top_dims, start=1):
                dim_val = float(y[dim_idx].item())
                ax.axvline(dim_idx, color="#d62728", linewidth=0.9, alpha=0.75, linestyle="--")
                ax.scatter([dim_idx], [dim_val], color="#d62728", s=18, zorder=3)
                ax.text(
                    dim_idx,
                    dim_val,
                    f"{dim_idx}",
                    color="#d62728",
                    fontsize=7,
                    ha="left",
                    va="bottom",
                )
            ax.set_title(
                f"Layer {layer_idx} | Avg activation of {name} tokens | count={count}"
            )
            ax.set_xlabel("LLM embedding dimension")
            ax.set_ylabel("Average activation value")
            ax.set_xticks(top_dims)
            ax.set_xticklabels([str(dim_idx) for dim_idx in top_dims])
            ax.grid(True, axis="y", alpha=0.25)

            out_path = os.path.join(save_dir, f"layer_{layer_idx:02d}_{name}_avg_activation.png")
            fig.tight_layout()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(out_path)

    return saved_paths


def plot_sink_vs_rest_activation_curves(
    *,
    aggregated: Dict[str, Any],
    save_dir: str,
    dpi: int = 180,
    topk_dims: int = 5,
) -> List[str]:
    """
    For every layer, save three plots:
      - average activation over the sink token
      - average activation over the remaining non-pad graph tokens
      - average activation over the pad tokens

    Each plot annotates the top-k embedding dimensions with the highest average activation.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it to save aggregated activation plots."
        ) from e

    mean_sink = aggregated["mean_sink"].detach().cpu().to(torch.float32)
    mean_rest_graph = aggregated["mean_rest_graph"].detach().cpu().to(torch.float32)
    mean_pad = aggregated["mean_pad"].detach().cpu().to(torch.float32)
    count_sink = aggregated["count_sink"].detach().cpu().to(torch.long)
    count_rest_graph = aggregated["count_rest_graph"].detach().cpu().to(torch.long)
    count_pad = aggregated["count_pad"].detach().cpu().to(torch.long)

    num_layers, hidden_dim = mean_sink.shape
    x = list(range(hidden_dim))
    os.makedirs(save_dir, exist_ok=True)

    saved_paths: List[str] = []
    for layer_idx in range(num_layers):
        plots = [
            ("sink", mean_sink[layer_idx], int(count_sink[layer_idx].item()), "#d62728"),
            ("rest_graph", mean_rest_graph[layer_idx], int(count_rest_graph[layer_idx].item()), "#1f77b4"),
            ("pad", mean_pad[layer_idx], int(count_pad[layer_idx].item()), "#666666"),
        ]

        for name, y, count, color in plots:
            fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
            ax.plot(x, y.tolist(), linewidth=1.0, color=color)
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)

            top_dims = []
            if count > 0:
                k = min(int(topk_dims), hidden_dim)
                _, top_idx = torch.topk(y, k=k)
                top_dims = sorted(int(v.item()) for v in top_idx)
                for dim_idx in top_dims:
                    dim_val = float(y[dim_idx].item())
                    ax.axvline(dim_idx, color="#d62728", linewidth=0.9, alpha=0.75, linestyle="--")
                    ax.scatter([dim_idx], [dim_val], color="#d62728", s=18, zorder=3)
                    ax.text(
                        dim_idx,
                        dim_val,
                        f"{dim_idx}",
                        color="#d62728",
                        fontsize=7,
                        ha="left",
                        va="bottom",
                    )
                ax.set_xticks(top_dims)
                ax.set_xticklabels([str(dim_idx) for dim_idx in top_dims])

            ax.set_title(
                f"Layer {layer_idx} | Avg activation of {name} tokens | count={count}"
            )
            ax.set_xlabel("LLM embedding dimension")
            ax.set_ylabel("Average activation value")
            ax.grid(True, axis="y", alpha=0.25)

            out_path = os.path.join(save_dir, f"layer_{layer_idx:02d}_{name}_avg_activation.png")
            fig.tight_layout()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(out_path)

    return saved_paths


def find_high_activation_graph_tokens(
    *,
    layer_graph_hidden_states: Optional[torch.Tensor] = None,
    layer_graph_activation_max_abs: Optional[torch.Tensor] = None,
    key_idx: Optional[torch.Tensor] = None,
    key_is_pad: Optional[torch.Tensor] = None,
    threshold: float = 20.0,
) -> Dict[str, Any]:
    """
    For one sample, find graph tokens whose activation is unusually high.

    A token is flagged at a layer when max(abs(hidden_state)) across hidden dimension
    exceeds `threshold`.
    """
    if layer_graph_activation_max_abs is None:
        if layer_graph_hidden_states is None:
            raise ValueError(
                "Provide either layer_graph_hidden_states or layer_graph_activation_max_abs."
            )
        if layer_graph_hidden_states.dim() != 3:
            raise ValueError("layer_graph_hidden_states must have shape [num_layers, K, D].")
        layer_graph_hidden_states = layer_graph_hidden_states.detach().cpu().to(torch.float32)
        max_abs = layer_graph_hidden_states.abs().amax(dim=-1)  # [L, K]
    else:
        if layer_graph_activation_max_abs.dim() != 2:
            raise ValueError(
                "layer_graph_activation_max_abs must have shape [num_layers, K]."
            )
        max_abs = layer_graph_activation_max_abs.detach().cpu().to(torch.float32)

    num_layers, num_tokens = max_abs.shape

    if key_idx is not None:
        key_idx = key_idx.detach().cpu().to(torch.long).reshape(-1)
        if key_idx.numel() != num_tokens:
            raise ValueError("key_idx length must match number of graph tokens.")

    if key_is_pad is not None:
        key_is_pad = key_is_pad.detach().cpu().bool().reshape(-1)
        if key_is_pad.numel() != num_tokens:
            raise ValueError("key_is_pad length must match number of graph tokens.")

    findings: List[Dict[str, Any]] = []
    for layer_idx in range(num_layers):
        token_mask = max_abs[layer_idx] > threshold
        token_indices = torch.nonzero(token_mask, as_tuple=False).reshape(-1)
        if token_indices.numel() == 0:
            continue

        entry: Dict[str, Any] = {
            "layer_index": int(layer_idx),
            "graph_token_indices": token_indices.tolist(),
            "max_abs_activations": max_abs[layer_idx, token_indices].tolist(),
        }
        if key_idx is not None:
            entry["prompt_token_indices"] = key_idx[token_indices].tolist()
        if key_is_pad is not None:
            entry["is_pad"] = key_is_pad[token_indices].tolist()
        findings.append(entry)

    return {
        "threshold": float(threshold),
        "num_layers": int(num_layers),
        "num_graph_tokens": int(num_tokens),
        "layer_graph_activation_max_abs": max_abs,
        "findings": findings,
    }


def plot_sample_graph_token_activations(
    *,
    layer_graph_hidden_states: torch.Tensor,
    sample_id: Any,
    save_dir: str,
    key_idx: Optional[torch.Tensor] = None,
    key_is_pad: Optional[torch.Tensor] = None,
    dpi: int = 180,
) -> List[str]:
    """
    Plot one figure for every (graph token, layer) pair for a sample.

    x-axis: LLM embedding dimension
    y-axis: activation value
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it to save activation plots."
        ) from e

    if layer_graph_hidden_states.dim() != 3:
        raise ValueError("layer_graph_hidden_states must have shape [num_layers, K, D].")

    layer_graph_hidden_states = layer_graph_hidden_states.detach().cpu().to(torch.float32)
    num_layers, num_tokens, hidden_dim = layer_graph_hidden_states.shape
    if num_layers == 0 or num_tokens == 0 or hidden_dim == 0:
        raise ValueError("No graph-token activations to plot.")

    if key_idx is not None:
        key_idx = key_idx.detach().cpu().to(torch.long).reshape(-1)
        if key_idx.numel() != num_tokens:
            raise ValueError("key_idx length must match number of graph tokens.")
    if key_is_pad is not None:
        key_is_pad = key_is_pad.detach().cpu().bool().reshape(-1)
        if key_is_pad.numel() != num_tokens:
            raise ValueError("key_is_pad length must match number of graph tokens.")

    sample_dir = os.path.join(save_dir, f"sample_{sample_id}")
    os.makedirs(sample_dir, exist_ok=True)

    x = list(range(hidden_dim))
    saved_paths: List[str] = []

    for token_idx in range(num_tokens):
        prompt_idx_str = str(int(key_idx[token_idx])) if key_idx is not None else str(token_idx)
        is_pad = bool(key_is_pad[token_idx]) if key_is_pad is not None else False
        pad_suffix = "_pad" if is_pad else "_graph"

        for layer_idx in range(num_layers):
            y = layer_graph_hidden_states[layer_idx, token_idx].tolist()
            fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
            ax.plot(
                x,
                y,
                linewidth=1.0,
                color="#666666" if is_pad else "#1f77b4",
                linestyle="--" if is_pad else "-",
            )
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
            ax.set_title(
                f"Sample {sample_id} | Token {token_idx} | Prompt {prompt_idx_str}"
                f"{' (P)' if is_pad else ''} | Layer {layer_idx}"
            )
            ax.set_xlabel("LLM embedding dimension")
            ax.set_ylabel("Activation value")
            ax.grid(True, axis="y", alpha=0.25)

            out_path = os.path.join(
                sample_dir,
                f"token_{token_idx:03d}_prompt_{prompt_idx_str}{pad_suffix}_layer_{layer_idx:02d}.png",
            )
            fig.tight_layout()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(out_path)

    return saved_paths


def analyze_and_plot_sample_graph_token_activations(
    *,
    generate_outputs,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    graphs: torch.Tensor,
    sample_id: Any,
    save_dir: str,
    keep_pad_tokens: bool = True,
    mm_use_graph_special_token: bool = False,
    use_hop: Optional[int] = None,
    sample_neighbor_size: Optional[int] = None,
    threshold: float = 20.0,
) -> Dict[str, Any]:
    """
    Convenience wrapper for per-sample activation analysis plus per-token/per-layer plotting.
    """
    analysis = compute_layerwise_graph_token_hidden_states(
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
    threshold_findings: List[Optional[Dict[str, Any]]] = []

    for b, is_valid in enumerate(analysis["valid"]):
        if not is_valid:
            threshold_findings.append(None)
            continue

        layer_hidden = analysis["layer_graph_hidden_states"][b]
        layer_max_abs = analysis["layer_graph_activation_max_abs"][b]
        key_idx = analysis["key_idx"][b]
        key_is_pad = analysis["key_is_pad"][b]

        suffix = sample_id if len(analysis["valid"]) == 1 else f"{sample_id}_b{b}"
        paths = plot_sample_graph_token_activations(
            layer_graph_hidden_states=layer_hidden,
            sample_id=suffix,
            save_dir=save_dir,
            key_idx=key_idx,
            key_is_pad=key_is_pad,
        )
        saved_paths.extend(paths)

        threshold_findings.append(
            find_high_activation_graph_tokens(
                layer_graph_activation_max_abs=layer_max_abs,
                key_idx=key_idx,
                key_is_pad=key_is_pad,
                threshold=threshold,
            )
        )

    analysis["saved_paths"] = saved_paths
    analysis["threshold_findings"] = threshold_findings
    return analysis


###### Activation Probe Pre-graph token
@dataclass
class PreGraphActivationAggState:
    sum_first: torch.Tensor      # [L, D]
    cnt_first: torch.Tensor      # [L]
    sum_pregraph: torch.Tensor   # [L, D]
    cnt_pregraph: torch.Tensor   # [L]
    num_valid_samples: int


def init_pregraph_activation_agg_state(
    *,
    num_layers: int,
    hidden_dim: int,
    device: torch.device = torch.device("cpu"),
) -> PreGraphActivationAggState:
    return PreGraphActivationAggState(
        sum_first=torch.zeros((num_layers, hidden_dim), dtype=torch.float32, device=device),
        cnt_first=torch.zeros((num_layers,), dtype=torch.long, device=device),
        sum_pregraph=torch.zeros((num_layers, hidden_dim), dtype=torch.float32, device=device),
        cnt_pregraph=torch.zeros((num_layers,), dtype=torch.long, device=device),
        num_valid_samples=0,
    )


@torch.no_grad()
@torch.no_grad()
def compute_layerwise_pregraph_token_hidden_states(
    *,
    generate_outputs,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    graphs: torch.Tensor,
    keep_pad_tokens: bool = True,
    mm_use_graph_special_token: bool = False,
    use_hop: Optional[int] = None,
    sample_neighbor_size: Optional[int] = None,
    apply_rmsnorm: bool = True,
    rmsnorm_eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    For each valid sample, extract:
      - the first token hidden state at every layer
      - all token hidden states before the first graph token at every layer

    Returns per sample:
      - layer_first_token_hidden_states: [L, D]
      - layer_pregraph_hidden_states: [L, K_pre, D]
      - pregraph_idx: [K_pre]
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    hs0 = _get_prompt_step_hidden_states(generate_outputs)
    num_layers = len(hs0)
    batch_size = hs0[0].shape[0]
    prompt_len = hs0[0].shape[1]

    if graphs is not None and graphs.dim() == 3:
        graphs_batched = graphs
    else:
        graphs_batched = graphs.unsqueeze(0).expand(batch_size, -1, -1)

    valid: List[bool] = []
    layer_first_token_hidden_states_list: List[Optional[torch.Tensor]] = []
    layer_pregraph_hidden_states_list: List[Optional[torch.Tensor]] = []
    pregraph_idx_list: List[Optional[torch.Tensor]] = []

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
            layer_first_token_hidden_states_list.append(None)
            layer_pregraph_hidden_states_list.append(None)
            pregraph_idx_list.append(None)
            continue

        key_idx, _, _, _ = idxs
        first_graph_idx = int(key_idx.min().item())
        pregraph_idx = torch.arange(
            0, first_graph_idx, device=input_ids.device, dtype=torch.long
        )

        if pregraph_idx.numel() == 0:
            valid.append(False)
            layer_first_token_hidden_states_list.append(None)
            layer_pregraph_hidden_states_list.append(None)
            pregraph_idx_list.append(None)
            continue

        layer_first_states: List[torch.Tensor] = []
        layer_pregraph_states: List[torch.Tensor] = []

        for layer_id in range(num_layers):
            layer_hidden_b = hs0[layer_id][b].to(torch.float32)   # [T, D]
            if apply_rmsnorm:
                layer_hidden_b = rmsnorm(layer_hidden_b, eps=rmsnorm_eps)

            first_hidden = layer_hidden_b[0]                      # [D]
            pregraph_hidden = layer_hidden_b.index_select(0, pregraph_idx)  # [K_pre, D]

            layer_first_states.append(first_hidden)
            layer_pregraph_states.append(pregraph_hidden)

        layer_first_token_hidden_states = torch.stack(layer_first_states, dim=0)   # [L, D]
        layer_pregraph_hidden_states = torch.stack(layer_pregraph_states, dim=0)   # [L, K_pre, D]

        valid.append(True)
        layer_first_token_hidden_states_list.append(layer_first_token_hidden_states)
        layer_pregraph_hidden_states_list.append(layer_pregraph_hidden_states)
        pregraph_idx_list.append(pregraph_idx)

    return {
        "valid": valid,
        "prompt_len": prompt_len,
        "layer_first_token_hidden_states": layer_first_token_hidden_states_list,
        "layer_pregraph_hidden_states": layer_pregraph_hidden_states_list,
        "pregraph_idx": pregraph_idx_list,
    }


def update_pregraph_activation_agg_state(
    *,
    state: PreGraphActivationAggState,
    layer_first_token_hidden_states: torch.Tensor,   # [L, D]
    layer_pregraph_hidden_states: torch.Tensor,      # [L, K_pre, D]
) -> PreGraphActivationAggState:
    layer_first_token_hidden_states = layer_first_token_hidden_states.detach().to(torch.float32).cpu()
    layer_pregraph_hidden_states = layer_pregraph_hidden_states.detach().to(torch.float32).cpu()

    state.sum_first += layer_first_token_hidden_states
    state.cnt_first += 1

    state.sum_pregraph += layer_pregraph_hidden_states.sum(dim=1)
    state.cnt_pregraph += layer_pregraph_hidden_states.shape[1]

    state.num_valid_samples += 1
    return state


def finalize_pregraph_activation_agg_state(
    state: PreGraphActivationAggState,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    cnt_first_f = state.cnt_first.to(torch.float32).unsqueeze(-1)
    cnt_pregraph_f = state.cnt_pregraph.to(torch.float32).unsqueeze(-1)

    mean_first = state.sum_first / (cnt_first_f + eps)
    mean_pregraph = state.sum_pregraph / (cnt_pregraph_f + eps)

    return {
        "num_valid_samples": state.num_valid_samples,
        "mean_first": mean_first,
        "count_first": state.cnt_first.clone(),
        "mean_pregraph": mean_pregraph,
        "count_pregraph": state.cnt_pregraph.clone(),
    }


def plot_pregraph_activation_curves(
    *,
    aggregated: Dict[str, Any],
    save_dir: str,
    dpi: int = 180,
    topk_dims: int = 5,
) -> List[str]:
    """
    For every layer, save two plots:
      - average activation of the very first token
      - average activation of all tokens before the graph block
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

    mean_first = aggregated["mean_first"].detach().cpu().to(torch.float32)
    mean_pregraph = aggregated["mean_pregraph"].detach().cpu().to(torch.float32)
    count_first = aggregated["count_first"].detach().cpu().to(torch.long)
    count_pregraph = aggregated["count_pregraph"].detach().cpu().to(torch.long)

    num_layers, hidden_dim = mean_first.shape
    x = list(range(hidden_dim))
    os.makedirs(save_dir, exist_ok=True)

    saved_paths: List[str] = []
    for layer_idx in range(num_layers):
        plots = [
            ("first_token", mean_first[layer_idx], int(count_first[layer_idx].item()), "#d62728"),
            ("all_pregraph_tokens", mean_pregraph[layer_idx], int(count_pregraph[layer_idx].item()), "#1f77b4"),
        ]

        for name, y, count, color in plots:
            fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
            ax.plot(x, y.tolist(), linewidth=1.0, color=color)
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)

            if count > 0:
                k = min(int(topk_dims), hidden_dim)
                _, top_idx = torch.topk(y.abs(), k=k)
                top_dims = sorted(int(v.item()) for v in top_idx)

                for dim_idx in top_dims:
                    dim_val = float(y[dim_idx].item())
                    ax.axvline(dim_idx, color="#d62728", linewidth=0.9, alpha=0.75, linestyle="--")
                    ax.scatter([dim_idx], [dim_val], color="#d62728", s=18, zorder=3)
                    ax.text(
                        dim_idx,
                        dim_val,
                        f"{dim_idx}",
                        color="#d62728",
                        fontsize=7,
                        ha="left",
                        va="bottom" if dim_val >= 0 else "top",
                    )

                ax.set_xticks(top_dims)
                ax.set_xticklabels([str(dim_idx) for dim_idx in top_dims])

            ax.set_title(f"Layer {layer_idx} | Avg activation of {name} | count={count}")
            ax.set_xlabel("LLM embedding dimension")
            ax.set_ylabel("Average activation value")
            ax.grid(True, axis="y", alpha=0.25)

            out_path = os.path.join(save_dir, f"layer_{layer_idx:02d}_{name}_avg_activation.png")
            fig.tight_layout()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(out_path)

    return saved_paths



######### Identify Sink tokens based on Sink Dimensions 

def detect_sink_graph_tokens(
    *,
    layer_graph_hidden_states:torch.Tensor,
    key_idx: Optional[torch.Tensor] = None,
    key_is_pad: Optional[torch.Tensor] = None,
    sink_dims: List[int] = [1512, 2298, 2533],
    threshold: float = 20.0,
    ignore_pad_tokens: bool = False,
    rmsnorm_eps: float = 1e-6,
    layer_index: int = -2,
) -> Dict[str, Any]:
    x = layer_graph_hidden_states.to(torch.float32).cpu()
    x = rmsnorm(x, eps=rmsnorm_eps)

    num_layers, num_tokens, hidden_dim = x.shape
    sink_dims = [int(d) for d in sink_dims if 0 <= int(d) < hidden_dim]

    if len(sink_dims) == 0:
        return {
            "sink_dims": [],
            "threshold": float(threshold),
            "num_layers": int(num_layers),
            "num_graph_tokens": int(num_tokens),
            "sink_scores": torch.zeros(num_tokens, dtype=torch.float32),
            "sink_token_indices": [],
            "prompt_token_indices": [],
            "is_pad": [],
        }

    sink_vals = x[:, :, sink_dims]                    # [L, K, num_sink_dims]
    layer_token_scores = sink_vals.amax(dim=-1)       # [L, K]
    # Use only the specified layer (default: second-to-last) instead of averaging all layers
    sink_scores = layer_token_scores[layer_index] # [K]

    sink_mask = sink_scores > threshold

    if key_is_pad is not None:
        key_is_pad = key_is_pad.cpu().bool().reshape(-1)
        if ignore_pad_tokens:
            sink_mask = sink_mask & (~key_is_pad)
    else:
        key_is_pad = torch.zeros(num_tokens, dtype=torch.bool)

    sink_token_indices = torch.nonzero(sink_mask, as_tuple=False).reshape(-1)

    if key_idx is not None:
        key_idx = key_idx.cpu().to(torch.long).reshape(-1)
        prompt_token_indices = key_idx[sink_token_indices].tolist()
    else:
        prompt_token_indices = sink_token_indices.tolist()

    return {
        "sink_dims": sink_dims,
        "threshold": float(threshold),
        "num_layers": int(num_layers),
        "num_graph_tokens": int(num_tokens),
        "sink_scores": sink_scores,
        "sink_token_indices": sink_token_indices.tolist(),
        "prompt_token_indices": prompt_token_indices,
        "is_pad": key_is_pad[sink_token_indices].tolist(),
    }


def plot_sink_token_index_histogram(
    *,
    all_sink_token_indices: List[List[int]],
    num_graph_tokens: int,
    save_path: str,
    dpi: int = 180,
) -> str:
    """
    Plot how often each graph-token index is detected as a sink across samples.

    x-axis: graph token index
    y-axis: frequency across all samples
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

    counts = torch.zeros(num_graph_tokens, dtype=torch.long)
    for sink_indices in all_sink_token_indices:
        for idx in sink_indices:
            idx = int(idx)
            if 0 <= idx < num_graph_tokens:
                counts[idx] += 1

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=dpi)
    x = torch.arange(num_graph_tokens)

    ax.bar(x.tolist(), counts.tolist(), width=0.8)
    ax.set_title("Distribution of detected sink graph-token indices")
    ax.set_xlabel("Graph token index")
    ax.set_ylabel("Frequency")
    ax.set_xticks(list(range(0, num_graph_tokens, 5)))
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_sink_distribution_shift(
    *,
    baseline_sink_token_indices: List[List[int]],
    post_sink_token_indices: List[List[int]],
    num_graph_tokens: int,
    save_path: str,
    dpi: int = 180,
    title: Optional[str] = None,
) -> str:
    """
    Overlay baseline vs post-prune sink-position distributions as line curves.

    Both inputs are per-sample lists of K-space sink indices; we aggregate each
    into a [K] count vector and draw the two distributions on shared axes so the
    pre/post shift is visible at a glance.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

    def _aggregate(per_sample):
        counts = torch.zeros(num_graph_tokens, dtype=torch.long)
        for sink_indices in per_sample:
            for idx in sink_indices:
                idx = int(idx)
                if 0 <= idx < num_graph_tokens:
                    counts[idx] += 1
        return counts

    base = _aggregate(baseline_sink_token_indices)
    post = _aggregate(post_sink_token_indices)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=dpi)
    x = torch.arange(num_graph_tokens)
    ax.plot(x.tolist(), base.tolist(), color="#1f77b4", marker="o", linewidth=2, label="baseline")
    ax.plot(x.tolist(), post.tolist(), color="#d62728", marker="x", linewidth=2, label="after pruning")
    ax.set_title(title if title is not None else "Sink position distribution: baseline vs post-prune")
    ax.set_xlabel("Graph token index")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")

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