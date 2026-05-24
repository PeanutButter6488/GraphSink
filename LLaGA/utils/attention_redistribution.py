"""Redistribute attention via a monkey-patched LlamaAttention.forward.

Four directions are supported:
  - "src_to_sinks":               multiply the source column by (1-p); spread the removed mass evenly to sinks.
  - "sinks_to_top_nonsink":       multiply each sink column by (1-p); per-(layer,head,query) dump the pooled
                                  removed mass into the argmax over non-sink graph tokens.
  - "sinks_to_nonsink_even":      multiply each sink column by (1-p); spread pooled removed mass evenly across
                                  non-sink graph tokens.
  - "sinks_to_nonsink_value_sim": multiply each sink column by (1-p); softmax-weight the removed mass across
                                  non-sink graph tokens by V_j . Delta_i / ||V_j||, where
                                  Delta_i = sum_{k in S} A_ik V_k (the lost sink contribution to A_i V).
                                  At p=1 this matches the value-similarity algorithm exactly; at p<1 it
                                  partially scales sinks while routing p*m via value similarity.

`p` is a fraction in [0, 1] of each source/sink's *current* attention to redistribute. Scale-invariant
(unlike an absolute cap). All directions are mass-preserving (rows still sum to 1).

Targets transformers 4.31 (the env's pinned version): attention is computed inline inside
LlamaAttention.forward, so we replace that bound forward and tag each layer with its layer_idx.
"""

import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)


_VALID_DIRECTIONS = (
    "src_to_sinks",
    "sinks_to_top_nonsink",
    "sinks_to_nonsink_even",
    "sinks_to_nonsink_value_sim",
)

_STATE = {
    "active": False,
    "direction": "src_to_sinks",
    "source_idx": None,
    "sink_indices": None,
    "nonsink_indices": None,
    "fraction": None,
    "layer_filter": None,
}
_ORIGINAL_FORWARD = None


def _to_long_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.long).reshape(-1)
    return torch.tensor(list(x), dtype=torch.long)


def set_redistribution_state(
    direction: str,
    fraction: float,
    sink_indices,
    *,
    source_idx: Optional[int] = None,
    nonsink_indices=None,
    layer_filter: Optional[Iterable[int]] = None,
):
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of {_VALID_DIRECTIONS}, got {direction!r}")
    if direction == "src_to_sinks" and source_idx is None:
        raise ValueError("source_idx is required for direction='src_to_sinks'")
    if direction != "src_to_sinks" and nonsink_indices is None:
        raise ValueError(f"nonsink_indices is required for direction={direction!r}")
    if not (0.0 <= float(fraction) <= 1.0):
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    _STATE["active"] = True
    _STATE["direction"] = direction
    _STATE["fraction"] = float(fraction)
    _STATE["sink_indices"] = _to_long_tensor(sink_indices)
    _STATE["source_idx"] = int(source_idx) if source_idx is not None else None
    _STATE["nonsink_indices"] = _to_long_tensor(nonsink_indices) if nonsink_indices is not None else None
    _STATE["layer_filter"] = set(int(x) for x in layer_filter) if layer_filter is not None else None


def clear_redistribution_state():
    _STATE["active"] = False


def _redistribute_attn_inplace(attn_weights, value_states, layer_idx):
    """Mutate `attn_weights` in place per the active state. No-op when inactive or layer is filtered out.

    `value_states` has shape [B, H, K, head_dim] (post-repeat_kv). Only the value_sim direction reads it;
    other directions ignore it.
    """
    if not _STATE["active"]:
        return
    if _STATE["layer_filter"] is not None and layer_idx not in _STATE["layer_filter"]:
        return

    direction = _STATE["direction"]
    p = _STATE["fraction"]
    K = attn_weights.shape[-1]
    sinks = _STATE["sink_indices"].to(attn_weights.device)
    if sinks.numel() == 0 or int(sinks.max()) >= K or p == 0.0:
        return

    if direction == "src_to_sinks":
        src = _STATE["source_idx"]
        if not (0 <= src < K):
            return
        src_col = attn_weights[..., src]
        excess = p * src_col                                       # [B, H, Q]
        share = (excess / sinks.numel()).unsqueeze(-1)             # [B, H, Q, 1]
        attn_weights[..., src] = src_col - excess
        attn_weights[..., sinks] = attn_weights[..., sinks] + share
        return

    # sinks_to_*: per-sink fraction; pool the removed mass.
    nonsinks = _STATE["nonsink_indices"].to(attn_weights.device)
    if nonsinks.numel() == 0 or int(nonsinks.max()) >= K:
        return
    sink_cols = attn_weights[..., sinks]                           # [B, H, Q, N_sinks]
    removed = p * sink_cols                                        # [B, H, Q, N_sinks]
    total_excess = removed.sum(dim=-1)                             # [B, H, Q]

    if direction == "sinks_to_nonsink_value_sim":
        # Delta_i = sum_{k in S} A_ik V_k     (uses ORIGINAL A, since direction of Delta is what matters)
        V_S = value_states[..., sinks, :]                          # [B, H, |S|, d]
        V_R = value_states[..., nonsinks, :]                       # [B, H, |R|, d]
        Delta = torch.matmul(sink_cols, V_S)                       # [B, H, Q, d]
        scores = torch.matmul(Delta, V_R.transpose(-1, -2))        # [B, H, Q, |R|]
        scores = scores / V_R.norm(dim=-1).clamp(min=1e-8).unsqueeze(-2)
        w = torch.softmax(scores.float(), dim=-1).to(attn_weights.dtype)  # [B, H, Q, |R|]
        attn_weights[..., sinks] = sink_cols - removed
        attn_weights[..., nonsinks] = (
            attn_weights[..., nonsinks] + total_excess.unsqueeze(-1) * w
        )
        return

    attn_weights[..., sinks] = sink_cols - removed
    if direction == "sinks_to_nonsink_even":
        share = (total_excess / nonsinks.numel()).unsqueeze(-1)   # [B, H, Q, 1]
        attn_weights[..., nonsinks] = attn_weights[..., nonsinks] + share
    else:  # sinks_to_top_nonsink
        nonsink_attn = attn_weights[..., nonsinks]                # [B, H, Q, N_nonsink]
        local_top = torch.argmax(nonsink_attn, dim=-1)            # [B, H, Q]
        global_top = nonsinks[local_top]                          # [B, H, Q]
        attn_weights.scatter_add_(
            -1, global_top.unsqueeze(-1), total_excess.unsqueeze(-1)
        )


def _patched_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
):
    """Drop-in replacement for transformers 4.31 LlamaAttention.forward with redistribution after softmax."""
    bsz, q_len, _ = hidden_states.size()

    if self.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
        query_states = torch.cat([F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)], dim=-1)
        key_states = torch.cat([F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)], dim=-1)
        value_states = torch.cat([F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)], dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    _redistribute_attn_inplace(attn_weights, value_states, getattr(self, "layer_idx", -1))

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def install_redistribution(model):
    """Replace LlamaAttention.forward (class method) with the redistribution version, and tag layer indices."""
    global _ORIGINAL_FORWARD
    if _ORIGINAL_FORWARD is None:
        _ORIGINAL_FORWARD = LlamaAttention.forward
    LlamaAttention.forward = _patched_llama_attention_forward
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = i


def uninstall_redistribution(model):
    global _ORIGINAL_FORWARD
    if _ORIGINAL_FORWARD is not None:
        LlamaAttention.forward = _ORIGINAL_FORWARD
        _ORIGINAL_FORWARD = None
    clear_redistribution_state()
