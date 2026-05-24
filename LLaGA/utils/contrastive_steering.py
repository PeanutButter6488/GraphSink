"""
Contrastive steering against structural activation sinks (RQ4).

Two inference-time interventions, both applied as a forward hook on a single
decoder layer. Given a sink direction vector ``s`` and a set of target prompt
positions, we modify the residual stream there:

  - mode='subtract':  h <- h - gamma * s
  - mode='project':   h <- (I - alpha * s_hat s_hat^T) h
                          = h - alpha * (h . s_hat) s_hat

The hook is bounds-checked, so when HF generate re-enters the layer at decode
steps with shorter hidden states (length 1 with a kv-cache), the prompt-relative
indices fall out of bounds and the hook becomes a no-op. This keeps the
intervention scoped to the prompt step only.
"""
from typing import Any, Dict, List, Optional, Sequence

import torch


def _resolve_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    raise ValueError("Could not find transformer layers on the provided model.")


def resolve_layer_indices(spec: str, num_layers: int) -> List[int]:
    """
    Parse --steering_layers into a sorted list of resolved (non-negative) indices.

      "all"         -> [0..num_layers-1]
      "5,10,15"     -> [5, 10, 15]      (negatives allowed: "-1,-2")
      "0-7"         -> [0, 1, ..., 7]   (inclusive; both endpoints non-negative)
      "-2"          -> [num_layers - 2]
    """
    s = spec.strip()
    if s.lower() == "all":
        return list(range(num_layers))
    if "-" in s and not s.startswith("-") and "," not in s:
        lo, hi = s.split("-")
        out = list(range(int(lo), int(hi) + 1))
    else:
        out = [int(x) for x in s.split(",") if x.strip()]
    out = [(i + num_layers) if i < 0 else i for i in out]
    for i in out:
        if not (0 <= i < num_layers):
            raise ValueError(f"layer index {i} out of range for {num_layers} layers (spec={spec!r})")
    return sorted(set(out))


def apply_contrastive_steering(
    model,
    sink_vectors: Optional[torch.Tensor],
    layer_indices: Sequence[int],
    target_positions: torch.Tensor,
    mode: str,
    strength: float,
    sink_positions: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Register a forward hook on each layer in ``layer_indices`` that perturbs
    the residual stream at ``target_positions``.

    Two sources for the steering direction ``s`` per hook:

      - global mode (``sink_positions is None``): ``sink_vectors[row]`` is a
        precomputed [D] vector for that layer. Same s for every sample.
      - per-sample mode (``sink_positions`` provided): the hook reads
        ``hidden_states[:, sink_positions, :]`` at the layer it sits on,
        means them, and uses the result as s for that sample/layer pair.
        Online — derived from the current forward pass, no second pass.
        ``sink_vectors`` is ignored and may be ``None``.

    layer_indices:    resolved (non-negative) decoder layer indices.
    target_positions: 1D long tensor of prompt positions to perturb.
    mode:             'subtract' or 'project'.
    strength:         gamma (subtract) or alpha in [0,1] (project).
    sink_positions:   1D long tensor of prompt positions to read s from.
    """
    if mode not in {"subtract", "project"}:
        raise ValueError(f"Unknown steering mode: {mode!r}")

    layers = _resolve_layers(model)
    if len(layer_indices) == 0:
        raise ValueError("layer_indices is empty")

    per_sample = sink_positions is not None
    if not per_sample:
        if sink_vectors is None or sink_vectors.dim() != 2 or sink_vectors.shape[0] != len(layer_indices):
            raise ValueError(
                f"sink_vectors must have shape [{len(layer_indices)}, D] "
                f"for global mode; got {None if sink_vectors is None else tuple(sink_vectors.shape)}"
            )

    pos_cpu = target_positions.detach().to(torch.long).reshape(-1).cpu()
    sink_pos_cpu = (
        sink_positions.detach().to(torch.long).reshape(-1).cpu() if per_sample else None
    )
    handles = []
    info_layers = []

    def apply_perturbation(h_slice, s_dev, s_hat_dev, mode, strength):
        if mode == "subtract":
            return h_slice - strength * s_dev
        coef = h_slice @ s_hat_dev  # [B, P]
        return h_slice - strength * coef.unsqueeze(-1) * s_hat_dev

    def make_global_hook(s: torch.Tensor, s_hat: Optional[torch.Tensor]):
        def hook(_module, _inputs, output, mode=mode, strength=float(strength),
                 s=s, s_hat=s_hat, pos_cpu=pos_cpu):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden_states) or hidden_states.dim() != 3:
                return output
            cur_pos = pos_cpu.to(hidden_states.device)
            cur_pos = cur_pos[(cur_pos >= 0) & (cur_pos < hidden_states.shape[1])]
            if cur_pos.numel() == 0 or strength == 0.0:
                return output
            orig_dtype = hidden_states.dtype
            h_slice = hidden_states[:, cur_pos, :].to(torch.float32)
            s_dev = s.to(hidden_states.device)
            s_hat_dev = s_hat.to(hidden_states.device) if s_hat is not None else None
            h_slice = apply_perturbation(h_slice, s_dev, s_hat_dev, mode, strength)
            updated = hidden_states.clone()
            updated[:, cur_pos, :] = h_slice.to(orig_dtype)
            return (updated,) + output[1:] if isinstance(output, tuple) else updated
        return hook

    def make_per_sample_hook():
        def hook(_module, _inputs, output, mode=mode, strength=float(strength),
                 pos_cpu=pos_cpu, sink_pos_cpu=sink_pos_cpu):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden_states) or hidden_states.dim() != 3:
                return output
            T = hidden_states.shape[1]
            cur_pos = pos_cpu.to(hidden_states.device)
            cur_pos = cur_pos[(cur_pos >= 0) & (cur_pos < T)]
            cur_sink = sink_pos_cpu.to(hidden_states.device)
            cur_sink = cur_sink[(cur_sink >= 0) & (cur_sink < T)]
            if cur_pos.numel() == 0 or cur_sink.numel() == 0 or strength == 0.0:
                return output
            orig_dtype = hidden_states.dtype
            # Derive s from the current sample's sink-token rows at this layer.
            # Mean across sinks within batch item 0 (eval is B=1).
            s = hidden_states[0, cur_sink, :].to(torch.float32).mean(dim=0)  # [D]
            if mode == "project":
                norm = torch.linalg.vector_norm(s)
                if float(norm.item()) == 0.0:
                    return output
                s_hat = s / norm
            else:
                s_hat = None
            h_slice = hidden_states[:, cur_pos, :].to(torch.float32)
            h_slice = apply_perturbation(h_slice, s, s_hat, mode, strength)
            updated = hidden_states.clone()
            updated[:, cur_pos, :] = h_slice.to(orig_dtype)
            return (updated,) + output[1:] if isinstance(output, tuple) else updated
        return hook

    for row, layer_idx in enumerate(layer_indices):
        if not (0 <= layer_idx < len(layers)):
            raise ValueError(f"layer_idx {layer_idx} out of range for {len(layers)} layers")
        if per_sample:
            handles.append(layers[layer_idx].register_forward_hook(make_per_sample_hook()))
            info_layers.append({"layer_idx": int(layer_idx), "source": "per_sample"})
        else:
            s = sink_vectors[row].detach().to(torch.float32).reshape(-1).cpu()
            if s.numel() == 0:
                raise ValueError(f"sink_vector for layer {layer_idx} is empty")
            if mode == "project":
                norm = float(torch.linalg.vector_norm(s).item())
                if norm == 0.0:
                    raise ValueError(f"sink_vector for layer {layer_idx} has zero norm")
                s_hat = s / norm
            else:
                s_hat = None
            handles.append(layers[layer_idx].register_forward_hook(make_global_hook(s, s_hat)))
            info_layers.append({
                "layer_idx": int(layer_idx),
                "source": "global",
                "sink_vector_norm": float(torch.linalg.vector_norm(s).item()),
            })

    return {
        "handles": handles,
        "info": {
            "mode": mode,
            "strength": float(strength),
            "num_target_positions": int(pos_cpu.numel()),
            "num_sink_positions": int(sink_pos_cpu.numel()) if per_sample else None,
            "source": "per_sample" if per_sample else "global",
            "layers": info_layers,
        },
    }


def remove_contrastive_steering(handles: Sequence[Any]) -> None:
    for h in handles:
        h.remove()


def select_target_positions(
    *,
    target: str,
    key_idx: torch.Tensor,
    query_idx: torch.Tensor,
    key_is_pad: Optional[torch.Tensor] = None,
    attention_mask_1d: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Resolve --steering_target into a 1D long tensor of positions.

      query       -> query_idx
      graph       -> key_idx (optionally drop pad slots)
      both        -> graph U query
      all_nonpad  -> every non-padding prompt position
    """
    target = target.lower()
    if target == "query":
        out = query_idx
    elif target == "graph":
        if key_is_pad is not None:
            out = key_idx[~key_is_pad.to(torch.bool)]
        else:
            out = key_idx
    elif target == "both":
        graph_pos = key_idx if key_is_pad is None else key_idx[~key_is_pad.to(torch.bool)]
        out = torch.cat([graph_pos, query_idx], dim=0)
    elif target == "all_nonpad":
        if attention_mask_1d is None:
            raise ValueError("--steering_target=all_nonpad needs attention_mask_1d")
        out = torch.nonzero(attention_mask_1d.to(torch.bool), as_tuple=False).reshape(-1)
    else:
        raise ValueError(f"Unknown steering target: {target!r}")

    return out.to(torch.long).reshape(-1)
