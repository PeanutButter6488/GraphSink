import random
from typing import Any, Dict, Optional, Sequence
import torch


def find_spike_dims(hidden_states, key_is_pad, k=1):
    """
    hidden_states: [num_layers, num_graph_tokens, hidden_dim]
    key_is_pad: [num_graph_tokens] bool tensor, True means padded token

    Returns:
        spike_dims_per_layer: list of lists
            e.g. [[12, 98], [5, 77], ...]
    """
    valid_hidden_states = hidden_states[:, ~key_is_pad, :]
    num_layers = valid_hidden_states.shape[0]

    spike_dims_per_layer = []
    for layer_idx in range(num_layers):
        layer_hidden = valid_hidden_states[layer_idx]          # [K_valid, D]
        layer_mean = layer_hidden.mean(dim=0)                  # [D]
        topk_indices = torch.topk(layer_mean, k=k, largest=True).indices
        spike_dims_per_layer.append(topk_indices.tolist())

    return spike_dims_per_layer


def sink_dim_pruning(
    model,
    spike_dims_per_layer: Sequence[Sequence[int]],
    graph_token_positions: torch.Tensor,
) -> Dict[str, Any]:
    """
    For each layer, zero out that layer's spike dimensions on all graph tokens.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        layers = model.model.decoder.layers
    else:
        raise ValueError("Could not find transformer layers on the provided model.")

    position_ids = graph_token_positions.detach().to(torch.long).reshape(-1).cpu()
    selected_dims_per_layer = [sorted({int(dim) for dim in dims}) for dims in spike_dims_per_layer]

    handles = []

    for layer_idx, layer in enumerate(layers):
        if layer_idx >= len(selected_dims_per_layer):
            break

        dim_ids = torch.tensor(selected_dims_per_layer[layer_idx], dtype=torch.long)

        def hook(_module, _inputs, output, dim_ids=dim_ids, position_ids=position_ids):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden_states) or hidden_states.dim() != 3:
                return output

            cur_pos = position_ids.to(hidden_states.device)
            cur_pos = cur_pos[(cur_pos >= 0) & (cur_pos < hidden_states.shape[1])]

            cur_dims = dim_ids.to(hidden_states.device)
            cur_dims = cur_dims[(cur_dims >= 0) & (cur_dims < hidden_states.shape[2])]

            if cur_pos.numel() == 0 or cur_dims.numel() == 0:
                return output

            updated_hidden = hidden_states.clone()
            updated_hidden[:, cur_pos.unsqueeze(-1), cur_dims] = 0

            if isinstance(output, tuple):
                return (updated_hidden,) + output[1:]
            return updated_hidden

        handles.append(layer.register_forward_hook(hook))

    return {
        "selected_dims_per_layer": selected_dims_per_layer,
        "handles": handles,
    }


def nonsink_dim_pruning(
    model,
    spike_dims_per_layer: Sequence[Sequence[int]],
    hidden_dim: int,
    graph_token_positions: torch.Tensor,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    For each layer, randomly sample the same number of dimensions from non-spike
    dimensions, then zero them out on all graph tokens.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        layers = model.model.decoder.layers
    else:
        raise ValueError("Could not find transformer layers on the provided model.")

    rng = random.Random(seed)
    position_ids = graph_token_positions.detach().to(torch.long).reshape(-1).cpu()

    selected_dims_per_layer = []
    handles = []

    for layer_idx, layer in enumerate(layers):
        if layer_idx >= len(spike_dims_per_layer):
            break

        sink_dims = sorted({int(dim) for dim in spike_dims_per_layer[layer_idx]})
        nonsink_candidates = [dim for dim in range(int(hidden_dim)) if dim not in sink_dims]

        num_zeroout = len(sink_dims)
        if num_zeroout == 0:
            selected_dims = []
        else:
            selected_dims = sorted(rng.sample(nonsink_candidates, k=min(num_zeroout, len(nonsink_candidates))))

        selected_dims_per_layer.append(selected_dims)
        dim_ids = torch.tensor(selected_dims, dtype=torch.long)

        def hook(_module, _inputs, output, dim_ids=dim_ids, position_ids=position_ids):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden_states) or hidden_states.dim() != 3:
                return output

            cur_pos = position_ids.to(hidden_states.device)
            cur_pos = cur_pos[(cur_pos >= 0) & (cur_pos < hidden_states.shape[1])]

            cur_dims = dim_ids.to(hidden_states.device)
            cur_dims = cur_dims[(cur_dims >= 0) & (cur_dims < hidden_states.shape[2])]

            if cur_pos.numel() == 0 or cur_dims.numel() == 0:
                return output

            updated_hidden = hidden_states.clone()
            updated_hidden[:, cur_pos.unsqueeze(-1), cur_dims] = 0

            if isinstance(output, tuple):
                return (updated_hidden,) + output[1:]
            return updated_hidden

        handles.append(layer.register_forward_hook(hook))

    return {
        "selected_dims_per_layer": selected_dims_per_layer,
        "handles": handles,
    }


def remove_dim_pruning(handles: Sequence[Any]) -> None:
    for handle in handles:
        handle.remove()

# Token-wise pruning: prune entire hidden dimensions of sink tokens

def token_pruning(
    *,
    model,
    token_positions,
):
    """
    Zero out the full hidden vector for selected token positions
    at every decoder layer.
    """
    if isinstance(token_positions, torch.Tensor):
        token_positions = token_positions.detach().cpu().tolist()
    token_positions = [int(p) for p in token_positions]

    handles = []

    # def make_hook(cur_positions):
    #     def hook(module, inputs, output):
    #         if isinstance(output, tuple):
    #             hidden_states = output[0]
    #             rest = output[1:]
    #         else:
    #             hidden_states = output
    #             rest = None

    #         if hidden_states is None or len(cur_positions) == 0:
    #             return output

    #         pos = [p for p in cur_positions if 0 <= p < hidden_states.shape[1]]
    #         if len(pos) == 0:
    #             return output

    #         hidden_states = hidden_states.clone()
    #         hidden_states[:, pos, :] = 0.0

    #         if rest is None:
    #             return hidden_states
    #         return (hidden_states,) + rest

    #     return hook

    def make_hook(cur_positions):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            if hidden_states is None or len(cur_positions) == 0:
                return output

            pos = [p for p in cur_positions if 0 <= p < hidden_states.shape[1]]
            print("hook tensor shape:", hidden_states.shape)
            print("requested positions:", cur_positions)
            print("valid positions:", pos)

            if len(pos) == 0:
                return output

            hidden_states = hidden_states.clone()
            hidden_states[:, pos, :] = 0.0

            check = hidden_states[0, min(pos):max(pos)+1, :8].detach().float().cpu()
            print("local slice after zeroing:\n", check)

            if rest is None:
                return hidden_states
            return (hidden_states,) + rest
        return hook

    for layer in model.model.layers:
        h = layer.register_forward_hook(make_hook(token_positions))
        handles.append(h)

    return {
        "token_positions": token_positions,
        "handles": handles,
    }


# Helper function to find nonsink tokens:

def sample_nonsink_token_positions(graph_prompt_positions, sink_prompt_positions, num_to_prune=2, seed=0):
    graph_prompt_positions = sorted({int(x) for x in graph_prompt_positions})
    sink_prompt_positions = sorted({int(x) for x in sink_prompt_positions})
    nonsink_candidates = [
        pos for pos in graph_prompt_positions
        if pos not in sink_prompt_positions
    ]

    rng = random.Random(seed)
    selected = rng.sample(nonsink_candidates, k=min(num_to_prune, len(nonsink_candidates)))
    return sorted(selected)


def remove_token_pruning(handles):
    for h in handles:
        h.remove()