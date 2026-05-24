from typing import Any, Dict

import torch


@torch.no_grad()
def compute_logit_lens(
    *,
    layer_graph_hidden_states: torch.Tensor,   # [L, K, D] — caller slices off embedding layer
    final_norm,                                # model.model.norm (LlamaRMSNorm)
    lm_head,                                   # model.lm_head (Linear V x D)
    tokenizer=None,
) -> Dict[str, Any]:
    """
    Project per-layer graph-token residuals through final_norm + lm_head and
    return the top-1 vocabulary token at each (layer, graph-token) cell.
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
