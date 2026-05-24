import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any


def plot_layerwise_token_stats_one_figure(
    *,
    per_layer_stats: Optional[list] = None,
    stats_txt_path: Optional[str] = None,
    save_dir: str,
    dataset_name: str = "",
    prefix: str = "layerwise_token_stats",
    dpi: int = 150,
) -> str:
    """
    Creates ONE figure showing mean/median/max over layers for each of the 5 graph tokens.

    You can provide either:
      (A) per_layer_stats: list of length L, each element dict with keys 'mean','median','max' (each shape [5])
      OR
      (B) stats_txt_path: the stats file produced by save_text2graph_dataset_heatmaps_with_stats()

    Output: saves PNG and returns its path.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Load data ---
    if per_layer_stats is None:
        if stats_txt_path is None:
            raise ValueError("Provide either per_layer_stats or stats_txt_path.")

        # Minimal parser for the format written by save_text2graph_dataset_heatmaps_with_stats
        means, medians, maxs = [], [], []
        with open(stats_txt_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines()]

        cur_layer = None
        for ln in lines:
            if ln.startswith("[Layer"):
                cur_layer = True
            elif cur_layer and ln.startswith("mean"):
                arr = ln.split(":", 1)[1].strip().split()
                means.append([float(x) for x in arr])
            elif cur_layer and ln.startswith("max"):
                arr = ln.split(":", 1)[1].strip().split()
                maxs.append([float(x) for x in arr])
            elif cur_layer and ln.startswith("median"):
                arr = ln.split(":", 1)[1].strip().split()
                medians.append([float(x) for x in arr])
                # done for this layer block

        mean_mat = np.array(means, dtype=np.float32)     # [L,5]
        median_mat = np.array(medians, dtype=np.float32) # [L,5]
        max_mat = np.array(maxs, dtype=np.float32)       # [L,5]
    else:
        L = len(per_layer_stats)
        mean_mat = np.stack([d["mean"] for d in per_layer_stats], axis=0).astype(np.float32)     # [L,5]
        median_mat = np.stack([d["median"] for d in per_layer_stats], axis=0).astype(np.float32) # [L,5]
        max_mat = np.stack([d["max"] for d in per_layer_stats], axis=0).astype(np.float32)       # [L,5]

    L, K = mean_mat.shape
    if K != 5:
        raise ValueError(f"Expected 5 graph tokens, got {K}.")

    x = np.arange(L)

    # --- Plot: 1x5 small multiples ---
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharey=True)
    if K == 1:
        axes = [axes]

    for k in range(5):
        ax = axes[k]
        ax.plot(x, mean_mat[:, k], linewidth=1.4, label="mean")
        ax.plot(x, median_mat[:, k], linewidth=1.4, label="median")
        ax.plot(x, max_mat[:, k], linewidth=1.4, label="max")

        ax.set_title(f"Graph token {k+1}", fontsize=16)
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.25)

        if k == 0:
            ax.set_ylabel("Cross Attention")

        # keep ticks readable
        ax.set_xticks([0, L // 2, L - 1])

    # Legend once, outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

    title_bits = []
    # if dataset_name:
    #     title_bits.append(str(dataset_name))
    #title_bits.append("Layerwise summary stats (mean / median / max)")
    fig.suptitle(" | ".join(title_bits), y=1.08, fontsize=16)

    fig.tight_layout()

    out_name = f"{prefix}_{dataset_name}.png" if dataset_name else f"{prefix}.png"
    out_path = os.path.join(save_dir, out_name)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return out_path

def plot_logit_lens_heatmap(
    *,
    top1_strings,        # List[List[str]] of shape [L][K]
    top1_probs,          # tensor or array of shape [L, K]
    save_path: str,
    sink_indices=None,   # iterable of K-axis indices to mark with an extra "<s>"
    cmap: str = "YlOrRd",
    dpi: int = 600,
    max_chars: int = 8,
    layer_stride: int = 2,
) -> str:
    probs = top1_probs.detach().cpu().numpy() if hasattr(top1_probs, "detach") else np.asarray(top1_probs)
    L_full, K = probs.shape

    sink_set = {int(s) for s in (sink_indices or [])}

    # Subsample layers: every `layer_stride`-th layer; always include the final layer.
    layer_indices = list(range(0, L_full, layer_stride))
    if layer_indices[-1] != L_full - 1:
        layer_indices.append(L_full - 1)
    L = len(layer_indices)
    probs = probs[layer_indices]
    strings = [top1_strings[l] for l in layer_indices]

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
    ax.set_yticklabels(
        [f"<g{k}><s>" if k in sink_set else f"<g{k}>" for k in range(K)],
        fontsize=16,
    )
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


if __name__ == "__main__":
    plot_layerwise_token_stats_one_figure(
        stats_txt_path=f"./analysis/arxiv/dataset_text2graph_attn/dataset_text_to_graph_arxiv_stats.txt",
        save_dir=f"./analysis/arxiv/dataset_text2graph_attn/",
        dataset_name=f"arxiv",
        prefix="layerwise_mean_median_max",
    )