"""RQ3 plots: sink position distribution shift — baseline vs after pruning.

For each (GLM, dataset) we read two per-sample JSONL records and overlay
their sink position distributions as Gaussian-smoothed *frequency* curves
(absolute counts, not unit-mass density — so panels show both how the
distribution shifts *and* how many sinks were detected):

  - baseline: ``sink_records.jsonl`` ``top2_sink_token_indices``
              (LLaGA / TEA-GLM) or ``sink_token_local_indices`` (InstructGLM)
  - after pruning: ``sink_reoccur.jsonl`` ``reoccur_sink_token_indices``

This is the curve-style counterpart to the per-cell ``sink_distribution_shift.png``
already emitted by each GLM's eval script (which uses dotted line plots).

Outputs (under ``utils/plots/rq3/``):
    - ``<glm>_<dataset>_sink_shift_<task>.png``   (one panel per cell)
    - ``rq3_sink_shift_grid_<task>.png``          (main-paper grid)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(REPO_ROOT, "utils", "plots", "rq3")

GLMS: Tuple[str, ...] = ("InstructGLM", "LLaGA", "TEA-GLM")
# Main-paper grid pairs LLaGA with TEA-GLM. InstructGLM has no reoccur records
# on disk; its loader returns None and the per-cell single panels above still
# emit baseline-only previews for the appendix once data lands.
MAIN_PAPER_GLMS: Tuple[str, ...] = ("LLaGA", "TEA-GLM")
DATASETS: Tuple[str, ...] = ("cora", "arxiv", "pubmed")
DATASET_LABELS: Dict[str, str] = {"cora": "Cora", "arxiv": "ArXiv", "pubmed": "PubMed"}
TASKS: Tuple[str, ...] = ("nc",)

LLAGA_NUM_GRAPH_TOKENS_BY_TASK = {"nc": 111, "lp": 222}
TEAGLM_NUM_GRAPH_TOKENS = 5

# Fixed colours for the two distributions — matches the colour pair in the
# legacy sink_distribution_shift.png so RQ3 reads consistently with prior figures.
BASELINE_COLOR = "#1f77b4"  # blue
POSTPRUNE_COLOR = "#d62728"  # red

# Gaussian-kernel bandwidth (sigma) for the smoothed-count curve, in graph-token
# units. Scaled with K so TEA-GLM (K=5) doesn't get oversmoothed while LLaGA
# (K=111) still draws as a clean curve. The smoothed values stay in count
# units (kernel integrates to 1 in continuous x), so peak heights remain
# comparable to the raw histograms in <analysis>/.../sink_*_distribution.png.
def _smoothing_sigma(num_graph_tokens: int) -> float:
    return max(0.5, 0.02 * num_graph_tokens)

TEAGLM_PREFIX_BY_TASK: Dict[str, str] = {
    "nc": "TEA-GLM_citation_meanpool_seed42",
    "lp": "TEA-GLM_citation_meanpool_seed42",
}
# Reoccur records live under a different run prefix (the prune-top2 sweep) and
# carry both baseline + post-prune indices inline, so we read post indices
# from this file via the ``post_sink_indices`` field.
TEAGLM_REOCCUR_PREFIX_BY_TASK: Dict[str, str] = {
    "nc": "TEA-GLM_citation_meanpool_prune_top2_seed42",
}


@dataclass
class ShiftCell:
    """Sink indices loaded for one (GLM, dataset) cell."""

    baseline: List[int]            # flat list of all sink indices (across samples)
    post_prune: List[int]          # flat list of reoccur sink indices (across samples)
    num_graph_tokens: int


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _llaga_dir(dataset: str, task: str) -> str:
    suffix = "ND" if task == "nc" else "ND_LP"
    return os.path.join(REPO_ROOT, "LLaGA", "analysis", f"{dataset}_{suffix}")


def _instructglm_dir(dataset: str, task: str) -> Optional[str]:
    if task != "nc":
        return None
    return os.path.join(REPO_ROOT, "InstructGLM", "analysis", "instructglm", dataset)


def _teaglm_dir(dataset: str, task: str) -> str:
    suffix = "" if task == "nc" else "_lp"
    return os.path.join(REPO_ROOT, "TEA-GLM", "analysis", f"{dataset}{suffix}", "global_stats")


def _flatten(jsonl_path: str, field: str) -> List[int]:
    out: List[int] = []
    if not os.path.exists(jsonl_path):
        return out
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for idx in rec.get(field, []):
                out.append(int(idx))
    return out


def _load_llaga(dataset: str, task: str = "nc") -> Optional[ShiftCell]:
    base_dir = _llaga_dir(dataset, task)
    baseline = _flatten(os.path.join(base_dir, "sink_records.jsonl"), "top2_sink_token_indices")
    post = _flatten(os.path.join(base_dir, "sink_reoccur.jsonl"), "reoccur_sink_token_indices")
    if not baseline and not post:
        return None
    return ShiftCell(baseline, post, LLAGA_NUM_GRAPH_TOKENS_BY_TASK[task])


def _load_instructglm(dataset: str, task: str = "nc") -> Optional[ShiftCell]:
    base_dir = _instructglm_dir(dataset, task)
    if base_dir is None:
        return None
    sink_path = os.path.join(base_dir, "sink_records.jsonl")
    reoccur_path = os.path.join(base_dir, "sink_reoccur.jsonl")
    if not (os.path.exists(sink_path) or os.path.exists(reoccur_path)):
        return None
    baseline = _flatten(sink_path, "sink_token_local_indices")
    post = _flatten(reoccur_path, "reoccur_sink_token_indices")
    if not baseline and not post:
        return None
    # Recover num_graph_tokens from the records.
    num_graph_tokens = 0
    for path, field in ((sink_path, "num_graph_tokens"), (reoccur_path, "num_graph_tokens")):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                num_graph_tokens = max(num_graph_tokens, int(rec.get(field, 0)))
    if num_graph_tokens == 0:
        # Fall back to the largest observed index.
        all_idx = baseline + post
        num_graph_tokens = (max(all_idx) + 1) if all_idx else 0
    if num_graph_tokens == 0:
        return None
    return ShiftCell(baseline, post, num_graph_tokens)


def _load_teaglm(dataset: str, task: str = "nc") -> Optional[ShiftCell]:
    base_dir = _teaglm_dir(dataset, task)
    sink_path = os.path.join(
        base_dir, f"{TEAGLM_PREFIX_BY_TASK[task]}_sink_records.jsonl",
    )
    reoccur_prefix = TEAGLM_REOCCUR_PREFIX_BY_TASK.get(task)
    reoccur_path = (
        os.path.join(base_dir, f"{reoccur_prefix}_sink_reoccur_records.jsonl")
        if reoccur_prefix is not None else ""
    )
    if not (os.path.exists(sink_path) or os.path.exists(reoccur_path)):
        return None
    baseline = _flatten(sink_path, "top2_sink_token_indices")
    post = _flatten(reoccur_path, "post_sink_indices")
    if not baseline and not post:
        return None
    return ShiftCell(baseline, post, TEAGLM_NUM_GRAPH_TOKENS)


LOADERS: Dict[str, Callable[[str, str], Optional[ShiftCell]]] = {
    "LLaGA": _load_llaga,
    "InstructGLM": _load_instructglm,
    "TEA-GLM": _load_teaglm,
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 22,
        "axes.titlesize": 26,
        "axes.labelsize": 26,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 20,
        "legend.title_fontsize": 22,
        "axes.linewidth": 1.4,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })


def _xtick_step(num_graph_tokens: int) -> int:
    if num_graph_tokens <= 10:
        return 1
    if num_graph_tokens <= 30:
        return 5
    if num_graph_tokens <= 60:
        return 10
    if num_graph_tokens <= 120:
        return 20
    if num_graph_tokens <= 200:
        return 30
    return 50


def _smoothed_count_curve(
    samples: List[int],
    *,
    num_graph_tokens: int,
    grid_points: int = 600,
    sigma: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Smoothed *count* curve: drop a unit-mass Gaussian at each integer bin
    (weighted by that bin's hit count) and evaluate the sum on a dense grid.

    Why kernel-sum on a dense grid (not gaussian_filter1d + np.interp)?
        Filtering then linearly interpolating gives a piecewise-linear curve
        with visible kinks when K is small (TEA-GLM, K=5). Summing Gaussians
        directly on a dense x-grid is C^∞ smooth at every K, so LLaGA (K=111)
        and TEA-GLM (K=5) both render as proper curves.

    Why count units (not density)?
        gaussian_kde would normalise each curve to integral 1 and hide the
        post-prune *magnitude* (e.g. TEA-GLM detects far fewer sinks after
        pruning). Keeping the kernel mass = 1 in continuous x means the
        y-values are counts-per-token-unit, and area-under-curve ≈ total
        sink occurrences — directly comparable to the raw bar histograms.
    """
    if not samples:
        return None
    K = num_graph_tokens
    counts = np.zeros(K, dtype=np.float64)
    for idx in samples:
        i = int(idx)
        if 0 <= i < K:
            counts[i] += 1
    if sigma is None:
        sigma = _smoothing_sigma(K)

    xs = np.linspace(0, K - 1, grid_points)
    bin_centers = np.arange(K, dtype=np.float64)
    diffs = xs[:, None] - bin_centers[None, :]   # [G, K]
    kernel = np.exp(-0.5 * (diffs / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    ys = kernel @ counts
    return xs, ys


def _plot_panel(
    ax: plt.Axes,
    cell: ShiftCell,
    *,
    xlabel: Optional[str] = "Graph token index",
    ylabel: Optional[str] = "Frequency",
    show_legend: bool = False,
    legend_title: Optional[str] = None,
    ylabel_kwargs: Optional[Dict] = None,
) -> List:
    """Draw one frequency panel; return the line handles for legend reuse."""
    K = cell.num_graph_tokens

    handles: List = []
    labels: List[str] = []
    for samples, color, label in (
        (cell.baseline, BASELINE_COLOR, "baseline"),
        (cell.post_prune, POSTPRUNE_COLOR, "after pruning"),
    ):
        curve = _smoothed_count_curve(samples, num_graph_tokens=K)
        if curve is None:
            continue
        xs, ys = curve
        ax.fill_between(xs, ys, color=color, alpha=0.35, linewidth=0)
        line, = ax.plot(xs, ys, color=color, linewidth=2.0, label=label)
        handles.append(line)
        labels.append(label)

    ax.set_xlim(0, K - 1)
    ax.set_ylim(bottom=0)
    step = _xtick_step(K)
    ax.set_xticks(list(range(0, K, step)))

    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel, **(ylabel_kwargs or {}))
    if show_legend and handles:
        ax.legend(
            handles, labels,
            title=legend_title,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
        )
    return handles


def plot_single(glm: str, dataset: str, cell: ShiftCell, save_path: str) -> str:
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    title = f"{DATASET_LABELS.get(dataset, dataset)} — {glm}: sink position frequency"
    ax.set_title(title, pad=10)
    _plot_panel(
        ax, cell,
        xlabel="Graph token index",
        ylabel="Frequency",
        show_legend=True,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_grid(
    cells: Dict[Tuple[str, str], Optional[ShiftCell]],
    save_path: str,
    *,
    glms: Tuple[str, ...] = MAIN_PAPER_GLMS,
    datasets: Tuple[str, ...] = DATASETS,
) -> str:
    """Combined LLaGA + TEA-GLM grid.

    Layout follows the user's main-paper spec: row labels are the GLM names
    (no "Frequency"), a single "Graph token index" sits under the bottom-center
    panel, and the figure-level legend just lists "baseline" / "after pruning"
    with no title.
    """
    _apply_paper_style()
    nrows, ncols = len(glms), len(datasets)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7.0 * ncols, 4.0 * nrows),
        squeeze=False,
    )

    legend_handles: List = []

    middle_col = ncols // 2

    for i, glm in enumerate(glms):
        for j, dataset in enumerate(datasets):
            ax = axes[i][j]
            cell = cells.get((glm, dataset))
            is_bottom_row = (i == nrows - 1)
            is_left_col = (j == 0)
            is_bottom_center = is_bottom_row and (j == middle_col)

            ylabel = glm if is_left_col else None
            ylabel_kwargs = (
                {"fontweight": "bold", "fontsize": 26, "labelpad": 14}
                if is_left_col else None
            )
            xlabel = "Graph token index" if is_bottom_center else None

            if cell is None:
                ax.text(
                    0.5, 0.5, "n/a",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=24, color="#888888",
                )
                ax.set_xticks([]); ax.set_yticks([])
                if ylabel:
                    ax.set_ylabel(ylabel, **(ylabel_kwargs or {}))
                if xlabel:
                    ax.set_xlabel(xlabel)
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
            else:
                handles = _plot_panel(
                    ax, cell,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    ylabel_kwargs=ylabel_kwargs,
                    show_legend=False,
                )
                if not legend_handles and handles:
                    legend_handles = handles

            if i == 0:
                ax.set_title(DATASET_LABELS.get(dataset, dataset), pad=10)

    if legend_handles:
        fig.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(legend_handles),
            frameon=False,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_all_cells(
    glms: Tuple[str, ...] = GLMS,
    datasets: Tuple[str, ...] = DATASETS,
    *,
    task: str = "nc",
) -> Dict[Tuple[str, str], Optional[ShiftCell]]:
    cells: Dict[Tuple[str, str], Optional[ShiftCell]] = {}
    for glm in glms:
        loader = LOADERS[glm]
        for dataset in datasets:
            cells[(glm, dataset)] = loader(dataset, task)
    return cells


def _emit_for_task(task: str) -> None:
    cells = load_all_cells(task=task)

    for (glm, dataset), cell in cells.items():
        if cell is None:
            print(f"[skip] {glm} / {dataset} ({task}): no reoccur records")
            continue
        slug = glm.lower().replace('-', '')
        out = os.path.join(PLOTS_DIR, f"{slug}_{dataset}_sink_shift_{task}.png")
        plot_single(glm, dataset, cell, out)
        print(
            f"[ok]   {glm} / {dataset} ({task}): "
            f"baseline={len(cell.baseline)} post={len(cell.post_prune)} -> {out}"
        )

    grid_path = os.path.join(PLOTS_DIR, f"rq3_sink_shift_grid_{task}.png")
    plot_grid(cells, grid_path, glms=MAIN_PAPER_GLMS)
    print(f"[ok]   grid (main paper, {MAIN_PAPER_GLMS}, {task}) -> {grid_path}")


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for task in TASKS:
        print(f"\n=== task: {task} ===")
        _emit_for_task(task)


if __name__ == "__main__":
    main()
