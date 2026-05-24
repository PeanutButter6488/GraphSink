"""RQ2 plots: cross-attention from query tokens to graph tokens.

For each task (node classification + link prediction), we emit:

    rq2_query_to_graph_grid_{nc,lp}.png   — Q x K (query offset vs graph token)
    rq2_layer_to_graph_grid_{nc,lp}.png   — L x K (transformer layer vs graph token)

The aggregated grids that go into the main paper show only LLaGA + TEA-GLM
(InstructGLM stays in the appendix via per-cell single panels, and only has
NC data anyway). Inputs are the RQ-arrays each architecture dumps next to
its existing plotting outputs (see the patches in LLaGA/eval/eval_pretrain.py,
InstructGLM/sink_analysis.py, and TEA-GLM/train_glm.py).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(REPO_ROOT, "utils", "plots", "rq2")

GLMS: Tuple[str, ...] = ("InstructGLM", "LLaGA", "TEA-GLM")
# The aggregated grids for the main paper drop InstructGLM (no LP data on
# disk, and the paper relegates it to the appendix). Per-cell single panels
# are still emitted for every GLM so the appendix can use them.
MAIN_PAPER_GLMS: Tuple[str, ...] = ("LLaGA", "TEA-GLM")
DATASETS: Tuple[str, ...] = ("cora", "arxiv", "pubmed")
DATASET_LABELS: Dict[str, str] = {"cora": "Cora", "arxiv": "ArXiv", "pubmed": "PubMed"}
# Tasks we plot for. InstructGLM has no LP arrays — its LP loader returns None.
TASKS: Tuple[str, ...] = ("nc", "lp")

# TEA-GLM stores its npy/json under a run-specific prefix. seed42 is the
# multi-seed run currently on disk for both tasks.
TEAGLM_PREFIX_BY_TASK: Dict[str, str] = {
    "nc": "TEA-GLM_citation_meanpool_seed42",
    "lp": "TEA-GLM_citation_meanpool_seed42",
}

# Drop query rows whose contributing-sample count falls below this threshold.
# Without it, InstructGLM panels show meaningless noise at very large query
# offsets where only a handful of samples ever reached.
MIN_SAMPLES_PER_QUERY_ROW = 30

# JSONL of per-sample (nonpad %, top-2 sink avg attention), produced by
# LLaGA/eval/eval_pretrain.py for each `{dataset}_ND` analysis folder.
LLAGA_TOP2_SINK_RECORDS_NAME = "top2_sink_attention_nonpad_records.jsonl"


@dataclass
class CrossAttnCell:
    query_to_graph: np.ndarray            # [Q, K] sample-averaged
    layer_to_graph: np.ndarray            # [L, K] sample-averaged
    query_count: Optional[np.ndarray]     # [Q] or [Q, K] per-row sample count, or None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _safe_load(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    return np.load(path) if os.path.exists(path) else None


# Per-task analysis-directory suffixes. Matches plot_rq1's convention.
def _llaga_rq_dir(dataset: str, task: str) -> str:
    suffix = "ND" if task == "nc" else "ND_LP"
    return os.path.join(REPO_ROOT, "LLaGA", "analysis", f"{dataset}_{suffix}", "rq_arrays")


def _instructglm_rq_dir(dataset: str, task: str) -> Optional[str]:
    if task != "nc":
        return None  # No LP rq_arrays on disk for InstructGLM.
    return os.path.join(
        REPO_ROOT, "InstructGLM", "analysis", "instructglm", dataset, "rq_arrays",
    )


def _teaglm_rq_dir(dataset: str, task: str) -> str:
    suffix = "" if task == "nc" else "_lp"
    return os.path.join(
        REPO_ROOT, "TEA-GLM", "analysis", f"{dataset}{suffix}", "global_stats", "rq_arrays",
    )


def _load_llaga(dataset: str, task: str = "nc") -> Optional[CrossAttnCell]:
    base = _llaga_rq_dir(dataset, task)
    qg = _safe_load(os.path.join(base, "query_to_graph.npy"))
    lg = _safe_load(os.path.join(base, "layer_to_graph.npy"))
    if qg is None or lg is None:
        return None
    return CrossAttnCell(query_to_graph=qg, layer_to_graph=lg, query_count=None)


def _load_instructglm(dataset: str, task: str = "nc") -> Optional[CrossAttnCell]:
    base = _instructglm_rq_dir(dataset, task)
    if base is None:
        return None
    qg = _safe_load(os.path.join(base, "query_to_graph.npy"))
    lg = _safe_load(os.path.join(base, "layer_to_graph.npy"))
    qc = _safe_load(os.path.join(base, "query_to_graph_count.npy"))
    if qg is None or lg is None:
        return None
    return CrossAttnCell(query_to_graph=qg, layer_to_graph=lg, query_count=qc)


def _load_teaglm(dataset: str, task: str = "nc") -> Optional[CrossAttnCell]:
    base = _teaglm_rq_dir(dataset, task)
    prefix = TEAGLM_PREFIX_BY_TASK[task]
    qg = _safe_load(os.path.join(base, f"{prefix}_query_to_graph.npy"))
    lg = _safe_load(os.path.join(base, f"{prefix}_layer_to_graph.npy"))
    qc = _safe_load(os.path.join(base, f"{prefix}_query_to_graph_count.npy"))
    if qg is None and lg is None:
        # Fallback: TEA-GLM also writes Q x K in the neighboring JSON file.
        # Useful for any dataset whose npy hasn't been regenerated yet.
        suffix = "" if task == "nc" else "_lp"
        json_path = os.path.join(
            REPO_ROOT, "TEA-GLM", "analysis", f"{dataset}{suffix}", "global_stats",
            f"{prefix}_query_to_graph_attention.json",
        )
        if os.path.exists(json_path):
            with open(json_path) as f:
                d = json.load(f)
            qg = np.asarray(d["mean_q_to_g"], dtype=np.float32)
            qc = np.asarray(d["count_q_to_g"], dtype=np.int64)
            return CrossAttnCell(query_to_graph=qg, layer_to_graph=None, query_count=qc)  # type: ignore[arg-type]
        return None
    if qg is None or lg is None:
        return None
    return CrossAttnCell(query_to_graph=qg, layer_to_graph=lg, query_count=qc)


LOADERS: Dict[str, Callable[[str, str], Optional[CrossAttnCell]]] = {
    "LLaGA": _load_llaga,
    "InstructGLM": _load_instructglm,
    "TEA-GLM": _load_teaglm,
}


def _trim_query_rows(qg: np.ndarray, qc: Optional[np.ndarray]) -> np.ndarray:
    """Drop trailing query rows whose sample count falls below the threshold.

    Some pipelines store ``count`` as ``[Q]`` (TEA-GLM, one sample per row),
    others as ``[Q, K]`` (InstructGLM, per-cell). Both collapse to a per-row
    count via max.
    """
    if qc is None:
        return qg
    if qc.ndim == 2:
        per_row = qc.max(axis=1)
    else:
        per_row = qc
    valid = per_row >= MIN_SAMPLES_PER_QUERY_ROW
    if not valid.any():
        return qg
    last = int(np.where(valid)[0].max()) + 1
    return qg[:last]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _apply_paper_style() -> None:
    """Match plot_rq1's paper style so RQ1 and RQ2 figures are visually
    consistent in the paper."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 22,
        "axes.titlesize": 26,
        "axes.labelsize": 26,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 22,
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


def _y_three_ticks(n: int) -> list:
    """Just three y-ticks: start, middle, end. Keeps the panels uncluttered for
    paper figures where the axis is meant to be read at a glance."""
    if n <= 1:
        return [0]
    if n == 2:
        return [0, 1]
    return [0, n // 2, n - 1]


def _heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
    xlabel: str,
    ylabel: str,
    cmap: str = "viridis",
) -> "matplotlib.image.AxesImage":
    Y, X = mat.shape
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap)

    xs = _xtick_step(X)
    ax.set_xticks(list(range(0, X, xs)))
    ax.set_yticks(_y_three_ticks(Y))

    if show_xlabel:
        ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(ylabel)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return im


def _draw_na(ax: plt.Axes) -> None:
    ax.text(
        0.5, 0.5, "n/a",
        ha="center", va="center",
        transform=ax.transAxes,
        fontsize=24, color="#888888",
    )
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _annotate_row_label(ax: plt.Axes, glm: str) -> None:
    ax.annotate(
        glm,
        xy=(-0.30, 0.5),
        xycoords="axes fraction",
        rotation=90,
        ha="center", va="center",
        fontsize=26, fontweight="bold",
    )


def plot_query_to_graph_grid(
    cells: Dict[Tuple[str, str], Optional[CrossAttnCell]],
    save_path: str,
    *,
    glms: Tuple[str, ...] = MAIN_PAPER_GLMS,
    datasets: Tuple[str, ...] = DATASETS,
) -> str:
    _apply_paper_style()
    nrows, ncols = len(glms), len(datasets)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.5 * ncols, 4.0 * nrows),
        squeeze=False,
    )

    for i, glm in enumerate(glms):
        for j, dataset in enumerate(datasets):
            ax = axes[i][j]
            cell = cells.get((glm, dataset))
            if cell is None or cell.query_to_graph is None:
                _draw_na(ax)
            else:
                qg = _trim_query_rows(cell.query_to_graph, cell.query_count)
                im = _heatmap(
                    ax, qg,
                    show_xlabel=(i == nrows - 1),
                    show_ylabel=(j == 0),
                    xlabel="Graph token index",
                    ylabel="Query offset",
                )
                fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

            if i == 0:
                ax.set_title(DATASET_LABELS.get(dataset, dataset), pad=10)
            if j == 0:
                _annotate_row_label(ax, glm)

    fig.tight_layout(rect=[0.04, 0, 1, 1])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_layer_to_graph_grid(
    cells: Dict[Tuple[str, str], Optional[CrossAttnCell]],
    save_path: str,
    *,
    glms: Tuple[str, ...] = MAIN_PAPER_GLMS,
    datasets: Tuple[str, ...] = DATASETS,
) -> str:
    _apply_paper_style()
    nrows, ncols = len(glms), len(datasets)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.5 * ncols, 4.0 * nrows),
        squeeze=False,
    )

    for i, glm in enumerate(glms):
        for j, dataset in enumerate(datasets):
            ax = axes[i][j]
            cell = cells.get((glm, dataset))
            if cell is None or cell.layer_to_graph is None:
                _draw_na(ax)
            else:
                im = _heatmap(
                    ax, cell.layer_to_graph,
                    show_xlabel=(i == nrows - 1),
                    show_ylabel=(j == 0),
                    xlabel="Graph token index",
                    ylabel="Transformer layer",
                )
                fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

            if i == 0:
                ax.set_title(DATASET_LABELS.get(dataset, dataset), pad=10)
            if j == 0:
                _annotate_row_label(ax, glm)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_single_query_to_graph(glm: str, dataset: str, cell: CrossAttnCell, save_path: str) -> str:
    _apply_paper_style()
    qg = _trim_query_rows(cell.query_to_graph, cell.query_count)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = _heatmap(
        ax, qg,
        show_xlabel=True, show_ylabel=True,
        xlabel="Graph token index",
        ylabel="Query offset",
    )
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_single_layer_to_graph(glm: str, dataset: str, cell: CrossAttnCell, save_path: str) -> str:
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = _heatmap(
        ax, cell.layer_to_graph,
        show_xlabel=True, show_ylabel=True,
        xlabel="Graph token index",
        ylabel="Transformer layer",
    )
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# LLaGA top-2 sink attention vs non-pad %
# ---------------------------------------------------------------------------

def _llaga_top2_sink_records_by_dataset() -> Dict[str, str]:
    """Discover every LLaGA `{dataset}_ND` folder that has the top-2-sink
    records JSONL on disk, and return a {dataset: records_path} mapping."""
    base = os.path.join(REPO_ROOT, "LLaGA", "analysis")
    out: Dict[str, str] = {}
    if not os.path.isdir(base):
        return out
    for entry in sorted(os.listdir(base)):
        if not entry.endswith("_ND"):
            continue
        records = os.path.join(base, entry, LLAGA_TOP2_SINK_RECORDS_NAME)
        if os.path.isfile(records):
            out[entry[: -len("_ND")]] = records
    return out


def _load_top2_sink_records(path: str) -> Optional[Dict[str, np.ndarray]]:
    xs: list = []
    ys: list = []
    n_total = 0
    n_bad = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Some records files (e.g. citeseer_ND, wikics_ND) contain
                # truncated lines from concurrent writes. Skip them.
                n_bad += 1
                continue
            if not bool(rec.get("valid", False)):
                continue
            x = rec.get("nonpad_percentage")
            y = rec.get("top2_sink_avg_attention")
            if x is None or y is None:
                continue
            xs.append(float(x))
            ys.append(float(y))
    if not xs:
        return None
    return {
        "x": np.asarray(xs, dtype=np.float32),
        "y": np.asarray(ys, dtype=np.float32),
        "n_valid": np.int64(len(xs)),
        "n_total": np.int64(n_total),
        "n_bad": np.int64(n_bad),
    }


def _scatter_top2_sink(
    ax: plt.Axes,
    agg: Dict[str, np.ndarray],
    *,
    point_alpha: float,
    point_size: int,
) -> None:
    ax.scatter(
        agg["x"], agg["y"],
        alpha=point_alpha,
        s=point_size,
        edgecolors="none",
    )
    ax.grid(True, linestyle="--", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_llaga_top2_sink_vs_nonpad(
    dataset: str,
    agg: Dict[str, np.ndarray],
    save_path: str,
    *,
    point_alpha: float = 0.35,
    point_size: int = 36,
) -> str:
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    _scatter_top2_sink(ax, agg, point_alpha=point_alpha, point_size=point_size)
    ax.set_xlabel("Non-padded graph tokens (%)")
    ax.set_ylabel("Average Attention")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_llaga_top2_sink_vs_nonpad_grid(
    aggs: Dict[str, Optional[Dict[str, np.ndarray]]],
    save_path: str,
    *,
    datasets: Tuple[str, ...] = ("arxiv", "cora", "pubmed"),
    point_alpha: float = 0.35,
    point_size: int = 36,
) -> str:
    _apply_paper_style()
    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(6.5 * len(datasets), 5.0),
        squeeze=False,
    )
    for j, dataset in enumerate(datasets):
        ax = axes[0][j]
        agg = aggs.get(dataset)
        if agg is None:
            _draw_na(ax)
        else:
            _scatter_top2_sink(ax, agg, point_alpha=point_alpha, point_size=point_size)
            ax.set_xlabel("Non-padded graph tokens (%)")
            if j == 0:
                ax.set_ylabel("Average Attention")
        ax.set_title(DATASET_LABELS.get(dataset, dataset), pad=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def _emit_llaga_top2_sink_vs_nonpad() -> None:
    discovered = _llaga_top2_sink_records_by_dataset()
    if not discovered:
        print("[skip] LLaGA top-2 sink vs non-pad %: no records JSONL found")
        return
    aggs: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
    for dataset, records_path in discovered.items():
        agg = _load_top2_sink_records(records_path)
        aggs[dataset] = agg
        if agg is None:
            print(f"[skip] LLaGA top-2 sink {dataset}: no valid samples")
            continue
        out = os.path.join(PLOTS_DIR, f"llaga_top2_sink_vs_nonpad_{dataset}.png")
        plot_llaga_top2_sink_vs_nonpad(dataset, agg, out)
        msg = (
            f"[ok]   LLaGA top-2 sink {dataset}: "
            f"n={int(agg['n_valid'])}/{int(agg['n_total'])}"
        )
        if int(agg["n_bad"]) > 0:
            msg += f" (skipped {int(agg['n_bad'])} malformed)"
        print(f"{msg} -> {out}")

    # Combined Arxiv/Cora/PubMed grid for the main paper.
    grid_datasets = ("arxiv", "cora", "pubmed")
    if any(aggs.get(d) is not None for d in grid_datasets):
        grid_out = os.path.join(PLOTS_DIR, "llaga_top2_sink_vs_nonpad_grid.png")
        plot_llaga_top2_sink_vs_nonpad_grid(aggs, grid_out, datasets=grid_datasets)
        print(f"[ok]   LLaGA top-2 sink grid {grid_datasets} -> {grid_out}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_all_cells(
    glms: Tuple[str, ...] = GLMS,
    datasets: Tuple[str, ...] = DATASETS,
    *,
    task: str = "nc",
) -> Dict[Tuple[str, str], Optional[CrossAttnCell]]:
    cells: Dict[Tuple[str, str], Optional[CrossAttnCell]] = {}
    for glm in glms:
        loader = LOADERS[glm]
        for dataset in datasets:
            cells[(glm, dataset)] = loader(dataset, task)
    return cells


def _emit_for_task(task: str) -> None:
    """Generate per-cell single panels (all GLMs) and aggregated grids
    (LLaGA + TEA-GLM only). Filenames carry a ``_{task}`` suffix so NC and
    LP outputs coexist."""
    cells = load_all_cells(task=task)

    for (glm, dataset), cell in cells.items():
        if cell is None:
            print(f"[skip] {glm} / {dataset} ({task}): no rq_arrays")
            continue
        slug = f"{glm.lower().replace('-', '')}_{dataset}"
        if cell.query_to_graph is not None:
            out = os.path.join(PLOTS_DIR, f"{slug}_query_to_graph_{task}.png")
            plot_single_query_to_graph(glm, dataset, cell, out)
            print(f"[ok]   {glm} / {dataset} ({task}): Q×K {cell.query_to_graph.shape} -> {out}")
        if cell.layer_to_graph is not None:
            out = os.path.join(PLOTS_DIR, f"{slug}_layer_to_graph_{task}.png")
            plot_single_layer_to_graph(glm, dataset, cell, out)
            print(f"[ok]   {glm} / {dataset} ({task}): L×K {cell.layer_to_graph.shape} -> {out}")

    # Main-paper grids: LLaGA + TEA-GLM only.
    qg_grid = os.path.join(PLOTS_DIR, f"rq2_query_to_graph_grid_{task}.png")
    plot_query_to_graph_grid(cells, qg_grid, glms=MAIN_PAPER_GLMS)
    print(f"[ok]   query-to-graph grid (main paper, {MAIN_PAPER_GLMS}, {task}) -> {qg_grid}")

    lg_grid = os.path.join(PLOTS_DIR, f"rq2_layer_to_graph_grid_{task}.png")
    plot_layer_to_graph_grid(cells, lg_grid, glms=MAIN_PAPER_GLMS)
    print(f"[ok]   layer-to-graph grid (main paper, {MAIN_PAPER_GLMS}, {task}) -> {lg_grid}")


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for task in TASKS:
        print(f"\n=== task: {task} ===")
        _emit_for_task(task)
    print("\n=== LLaGA top-2 sink attention vs non-pad % ===")
    _emit_llaga_top2_sink_vs_nonpad()


if __name__ == "__main__":
    main()
