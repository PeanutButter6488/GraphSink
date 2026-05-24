"""RQ1 plots: graph sink token-position distribution across the three GLMs.

For each (GLM, dataset) we read the per-sample ``sink_records.jsonl`` and
count how often each *graph-token local index* is detected as a sink. Each
GLM keeps these records in its own analysis directory with its own field
naming, so the loader normalizes them to a common shape:

    list[list[int]]   # one inner list of sink graph-token indices per sample

Outputs (under ``utils/plots/rq1/``):
    - ``<glm>_<dataset>_sink_distribution.png``   (one panel per cell)
    - ``rq1_sink_distribution_grid.png``          (3 GLMs x 3 datasets)
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
PLOTS_DIR = os.path.join(REPO_ROOT, "utils", "plots", "rq1")

GLMS: Tuple[str, ...] = ("InstructGLM", "LLaGA", "TEA-GLM")
# The aggregated grids that go into the main paper show only these two GLMs;
# InstructGLM's per-panel plots are still emitted for the appendix.
MAIN_PAPER_GLMS: Tuple[str, ...] = ("LLaGA", "TEA-GLM")
DATASETS: Tuple[str, ...] = ("cora", "arxiv", "pubmed")
# Tasks we plot for. InstructGLM only has NC data; LP loaders return None for it.
TASKS: Tuple[str, ...] = ("nc", "lp")
DATASET_LABELS: Dict[str, str] = {"cora": "Cora", "arxiv": "ArXiv", "pubmed": "PubMed"}

# Bar color shared across panels — distinct per GLM so the rows read at a glance.
GLM_COLORS: Dict[str, str] = {
    "InstructGLM": "#4C72B0",
    "LLaGA": "#DD8452",
    "TEA-GLM": "#55A467",
}

# LLaGA uses a fixed-size graph-token block in every sample.
LLAGA_NUM_GRAPH_TOKENS_BY_TASK = {"nc": 111, "lp": 222}
# TEA-GLM uses 5 learnable graph tokens.
TEAGLM_NUM_GRAPH_TOKENS = 5

# Layer to slice from [L, D] arrays for the sink-dim curve. -2 (second-to-last)
# matches the canonical per-GLM PNGs already on disk.
SINK_DIM_LAYER_INDEX = -2

# Per-GLM absolute floor on the sink-dim curve, matched to each GLM's own eval
# pipeline so the RQ1 grid agrees with the per-GLM PNGs on disk
# (LLaGA: eval_pretrain.py uses 15.0; TEA-GLM/InstructGLM use 5.0).
SINK_DIM_MIN_VALUE: Dict[str, float] = {
    "LLaGA": 15.0,
    "TEA-GLM": 5.0,
    "InstructGLM": 5.0,
}


@dataclass
class CellData:
    """Sink indices loaded for one (GLM, dataset) cell."""

    sink_indices_per_sample: List[List[int]]
    num_graph_tokens: int  # x-axis upper bound (exclusive)


# ---------------------------------------------------------------------------
# Loaders — one per GLM, normalizing the per-sample sink record format.
# ---------------------------------------------------------------------------

# Per-task analysis-directory suffixes. Each GLM lays its NC/LP outputs under
# slightly different names — the helpers below normalise that.
def _llaga_dir(dataset: str, task: str) -> str:
    suffix = "ND" if task == "nc" else "ND_LP"
    return os.path.join(REPO_ROOT, "LLaGA", "analysis", f"{dataset}_{suffix}")


def _instructglm_dir(dataset: str, task: str) -> Optional[str]:
    if task != "nc":
        return None  # InstructGLM has no LP data on disk yet.
    return os.path.join(REPO_ROOT, "InstructGLM", "analysis", "instructglm", dataset)


def _teaglm_dir(dataset: str, task: str) -> str:
    suffix = "" if task == "nc" else "_lp"
    return os.path.join(REPO_ROOT, "TEA-GLM", "analysis", f"{dataset}{suffix}", "global_stats")


# TEA-GLM's records and per-dim arrays carry a run-specific prefix. NC data on
# disk uses an older seed; LP data uses the new seed42 run.
TEAGLM_PREFIX_BY_TASK: Dict[str, str] = {
    "nc": "TEA-GLM_citation_meanpool_seed42",
    "lp": "TEA-GLM_citation_meanpool_seed42",
}


def _records_path_llaga(dataset: str, task: str = "nc") -> str:
    return os.path.join(_llaga_dir(dataset, task), "sink_records.jsonl")


def _records_path_instructglm(dataset: str, task: str = "nc") -> Optional[str]:
    base = _instructglm_dir(dataset, task)
    return os.path.join(base, "sink_records.jsonl") if base else None


def _records_path_teaglm(dataset: str, task: str = "nc") -> str:
    return os.path.join(
        _teaglm_dir(dataset, task),
        f"{TEAGLM_PREFIX_BY_TASK[task]}_sink_records.jsonl",
    )


def _load_llaga(dataset: str, task: str = "nc") -> Optional[CellData]:
    path = _records_path_llaga(dataset, task)
    if not os.path.exists(path):
        return None
    sinks: List[List[int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sinks.append(list(rec.get("top2_sink_token_indices", [])))
    return CellData(sinks, LLAGA_NUM_GRAPH_TOKENS_BY_TASK[task])


def _load_instructglm(dataset: str, task: str = "nc") -> Optional[CellData]:
    path = _records_path_instructglm(dataset, task)
    if path is None or not os.path.exists(path):
        return None
    sinks: List[List[int]] = []
    max_k = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sinks.append(list(rec.get("sink_token_local_indices", [])))
            max_k = max(max_k, int(rec.get("num_graph_tokens", 0)))
    if max_k == 0:
        return None
    return CellData(sinks, max_k)


def _load_teaglm(dataset: str, task: str = "nc") -> Optional[CellData]:
    path = _records_path_teaglm(dataset, task)
    if not os.path.exists(path):
        return None
    sinks: List[List[int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sinks.append(list(rec.get("top2_sink_token_indices", [])))
    return CellData(sinks, TEAGLM_NUM_GRAPH_TOKENS)


LOADERS: Dict[str, Callable[[str, str], Optional[CellData]]] = {
    "LLaGA": _load_llaga,
    "InstructGLM": _load_instructglm,
    "TEA-GLM": _load_teaglm,
}


# ---------------------------------------------------------------------------
# Sink-dimension curves — per-dim |RMSNorm| mean across graph tokens.
# Each GLM dumps its per-dim array to <analysis>/rq_arrays/. Shapes differ:
# LLaGA / TEA-GLM persist [L, D]; InstructGLM only probes one layer so it's [D].
# ---------------------------------------------------------------------------

SINK_DIM_KINDS: Tuple[str, ...] = ("all", "sink_only")


def _dim_array_path_llaga(dataset: str, kind: str, task: str = "nc") -> str:
    return os.path.join(
        _llaga_dir(dataset, task), "rq_arrays", f"mean_per_dim_{kind}.npy",
    )


def _dim_array_path_instructglm(dataset: str, kind: str, task: str = "nc") -> Optional[str]:
    base = _instructglm_dir(dataset, task)
    return os.path.join(base, "rq_arrays", f"mean_per_dim_{kind}.npy") if base else None


def _dim_array_path_teaglm(dataset: str, kind: str, task: str = "nc") -> str:
    prefix = TEAGLM_PREFIX_BY_TASK[task]
    return os.path.join(
        _teaglm_dir(dataset, task), "rq_arrays", f"{prefix}_mean_per_dim_{kind}.npy",
    )


DIM_ARRAY_PATHS: Dict[str, Callable[[str, str, str], Optional[str]]] = {
    "LLaGA": _dim_array_path_llaga,
    "InstructGLM": _dim_array_path_instructglm,
    "TEA-GLM": _dim_array_path_teaglm,
}


def _load_sink_dim_curve(
    glm: str,
    dataset: str,
    *,
    kind: str = "sink_only",
    layer_index: int = SINK_DIM_LAYER_INDEX,
    task: str = "nc",
) -> Optional[np.ndarray]:
    """Return a 1-D ``[D]`` curve, or ``None`` if the array isn't on disk.

    ``kind="all"`` averages over all graph tokens (matches LLaGA's
    ``sink_dim_mean_activation.png``); ``kind="sink_only"`` averages only over
    detected sink tokens (matches LLaGA's ``sink_only_dim_mean_activation.png``).
    """
    if kind not in SINK_DIM_KINDS:
        raise ValueError(f"kind must be one of {SINK_DIM_KINDS}, got {kind!r}")
    path = DIM_ARRAY_PATHS[glm](dataset, kind, task)
    if path is None or not os.path.exists(path):
        return None
    arr = np.load(path)
    if arr.ndim == 2:
        return arr[layer_index]
    if arr.ndim == 1:
        return arr
    raise ValueError(f"Unexpected shape {arr.shape} for {path}")


def _counts(cell: CellData) -> np.ndarray:
    counts = np.zeros(cell.num_graph_tokens, dtype=np.int64)
    for sample in cell.sink_indices_per_sample:
        for idx in sample:
            i = int(idx)
            if 0 <= i < cell.num_graph_tokens:
                counts[i] += 1
    return counts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _apply_paper_style() -> None:
    """Tune rcParams for paper-quality figures with large legible fonts."""
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


# Sink-dim tick labels are bumped up further for emphasis (vs. the 0/last ticks).
SINK_DIM_TICK_FONTSIZE = 26


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


def _plot_panel(ax: plt.Axes, cell: CellData, *, color: str, show_ylabel: bool, show_xlabel: bool) -> None:
    counts = _counts(cell)
    x = np.arange(cell.num_graph_tokens)

    ax.bar(x, counts, width=0.85, color=color, edgecolor="none")

    step = _xtick_step(cell.num_graph_tokens)
    ax.set_xticks(list(range(0, cell.num_graph_tokens, step)))
    ax.set_xlim(-0.5, cell.num_graph_tokens - 0.5)

    ax.grid(True, axis="y", alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if show_xlabel:
        ax.set_xlabel("Graph token index")
    if show_ylabel:
        ax.set_ylabel("Frequency")


def plot_single(glm: str, dataset: str, cell: CellData, save_path: str) -> str:
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    _plot_panel(ax, cell, color=GLM_COLORS[glm], show_ylabel=True, show_xlabel=True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_instructglm_distribution_grid(
    cells: Dict[Tuple[str, str], Optional[CellData]],
    save_path: str,
    *,
    datasets: Tuple[str, ...] = DATASETS,
) -> str:
    """Standalone 1xN sink-distribution figure for InstructGLM.

    Differs from ``plot_grid`` in two ways:
      - No left "InstructGLM" row label (the file name carries the GLM).
      - Fonts are scaled up so the 1-row figure stays visually balanced —
        the main-paper 2-row grid uses the same point sizes but each row
        gets twice the panel height, so the same fonts read smaller there.
    """
    _apply_paper_style()
    overrides = {
        "axes.titlesize": 32,
        "axes.labelsize": 32,
        "xtick.labelsize": 28,
        "ytick.labelsize": 28,
        "font.size": 28,
    }
    glm = "InstructGLM"
    with plt.rc_context(overrides):
        ncols = len(datasets)
        fig, axes = plt.subplots(
            1, ncols,
            figsize=(6.5 * ncols, 4.5),
            squeeze=False,
        )
        for j, dataset in enumerate(datasets):
            ax = axes[0][j]
            cell = cells.get((glm, dataset))
            if cell is None:
                ax.text(
                    0.5, 0.5, "n/a",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=28, color="#888888",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
            else:
                _plot_panel(
                    ax, cell,
                    color=GLM_COLORS[glm],
                    show_ylabel=(j == 0),
                    show_xlabel=True,
                )
            ax.set_title(DATASET_LABELS.get(dataset, dataset), pad=10)

        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
    return save_path


def plot_grid(
    cells: Dict[Tuple[str, str], Optional[CellData]],
    save_path: str,
    *,
    glms: Tuple[str, ...] = GLMS,
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
            is_bottom_row = (i == nrows - 1)
            is_left_col = (j == 0)

            if cell is None:
                ax.text(
                    0.5, 0.5, "n/a",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=24, color="#888888",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
            else:
                _plot_panel(
                    ax, cell,
                    color=GLM_COLORS[glm],
                    show_ylabel=is_left_col,
                    show_xlabel=is_bottom_row,
                )

            if i == 0:
                ax.set_title(DATASET_LABELS.get(dataset, dataset), pad=10)
            if j == 0:
                ax.annotate(
                    glm,
                    xy=(-0.22, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    ha="center", va="center",
                    fontsize=22, fontweight="bold",
                )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# Style applied to the x-tick labels that mark detected sink dims (0 and the
# last dim keep the default styling). Matches LLaGA's existing convention in
# sink_only_dim_mean_activation.png.
SINK_DIM_TICK_COLOR = "#d62728"


def _style_sink_dim_ticks(ax: plt.Axes, xticks: List[int], sink_dims: List[int]) -> None:
    """Color + italicise the tick labels for sink dims; leave others default."""
    sink_set = set(sink_dims)
    for label, t in zip(ax.get_xticklabels(), xticks):
        if int(t) in sink_set:
            label.set_color(SINK_DIM_TICK_COLOR)
            label.set_fontstyle("italic")
            label.set_fontweight("bold")
            label.set_fontsize(SINK_DIM_TICK_FONTSIZE)


def _detect_sink_dims(
    curve: np.ndarray,
    *,
    top_k: int = 3,
    rel_threshold: float = 0.2,
    min_separation: int = 500,
    min_value: Optional[float] = None,
) -> List[int]:
    """Pick the prominent peaks for x-tick labelling. Walks dims in descending
    magnitude and accepts up to ``top_k`` that (a) clear ``rel_threshold * max``,
    (b) clear the absolute ``min_value`` floor (when given — this is what each
    GLM's own eval pipeline uses to call a dim a sink), and (c) sit at least
    ``min_separation`` dims away from every dim already picked — without that
    NMS-style guard, near-adjacent peaks (e.g. dims 1415 and 1512) produce
    labels that visually overlap on the x-axis."""
    if curve.size == 0:
        return []
    threshold = float(curve.max()) * rel_threshold
    picks: List[int] = []
    for d in np.argsort(curve)[::-1]:
        d = int(d)
        v = float(curve[d])
        if v <= threshold:
            break
        if min_value is not None and v < min_value:
            continue
        if any(abs(d - p) < min_separation for p in picks):
            continue
        picks.append(d)
        if len(picks) >= top_k:
            break
    return sorted(picks)


def plot_sink_dim_grid(
    save_path: str,
    *,
    glms: Tuple[str, ...] = MAIN_PAPER_GLMS,
    datasets: Tuple[str, ...] = DATASETS,
    layer_index: int = SINK_DIM_LAYER_INDEX,
    kind: str = "sink_only",
    top_k_sink_dims: int = 3,
    task: str = "nc",
) -> str:
    """Combined sink-dim figure for the main paper: one row per GLM, one
    column per dataset. Each cell is the per-hidden-dim mean activation curve
    for that ``(glm, dataset)``."""
    _apply_paper_style()

    nrows, ncols = len(glms), len(datasets)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7.5 * ncols, 4.0 * nrows),
        squeeze=False,
    )

    for i, glm in enumerate(glms):
        color = GLM_COLORS[glm]
        for j, dataset in enumerate(datasets):
            ax = axes[i][j]
            curve = _load_sink_dim_curve(glm, dataset, kind=kind, layer_index=layer_index, task=task)

            is_bottom_row = (i == nrows - 1)
            is_left_col = (j == 0)

            if curve is None:
                ax.text(
                    0.5, 0.5, "n/a",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=24, color="#888888",
                )
                ax.set_xticks([]); ax.set_yticks([])
            else:
                D = curve.shape[0]
                ax.plot(np.arange(D), curve, color=color, linewidth=1.2)
                ax.set_xlim(0, D - 1)

                sink_dims = _detect_sink_dims(
                    curve,
                    top_k=top_k_sink_dims,
                    min_value=SINK_DIM_MIN_VALUE.get(glm),
                )
                # Drop boundary ticks (0, D-1) if a sink dim is too close —
                # otherwise their labels collide with the much larger sink-dim
                # labels.
                boundary_buffer = 800
                boundaries = {b for b in (0, D - 1)
                              if all(abs(b - s) >= boundary_buffer for s in sink_dims)}
                xticks = sorted({*boundaries, *sink_dims})
                ax.set_xticks(xticks)
                ax.set_xticklabels([str(t) for t in xticks])
                _style_sink_dim_ticks(ax, xticks, sink_dims)

                ax.grid(True, axis="y", alpha=0.3, linewidth=0.8)
                ax.set_axisbelow(True)
                if is_bottom_row:
                    ax.set_xlabel("Hidden dimension")
                if is_left_col:
                    ax.set_ylabel("Activation", labelpad=8)

            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

            if i == 0:
                ax.set_title(DATASET_LABELS.get(dataset, dataset), pad=10)
            if is_left_col:
                ax.annotate(
                    glm,
                    xy=(-0.30, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    ha="center", va="center",
                    fontsize=26, fontweight="bold",
                )

    fig.tight_layout(rect=[0.04, 0, 1, 1])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_sink_dim_curves_for_glm(
    glm: str,
    save_path: str,
    *,
    datasets: Tuple[str, ...] = DATASETS,
    layer_index: int = SINK_DIM_LAYER_INDEX,
    kind: str = "sink_only",
    top_k_sink_dims: int = 3,
    task: str = "nc",
) -> str:
    """One figure for ``glm``: 1 row x 3 cols, one panel per dataset showing
    the per-hidden-dim mean activation curve. ``kind`` selects which token
    subset the per-dim mean is taken over: ``"all"`` (every graph token) or
    ``"sink_only"`` (only detected sink tokens).
    """
    _apply_paper_style()

    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(7.5 * len(datasets), 5.0),
        squeeze=False,
    )

    color = GLM_COLORS[glm]
    for j, dataset in enumerate(datasets):
        ax = axes[0][j]
        curve = _load_sink_dim_curve(glm, dataset, kind=kind, layer_index=layer_index, task=task)
        if curve is None:
            ax.text(
                0.5, 0.5, "n/a",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=24, color="#888888",
            )
            ax.set_xticks([]); ax.set_yticks([])
        else:
            D = curve.shape[0]
            x = np.arange(D)
            ax.plot(x, curve, color=color, linewidth=1.2)
            ax.set_xlim(0, D - 1)

            sink_dims = _detect_sink_dims(
                curve,
                top_k=top_k_sink_dims,
                min_value=SINK_DIM_MIN_VALUE.get(glm),
            )
            boundary_buffer = 800
            boundaries = {b for b in (0, D - 1)
                          if all(abs(b - s) >= boundary_buffer for s in sink_dims)}
            xticks = sorted({*boundaries, *sink_dims})
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(t) for t in xticks])
            _style_sink_dim_ticks(ax, xticks, sink_dims)

            ax.grid(True, axis="y", alpha=0.3, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_xlabel("Hidden dimension")
            if j == 0:
                ax.set_ylabel("Activation")

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_title(DATASET_LABELS.get(dataset, dataset), pad=10)

    fig.tight_layout()
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
) -> Dict[Tuple[str, str], Optional[CellData]]:
    cells: Dict[Tuple[str, str], Optional[CellData]] = {}
    for glm in glms:
        loader = LOADERS[glm]
        for dataset in datasets:
            cells[(glm, dataset)] = loader(dataset, task)
    return cells


def _emit_for_task(task: str) -> None:
    """Generate the full set of sink-distribution + sink-dim figures for one task.
    File names carry a ``_{task}`` suffix so NC and LP outputs coexist."""
    cells = load_all_cells(task=task)

    for (glm, dataset), cell in cells.items():
        if cell is None:
            print(f"[skip] {glm} / {dataset} ({task}): no records found")
            continue
        slug = glm.lower().replace('-', '')
        out = os.path.join(
            PLOTS_DIR, f"{slug}_{dataset}_sink_distribution_{task}.png"
        )
        plot_single(glm, dataset, cell, out)
        print(f"[ok]   {glm} / {dataset} ({task}): {len(cell.sink_indices_per_sample)} samples -> {out}")

    # Main-paper distribution grid: LLaGA + TEA-GLM only (InstructGLM goes in
    # the appendix via the per-GLM single panels emitted above).
    grid_path = os.path.join(PLOTS_DIR, f"rq1_sink_distribution_grid_{task}.png")
    plot_grid(cells, grid_path, glms=MAIN_PAPER_GLMS)
    print(f"[ok]   grid (main paper, {MAIN_PAPER_GLMS}, {task}) -> {grid_path}")

    # Appendix-only: 1xN sink-distribution grid for InstructGLM. Kept as a
    # separate file so the main-paper grid above stays untouched.
    if any(cells.get(("InstructGLM", d)) is not None for d in DATASETS):
        ig_grid_path = os.path.join(
            PLOTS_DIR, f"rq1_sink_distribution_grid_instructglm_{task}.png"
        )
        plot_instructglm_distribution_grid(cells, ig_grid_path)
        print(f"[ok]   grid (InstructGLM, {task}) -> {ig_grid_path}")

    for glm in GLMS:
        slug = glm.lower().replace('-', '')
        for kind in SINK_DIM_KINDS:
            # Skip per-GLM sink-dim plots whose array isn't on disk for this task
            # (e.g. InstructGLM has no LP).
            if all(_load_sink_dim_curve(glm, d, kind=kind, task=task) is None for d in DATASETS):
                print(f"[skip] sink-dim curves ({glm}, {kind}, {task}): no arrays found")
                continue
            out = os.path.join(PLOTS_DIR, f"rq1_sink_dim_curves_{slug}_{kind}_{task}.png")
            plot_sink_dim_curves_for_glm(glm, out, kind=kind, task=task)
            print(f"[ok]   sink-dim curves ({glm}, {kind}, {task}) -> {out}")

    # Main-paper combined sink-dim grid (LLaGA + TEA-GLM in one figure).
    for kind in SINK_DIM_KINDS:
        out = os.path.join(PLOTS_DIR, f"rq1_sink_dim_grid_{kind}_{task}.png")
        plot_sink_dim_grid(out, kind=kind, task=task)
        print(f"[ok]   sink-dim grid (main paper, {kind}, {task}) -> {out}")


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for task in TASKS:
        print(f"\n=== task: {task} ===")
        _emit_for_task(task)


if __name__ == "__main__":
    main()
