import sys
sys.path.append("./")
sys.path.append("./utils")
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import matplotlib.pyplot as plt

from utils.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN
from utils.conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from utils.utils import disable_torch_init, tokenizer_graph_token, get_model_name_from_path
from torch_geometric.utils import k_hop_subgraph, degree, remove_self_loops, add_self_loops
from torch_geometric.nn import MessagePassing
import math
from utils.sinks import *
from utils.attention_probes import *
from utils.activation_probes import *
from utils.graph_remap import *


def plot_postpad_cosine_distribution(
    jsonl_path: str,
    out_path: str = "postpad_center_cosine_similarity_hist.png",
    *,
    value_key: str = "postpad_center_cosine_similarity",
    valid_key: str = "postpad_center_cosine_valid",
    bins: int = 40,
):
    """
    Reads a JSONL file where each line is a JSON object.
    Plots the distribution of postpad_center_cosine_similarity and prints the mean.

    Filtering:
      - If valid_key exists: keep only rows where it is True.
      - Keep only finite float values for value_key.
    """
    values = []
    total = 0
    kept = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)

            if valid_key in obj and not bool(obj[valid_key]):
                continue

            v = obj.get(value_key, None)
            if v is None:
                continue

            try:
                v = float(v)
            except (TypeError, ValueError):
                continue

            if not math.isfinite(v):
                continue

            values.append(v)
            kept += 1

    if kept == 0:
        raise ValueError(f"No valid '{value_key}' values found in {jsonl_path}")

    mean_val = sum(values) / kept

    plt.figure(figsize=(8, 4.5))
    plt.hist(values, bins=bins)
    plt.xlabel(value_key)
    plt.ylabel("Count")
    plt.title(f"Consine Distribution")
    plt.tight_layout()
    plt.show()
    plt.savefig(out_path, dpi=600)
    return mean_val, values

# agg = aggregate_token_scores_by_relative_position_from_jsonl(
#     records_path="analysis/pubmed_ND/first_postpad_records.jsonl",
#     n_bins=110,
# )

# plot_dataset_relative_position_attention(
#     aggregated=agg,
#     save_path="analysis/pubmed_ND/relative_position_attention_heatmap.png",
#     include_splits=True,
# )

# mean_cos, values = plot_postpad_cosine_distribution("./analysis/cora_ND/first_postpad_records.jsonl", "./analysis/cora_ND/cosine_distribution.png")
# print("mean_cos", mean_cos)
# print("values", values)

# Count number of dimensions across datasets
plot_activation_topdims_count_aggregate(
    summary_paths=[
        "analysis/cora_ND/activation_topdims.json",
        "analysis/pubmed_ND/activation_topdims.json",
        "analysis/cora_HO/activation_topdims.json",
        "analysis/pubmed_HO/activation_topdims.json"
    ],
    save_path="analysis/llaga_topdims_counts.png",
)

