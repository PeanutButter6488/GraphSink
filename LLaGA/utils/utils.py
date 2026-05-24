import torch
import os
import json
import re
from utils.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def tokenizer_graph_token(prompt, tokenizer, graph_token_index=GRAPH_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_GRAPH_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [graph_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

# Experiments: pruning; for saving pruning experiments
def add_output_suffix(path, suffix):
    root, ext = os.path.splitext(path)
    if ext:
        return f"{root}{suffix}{ext}"
    return f"{path}{suffix}"


def format_dim_list(dims):
    if not dims:
        return "none"
    return "-".join(str(int(dim)) for dim in dims)


# Experiments: sink token whole dimension pruning

def hashable_question_id(qid):
    """Make a question_id usable as a dict key. NC ids are scalar ints; LP ids
    are lists like [node_a, node_b], which aren't hashable. Convert lists/tuples
    to tuples and leave anything else (int, str) untouched."""
    if isinstance(qid, list):
        return tuple(qid)
    if isinstance(qid, tuple):
        return qid
    return qid


def load_sink_prompt_index_map(sink_records_path, mode="top2"):
    if mode not in {"top2", "all"}:
        raise ValueError(f"Unsupported sink pruning mode: {mode}")

    key = "all_sink_indices" if mode == "all" else "top2_sink_prompt_token_indices"
    sink_map = {}
    with open(sink_records_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            qid = hashable_question_id(rec["question_id"])
            sink_map[qid] = sorted({int(pos) for pos in rec.get(key, [])})
    return sink_map

def count_jsonl_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r") as f:
        return sum(1 for line in f if line.strip())


def _sanitize_plot_sample_id(sample_id):
    text = re.sub(r"\s+", "_", str(sample_id).strip())
    text = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "sample"


def _build_attention_to_sink_plot_path(dataset_name, template_name, sample_id):
    safe_sample_id = _sanitize_plot_sample_id(sample_id)
    return f"analysis/{dataset_name}_{template_name}/attention_to_sink/{safe_sample_id}.jpg"