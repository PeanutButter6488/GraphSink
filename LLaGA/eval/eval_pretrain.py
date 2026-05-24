import sys
sys.path.append("./")
sys.path.append("./utils")
import argparse
import random
import torch
import numpy as np
import os
import json
from tqdm import tqdm
import shortuuid
import traceback
import re

from utils.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN
from utils.conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from utils.utils import disable_torch_init, tokenizer_graph_token, get_model_name_from_path
from torch_geometric.utils import k_hop_subgraph, degree, remove_self_loops, add_self_loops
from torch_geometric.nn import MessagePassing
import math
from utils.attention_probes import *
from utils.activation_probes import *
from utils.dimension_pruning import *
from utils.graph_remap import *
from utils.utils import *
from utils.attention_redistribution import (
    install_redistribution,
    uninstall_redistribution,
    set_redistribution_state,
    clear_redistribution_state,
)
from utils.contrastive_steering import (
    apply_contrastive_steering,
    remove_contrastive_steering,
    resolve_layer_indices,
    select_target_positions,
)
from utils.logit_lens import (
    compute_logit_lens,
    aggregate_logit_lens,
    plot_logit_lens_heatmap,
)

# Sample filter (NC): keep samples whose detected sink graph-token indices
# are *exactly* LOGIT_LENS_SINK_POSITIONS (i.e. sinks live at 4 and 5, and
# only there). The heatmap displays LOGIT_LENS_DISPLAY_POSITIONS so the
# sink rows (4, 5) are shown alongside non-sink reference rows (0..3).
LOGIT_LENS_SINK_POSITIONS = (4, 5)
LOGIT_LENS_DISPLAY_POSITIONS = (0, 1, 2, 3, 4, 5)
 
SMALL_DATASETS=["pubmed", "cora"]
BUILTIN_DATA_DIRS = {
    "arxiv": "dataset/arxiv",
    "products": "dataset/ogbn-products",
    "pubmed": "dataset/pubmed",
    "cora": "dataset/cora",
}


class MP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    # Added: propagate_type: (x: torch.Tensor, norm: torch.Tensor)
    def forward(self, x, edge_index, norm):
        return self.propagate(edge_index, x=x, norm=norm)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def _sanitize_plot_sample_id(sample_id):
    text = re.sub(r"\s+", "_", str(sample_id).strip())
    text = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "sample"


def _build_attention_to_sink_plot_path(dataset_name, template_name, sample_id):
    safe_sample_id = _sanitize_plot_sample_id(sample_id)
    return f"analysis/{dataset_name}_{template_name}/attention_to_sink/{safe_sample_id}.jpg"


def _build_dataset_remap_plot_path(dataset_name, template_name, sample_id):
    safe_sample_id = _sanitize_plot_sample_id(sample_id)
    return f"analysis/{dataset_name}_{template_name}/remap_plot/{safe_sample_id}.png"


def load_pretrain_embedding_graph(data_dir, pretrained_embedding_type):
    if pretrained_embedding_type == "simteg":
        simteg_sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))
        simteg_roberta = torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))
        simteg_e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))
        pretrained_emb = torch.concat([simteg_sbert, simteg_roberta, simteg_e5], dim=-1)
    else:
        pretrained_emb = torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))
    return pretrained_emb

def load_pretrain_embedding_hop(data_dir, pretrained_embedding_type, hop, mask):
    if pretrained_embedding_type == "simteg":
        simteg_sbert=[torch.load(os.path.join(data_dir, f"simteg_sbert_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        simteg_roberta = [torch.load(os.path.join(data_dir, f"simteg_roberta_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        simteg_e5 = [torch.load(os.path.join(data_dir, f"simteg_e5_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        pretrained_embs = [torch.cat([simteg_sbert[i], simteg_roberta[i], simteg_e5[i]], dim=-1) for i in range(hop + 1)]
    else:
        pretrained_embs = [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))[mask]]+  [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x.pt"))[mask] for i in range(1, hop+1)]

    return pretrained_embs

def load_pretrain_embedding_hop_lp(data_dir, pretrained_embedding_type, hop):
    mask = torch.load(os.path.join(data_dir, f"no_test_link_mask.pt"))
    if pretrained_embedding_type == "simteg":
        simteg_sbert=[torch.load(os.path.join(data_dir, f"simteg_sbert_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x_notestlink.pt")) for i in range(1, hop + 1)]
        simteg_roberta = [torch.load(os.path.join(data_dir, f"simteg_roberta_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x_notestlink.pt")) for i in range(1, hop + 1)]
        simteg_e5 = [torch.load(os.path.join(data_dir, f"simteg_e5_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x_notestlink.pt")) for i in range(1, hop + 1)]
        pretrained_embs = [torch.cat([simteg_sbert[i], simteg_roberta[i], simteg_e5[i]], dim=-1) for i in range(hop + 1)]
    else:
        pretrained_embs = [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))[mask]]+  [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x_notestlink.pt")) for i in range(1, hop+1)]

    return pretrained_embs, mask


def resolve_eval_data_dir(args):
    if args.data_dir is not None:
        return os.path.expanduser(args.data_dir)

    if args.dataset in BUILTIN_DATA_DIRS:
        return BUILTIN_DATA_DIRS[args.dataset]

    raise ValueError(
        f"Unknown dataset '{args.dataset}'. Provide --data_dir for external OOD datasets."
    )


def resolve_eval_paths(args, data_dir):
    if args.test_path is not None:
        prompt_file = os.path.expanduser(args.test_path)
    elif args.task in ["nc", "nd", "nda", "nctext"]:
        if args.template == "HO":
            prompt_file = os.path.join(data_dir, "sampled_2_10_test.jsonl")
        else:
            prompt_file = os.path.join(
                data_dir,
                f"sampled_{args.use_hop}_{args.sample_neighbor_size}_test.jsonl",
            )
    elif args.task in ["lp"]:
        if args.template == "HO":
            prompt_file = os.path.join(data_dir, "edge_sampled_2_10_only_test.jsonl")
        else:
            prompt_file = os.path.join(
                data_dir,
                f"edge_sampled_{args.use_hop}_{args.sample_neighbor_size}_only_test.jsonl",
            )
    else:
        raise ValueError

    data_path = os.path.join(data_dir, "processed_data.pt")
    return prompt_file, data_path


def resolve_nc_prompt(args, line):
    conversations = line.get("conversations")
    if isinstance(conversations, list) and len(conversations) > 0:
        first_turn = conversations[0]
        if isinstance(first_turn, dict) and first_turn.get("value"):
            return first_turn["value"]

    if args.dataset == "products":
        return (
            f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent "
            "products sold in Amazon, and edges between products indicate they are "
            "purchased together. We need to classify the center node into 47 classes: "
            "Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, "
            "Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, "
            "Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, "
            "Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, "
            "Office Products, Industrial & Scientific, Musical Instruments, Tools & Home "
            "Improvement, Magazine Subscriptions, Baby Products, label 25, Appliances, "
            "Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon "
            "Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & "
            "Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & "
            "Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, "
            "Buy a Kindle, Furniture & D&#233;cor, #508510, please tell me which class the "
            "center node belongs to?"
        )

    raise ValueError(
        "NC evaluation requires conversations[0]['value'] for external datasets."
    )

def eval_model(args):
    # Model
    disable_torch_init()

    sink_dims_list = [int(d.strip()) for d in args.sink_dims.split(",") if d.strip()]
    sink_threshold = float(args.sink_threshold)

    # Tag analysis output dir with _LP for link-prediction runs so they don't
    # overwrite the NC analysis artifacts (which use the same {dataset}_{template}
    # directory). NC behavior is unchanged.
    analysis_dir = f"analysis/{args.dataset}_{args.template}" + (
        "_LP" if args.task == "lp" else ""
    )

    # Histogram bin count for plot_sink_token_index_histogram. Must match the
    # actual graph-key K so sink indices >= 111 (ND+LP) aren't dropped silently.
    if args.template == "HO":
        _tokens_per_subgraph = args.use_hop + 1
    elif args.template == "ND":
        _tokens_per_subgraph = 1 + sum(
            args.sample_neighbor_size ** i for i in range(1, args.use_hop + 1)
        )
    else:
        _tokens_per_subgraph = 111  # fallback to NC/ND-default
    _num_subgraphs = 2 if args.task == "lp" else 1
    num_graph_tokens = _num_subgraphs * _tokens_per_subgraph

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loaded from {model_path}. Model Base: {args.model_base}")
    tokenizer, model, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                          cache_dir=args.cache_dir)
    model = model.to(torch.float16).cuda()

    if args.redistribute:
        install_redistribution(model)
    data_dir = resolve_eval_data_dir(args)
    prompt_file, data_path = resolve_eval_paths(args, data_dir)

    data = torch.load(data_path, weights_only=False)
    print(f"Using data_dir={data_dir}")
    print(f"Load from {prompt_file}\n")
    lines = open(prompt_file, "r").readlines()

    if args.start >= 0:
        if args.end < 0:
            args.end = len(lines)
        lines = lines[args.start:args.end]
    elif args.end > 0:
        lines = lines[:args.end]

    if args.max_samples is not None and args.max_samples > 0 and len(lines) > args.max_samples:
        print(f"Limiting evaluation to {args.max_samples} samples (out of {len(lines)}).")
        lines = lines[:args.max_samples]

    base_answers_file = os.path.expanduser(args.answers_file)
    answers_file = base_answers_file
    rename_answers_file_to = None

    if args.pruning:
        if args.pruning_nonsink:
            prune_suffix = f"_prune_nonsinktoken_{args.num_prune}_run{args.run_idx}"
        else:
            prune_suffix = f"_prune_sinktoken_{args.pruning_mode}_run{args.run_idx}"
        answers_file = add_output_suffix(base_answers_file, prune_suffix)
        args.answers_file = answers_file
        print(f"Token pruning enabled. Saving answers to {answers_file}")

    if args.reposition_mode != "none":
        assert not args.pruning, "--reposition_mode is mutually exclusive with --pruning."
        if args.reposition_mode == "front_top2":
            reposition_suffix = "_reposition_front_top2"
        elif args.reposition_mode == "front_all":
            reposition_suffix = "_reposition_front_all"
        else:
            reposition_suffix = f"_reposition_swap_sink_nonsink_k{args.num_swap}_run{args.run_idx}"
        answers_file = add_output_suffix(base_answers_file, reposition_suffix)
        args.answers_file = answers_file
        print(f"Reposition enabled ({args.reposition_mode}). Saving answers to {answers_file}")

    if args.steering_mode != "none":
        assert not args.pruning, "--steering_mode is mutually exclusive with --pruning."
        assert args.reposition_mode == "none", "--steering_mode is mutually exclusive with --reposition_mode."
        layers_tag = re.sub(r"[^A-Za-z0-9_-]", "_", args.steering_layers.strip())
        source_tag = "ps" if args.steering_source == "per_sample" else "gl"
        steering_suffix = (
            f"_steer_{args.steering_mode}_{source_tag}_L{layers_tag}"
            f"_s{args.steering_strength:g}_t{args.steering_target}"
            + ("_abs" if args.steering_use_abs else "")
        )
        answers_file = add_output_suffix(base_answers_file, steering_suffix)
        args.answers_file = answers_file
        print(f"Steering enabled ({args.steering_mode}, layers={args.steering_layers}, "
              f"s={args.steering_strength}, target={args.steering_target}). "
              f"Saving answers to {answers_file}")

    redistribute_layer_filter = None
    if args.redistribute:
        assert not args.pruning, "--redistribute is mutually exclusive with --pruning."
        assert args.reposition_mode == "none", "--redistribute is mutually exclusive with --reposition_mode."
        layers_arg = args.redistribute_layers.strip()
        if layers_arg.lower() != "all":
            if "-" in layers_arg:
                lo, hi = layers_arg.split("-")
                redistribute_layer_filter = list(range(int(lo), int(hi) + 1))
            else:
                redistribute_layer_filter = [int(x) for x in layers_arg.split(",") if x.strip()]
        if args.redistribute_direction == "src_to_sinks":
            redistribute_suffix = (
                f"_redistribute_src_to_sinks_pct{args.redistribute_fraction}"
                f"_src{args.redistribute_source_idx}_run{args.run_idx}"
            )
        else:
            redistribute_suffix = (
                f"_redistribute_{args.redistribute_direction}"
                f"_pct{args.redistribute_fraction}_run{args.run_idx}"
            )
        answers_file = add_output_suffix(base_answers_file, redistribute_suffix)
        args.answers_file = answers_file
        print(
            f"Redistribute enabled (direction={args.redistribute_direction}, "
            f"fraction={args.redistribute_fraction}, "
            f"source_idx={args.redistribute_source_idx}, layers={args.redistribute_layers}). "
            f"Saving answers to {answers_file}"
        )
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "w")
    # Save sink tokens
    sink_records_path = f"{analysis_dir}/sink_records.jsonl"
    os.makedirs(os.path.dirname(sink_records_path), exist_ok=True)
    sink_reoccur_path = f"{analysis_dir}/sink_reoccur.jsonl"
    # Numerical artifacts for offline replotting (utils/plot_rq*.py).
    rq_arrays_dir = f"{analysis_dir}/rq_arrays"
    os.makedirs(rq_arrays_dir, exist_ok=True)

    reposition_tag = None
    reposition_sink_records_path = None
    reposition_sink_file = None
    all_reposition_sink_token_indices = []
    if args.reposition_mode == "front_top2":
        reposition_tag = "front_top2"
    elif args.reposition_mode == "front_all":
        reposition_tag = "front_all"
    elif args.reposition_mode == "swap_sink_nonsink":
        reposition_tag = f"swap_sink_nonsink_k{args.num_swap}_run{args.run_idx}"
    if reposition_tag is not None:
        reposition_sink_records_path = (
            f"{analysis_dir}/reposition_{reposition_tag}_sink_records.jsonl"
        )
        os.makedirs(os.path.dirname(reposition_sink_records_path), exist_ok=True)
        reposition_sink_file = open(reposition_sink_records_path, "w")

    top2_sink_attention_nonpad_records_path = f"{analysis_dir}/top2_sink_attention_nonpad_records.jsonl"
    os.makedirs(os.path.dirname(top2_sink_attention_nonpad_records_path), exist_ok=True)
    open(top2_sink_attention_nonpad_records_path, 'w').close()

    highest_attention_graph_token_records_path = None
    if args.attention_probe and args.template == "ND":
        highest_attention_graph_token_records_path = (
            f"{analysis_dir}/highest_attention_graph_token_records.jsonl"
        )
        os.makedirs(os.path.dirname(highest_attention_graph_token_records_path), exist_ok=True)
        open(highest_attention_graph_token_records_path, "w").close()

    questions = [json.loads(q) for q in lines]
    num_questions = len(questions)

    existing_sink_record_lines = count_jsonl_lines(sink_records_path)
    use_existing_sink_records = (
        os.path.exists(sink_records_path)
        and existing_sink_record_lines == num_questions
        and args.use_existing_sink_records
    )

    sink_file = None
    if use_existing_sink_records:
        print(
            f"Found existing sink records at {sink_records_path} "
            f"with {existing_sink_record_lines} lines matching test set length {num_questions}. "
            f"Will reuse them and skip sink detection."
        )
    else:
        if os.path.exists(sink_records_path):
            print(
                f"Sink records at {sink_records_path} have {existing_sink_record_lines} lines, "
                f"but test set length is {num_questions}. Will recompute and overwrite."
            )
        else:
            print(f"No sink records found at {sink_records_path}. Will detect sinks and save them.")
        sink_file = open(sink_records_path, "w")

    index = None
    if args.template == "ND":
        pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
        structure_emb = torch.load(
            f"dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt")

    elif args.template == "HO":
        n = data.num_nodes
        if args.dataset in SMALL_DATASETS and args.task == "lp":
            pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
        elif args.task == "lp":
            # for small dataset, we remove test link during testing
            # for large dataset, remove test link and compute embedding may be more memory- and time-consuming , we precompute the embedding
            pretrained_emb, mask = load_pretrain_embedding_hop_lp(data_dir, args.pretrained_embedding_type,args.use_hop)
            index = torch.full([n], fill_value=n + 1, dtype=torch.long)
            test_index = torch.arange(mask.sum())
            index[mask] = test_index
        else:
            mask = torch.full([n], fill_value=False, dtype=torch.bool)
            for q in questions:
                idx = q["id"]
                if "lp" in  args.task:
                    assert len(idx) == 2
                    mask[idx[0]] = True
                    mask[idx[1]] = True
                elif args.task  in ["nc", "nd", "nctext"]:
                    assert isinstance(idx, int)
                    mask[idx] = True
            pretrained_emb = load_pretrain_embedding_hop(data_dir, args.pretrained_embedding_type, args.use_hop, mask)
            index = torch.full([n], fill_value=n + 1, dtype=torch.long)
            test_index = torch.arange(mask.sum())
            index[mask] = test_index
        structure_emb = None
    else:
        raise ValueError

    # Initialization
    sink_activation_agg_state = None
    pregraph_activation_agg_state = None
    activation_topdims_state = None
    sink_only_activation_topdims_state = None
    all_sample_layer_query_to_graph = []
    all_sample_layer_to_graph_lk = []
    all_sample_generated_key_is_pad = []
    all_sink_token_indices = []
    all_reoccur_sink_token_indices = []

    # Logit-lens buffers (NC only; we collect per-sample top-1 ids/probs
    # and aggregate to a modal-token + mean-prob heatmap after the loop).
    logit_lens_records_ids = []
    logit_lens_records_probs = []
    logit_lens_total_seen = 0
    logit_lens_filter_set = set(LOGIT_LENS_SINK_POSITIONS)
    sink_pad_hits = [0, 0]
    sink_pad_totals = [0, 0]
    num_attention_probe_plots = 0
    num_attention_probe_skipped = 0
    num_remap_plots = 0
    num_remap_plot_skipped = 0

    # Pre-load sink map
    sink_prompt_index_map = None
    top2_sink_prompt_index_map = None
    all_sink_prompt_index_map = None
    sink_map_mode = "all" if args.pruning_nonsink else args.pruning_mode
    do_sink_reoccur = (
        args.sink_reoccur
        and args.pruning
        and (not args.pruning_nonsink)
        and args.pruning_mode == "all"
    )
    if args.sink_reoccur and not do_sink_reoccur:
        print("sink_reoccur only runs when pruning all saved sink tokens.")
    sink_reoccur_file = open(sink_reoccur_path, "w") if do_sink_reoccur else None
    if use_existing_sink_records:
        sink_prompt_index_map = load_sink_prompt_index_map(
            sink_records_path,
            mode=sink_map_mode,
        )
        top2_sink_prompt_index_map = load_sink_prompt_index_map(
            sink_records_path,
            mode="top2",
        )
        all_sink_prompt_index_map = load_sink_prompt_index_map(
            sink_records_path,
            mode="all",
        )
        print(f"Loaded sink records from {sink_records_path}")
    elif args.pruning and args.sink_records_path is not None and os.path.exists(args.sink_records_path):
        sink_prompt_index_map = load_sink_prompt_index_map(
            args.sink_records_path,
            mode=sink_map_mode,
        )
        print(f"Loaded sink records from {args.sink_records_path}")

    if args.reposition_mode != "none":
        assert use_existing_sink_records, (
            f"--reposition_mode={args.reposition_mode} requires precomputed sink records at "
            f"{sink_records_path} with {num_questions} lines."
        )
    if args.redistribute:
        assert use_existing_sink_records, (
            f"--redistribute requires precomputed sink records at "
            f"{sink_records_path} with {num_questions} lines."
        )

    steering_vectors = None
    steering_layer_indices = None
    if args.steering_mode != "none":
        num_model_layers = len(model.model.layers)
        steering_layer_indices = resolve_layer_indices(args.steering_layers, num_model_layers)
        if args.steering_source == "per_sample":
            assert use_existing_sink_records, (
                "--steering_source=per_sample requires --use_existing_sink_records "
                "(per-sample sink positions are read from the saved sink_records.jsonl)."
            )
            print(f"[steering] per-sample, layers={steering_layer_indices} "
                  f"(s derived online from each sample's sink-token hidden states)")
        else:
            if args.steering_vector_path is None:
                fname = "mean_per_dim_sink_only.npy" if args.steering_use_abs else "mean_per_dim_sink_only_signed.npy"
                steering_vector_path = f"{analysis_dir}/rq_arrays/{fname}"
            else:
                steering_vector_path = args.steering_vector_path
            assert os.path.exists(steering_vector_path), f"Steering vector not found: {steering_vector_path}"
            arr = np.load(steering_vector_path)
            assert arr.ndim == 2, f"Expected [num_layers, D], got {arr.shape}"
            max_idx = max(steering_layer_indices)
            assert max_idx < arr.shape[0], (
                f"--steering_layers references layer {max_idx} but vector file has only "
                f"{arr.shape[0]} layers in {steering_vector_path}"
            )
            steering_vectors = torch.from_numpy(arr[steering_layer_indices]).to(torch.float32)
            norms = torch.linalg.vector_norm(steering_vectors, dim=1).tolist()
            print(f"[steering] global, {steering_vector_path} layers={steering_layer_indices} "
                  f"||s||={[round(n, 3) for n in norms]}")

    # Main Loop
    for line in tqdm(questions):
        top2_sink_token_indices = []
        top2_sink_prompt_token_indices = []
        top2_sink_scores = []
        top2_sink_is_pad = []
        all_sinks = []
        reoccur_sink_token_indices = []
        reoccur_sink_prompt_token_indices = []
        reoccur_sink_scores = []
        idx = line["id"]
        if args.task in ["nd", "nda"]:
            qs=f"Please briefly describe the center node of {DEFAULT_GRAPH_TOKEN}."
        elif args.task == "nc":
            qs = resolve_nc_prompt(args, line)
        elif args.task == "nctext":
            text = data.raw_texts[line['id']]
            text = text[:2000]
            if args.dataset == "arxiv":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 40 classes: cs.NA(Numerical Analysis), cs.MM(Multimedia), cs.LO(Logic in Computer Science), cs.CY(Computers and Society), cs.CR(Cryptography and Security), cs.DC(Distributed, Parallel, and Cluster Computing), cs.HC(Human-Computer Interaction), cs.CE(Computational Engineering, Finance, and Science), cs.NI(Networking and Internet Architecture), cs.CC(Computational Complexity), cs.AI(Artificial Intelligence), cs.MA(Multiagent Systems), cs.GL(General Literature), cs.NE(Neural and Evolutionary Computing), cs.SC(Symbolic Computation), cs.AR(Hardware Architecture), cs.CV(Computer Vision and Pattern Recognition), cs.GR(Graphics), cs.ET(Emerging Technologies), cs.SY(Systems and Control), cs.CG(Computational Geometry), cs.OH(Other Computer Science), cs.PL(Programming Languages), cs.SE(Software Engineering), cs.LG(Machine Learning), cs.SD(Sound), cs.SI(Social and Information Networks), cs.RO(Robotics), cs.IT(Information Theory), cs.PF(Performance), cs.CL(Computational Complexity), cs.IR(Information Retrieval), cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory), cs.DS(Data Structures and Algorithms), cs.OS(Operating Systems), cs.GT(Computer Science and Game Theory), cs.DB(Databases), cs.DL(Digital Libraries), cs.DM(Discrete Mathematics), please tell me which class the center node belongs to? Direct tell me the class name."
            elif args.dataset == "products":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent products sold in Amazon, and edges between products indicate they are purchased together, the node feature of center node is {text}. We need to classify the center node into 47 classes: Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, Baby Products, label 25, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & D&#233;cor, #508510, please tell me which class the center node belongs to? Direct tell me the class name."
            elif args.dataset == "pubmed":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers about Diabetes and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 3 classes: Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2, please tell me which class the center node belongs to? Direct tell me the class name."
            elif args.dataset == "cora":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory, please tell me which class the center node belongs to? Direct tell me the class name."
            else:
                raise ValueError
        elif args.task == "lp":
            qs=f"Given two node-centered subgraphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, we need to predict whether these two nodes connect with each other. Please tell me whether two center nodes in the subgraphs should connect to each other."
        else:
            print(f"NOT SUPPORT {args.task}!!!")
            raise ValueError
        cur_prompt = qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_graph_token(prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()  # inserts graph token (-200) into token sequence, len = 120
        if not isinstance(line['graph'][0], list):
            line['graph'] = [line['graph']]
        if args.template == "ND":
            graph = torch.LongTensor(line['graph'])
            mask = graph != DEFAULT_GRAPH_PAD_ID
            masked_graph_emb = pretrained_emb[graph[mask]]
            s, n, d = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]   # batch size, number of graph tokens (including paddings, i.e., -500), hidden dimension
            graph_emb = torch.zeros((s, n, d))
            graph_emb[mask] = masked_graph_emb
            if structure_emb is not None:
                graph_emb = torch.cat([graph_emb, structure_emb.unsqueeze(0).expand(s, -1, -1)], dim=-1)   # the reason why the two cosine similarities for the same node at different positions have different embeddings
        elif args.template == "HO":
            # for small dataset, we remove test link during testing
            # for large dataset, remove test link and compute embedding may be more memory- and time-consuming , we precompute the embedding

            if args.dataset in SMALL_DATASETS and args.task == "lp":
                mp = MP()
                center_nodes = []
                for g in range(len(line['graph'])):
                    center_id = line['graph'][g][0]
                    line['graph'][g] = [center_id] * (args.use_hop + 1)
                    center_nodes.append(center_id)
                graph = torch.LongTensor(line['graph'])
                center_id = graph[:, 0]
                graph_embs = [pretrained_emb[center_id].cuda()]
                subset, edge_index, mapping, edge_mask = k_hop_subgraph(center_nodes, args.use_hop, data.edge_index,
                                                                        relabel_nodes=True)
                local_edge_mask = ((edge_index[0] == mapping[0]) & (edge_index[1] == mapping[1])) | (
                            (edge_index[0] == mapping[1]) & (edge_index[1] == mapping[0]))
                edge_index = edge_index[:, ~local_edge_mask]
                local_x = pretrained_emb[subset].cuda()
                n = subset.shape[0]
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index)
                edge_index = edge_index.cuda()
                row, col = edge_index
                deg = degree(col, n, dtype=pretrained_emb.dtype)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                # local_x = pretrained_emb
                # for _ in range(args.use_hop):
                #     local_x = mp.propagate(edge_index, x=local_x, norm=norm)
                #     graph_embs.append(local_x[mapping])
                # graph_emb = torch.stack(graph_embs, dim=1)
                for _ in range(args.use_hop):
                    local_x = mp(local_x, edge_index, norm)
                    graph_embs.append(local_x[mapping])
                graph_emb = torch.stack(graph_embs, dim=1)
            else:

                for g in range(len(line['graph'])):
                    center_id = line['graph'][g][0]
                    line['graph'][g] = [center_id]*(args.use_hop+1)
                graph = torch.LongTensor(line['graph'])
                center_id = graph[:, 0]
                graph_emb = torch.stack([emb[index[center_id]] for emb in pretrained_emb], dim=1)
        else:
            raise ValueError


        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        # LP question_ids are lists ([node_a, node_b]); make them hashable so
        # they match the keys produced by load_sink_prompt_index_map.
        idx_key = hashable_question_id(idx)

        saved_sink_prompt_positions = []
        if sink_prompt_index_map is not None:
            saved_sink_prompt_positions = sink_prompt_index_map.get(idx_key, [])

        saved_top2_sink_prompt_positions = []
        if top2_sink_prompt_index_map is not None:
            saved_top2_sink_prompt_positions = top2_sink_prompt_index_map.get(idx_key, [])

        saved_all_sink_prompt_positions = []
        if all_sink_prompt_index_map is not None:
            saved_all_sink_prompt_positions = all_sink_prompt_index_map.get(idx_key, [])

        nonpad_graph_prompt_positions = []
        if args.pruning and args.pruning_nonsink:
            expanded_prompt_len = int(input_ids.shape[1] - graph.shape[0] + graph.numel())
            if getattr(model.config, "mm_use_graph_special_token", False):
                expanded_prompt_len += int(graph.shape[0] * (getattr(model.config, "use_hop", 0) + 2))
            graph_idx_info = get_expanded_graph_key_query_indices(
                input_ids_1d=input_ids[0],
                attention_mask_1d=torch.ones_like(input_ids[0]),
                graphs=graph.cuda(),
                prompt_len=expanded_prompt_len,
                keep_pad_tokens=False,
                mm_use_graph_special_token=getattr(model.config, "mm_use_graph_special_token", False),
                use_hop=getattr(model.config, "use_hop", None),
                sample_neighbor_size=getattr(model.config, "sample_neighbor_size", None),
            )
            if graph_idx_info is not None:
                nonpad_graph_prompt_positions = graph_idx_info[0].detach().cpu().tolist()

        selected_nonsink_token_indices = []
        prune_token_positions = []
        graph_key_idx_all = None
        graph_key_is_pad_all = None

        if args.pruning and args.pruning_nonsink and len(saved_sink_prompt_positions) > 0:
            selected_nonsink_token_indices = sample_nonsink_token_positions(
                graph_prompt_positions=nonpad_graph_prompt_positions,
                sink_prompt_positions=saved_sink_prompt_positions,
                num_to_prune=args.num_prune,
                seed=args.seed,
            )
            prune_token_positions = selected_nonsink_token_indices
        elif args.pruning and len(saved_sink_prompt_positions) > 0:
            prune_token_positions = saved_sink_prompt_positions

        if do_sink_reoccur and len(prune_token_positions) > 0:
            expanded_prompt_len = int(input_ids.shape[1] - graph.shape[0] + graph.numel())
            if getattr(model.config, "mm_use_graph_special_token", False):
                expanded_prompt_len += int(graph.shape[0] * (getattr(model.config, "use_hop", 0) + 2))
            graph_idx_info_all = get_expanded_graph_key_query_indices(
                input_ids_1d=input_ids[0],
                attention_mask_1d=torch.ones_like(input_ids[0]),
                graphs=graph.cuda(),
                prompt_len=expanded_prompt_len,
                keep_pad_tokens=True,
                mm_use_graph_special_token=getattr(model.config, "mm_use_graph_special_token", False),
                use_hop=getattr(model.config, "use_hop", None),
                sample_neighbor_size=getattr(model.config, "sample_neighbor_size", None),
            )
            if graph_idx_info_all is not None:
                graph_key_idx_all = graph_idx_info_all[0].detach().cpu()
                graph_key_is_pad_all = graph_idx_info_all[3].detach().cpu()

        reposition_perm = None
        reposition_swapped = None
        reposition_applied = False
        has_sink = len(saved_all_sink_prompt_positions) > 0 and (
            args.reposition_mode != "front_top2" or len(saved_top2_sink_prompt_positions) >= 2
        )
        if args.reposition_mode != "none" and has_sink:
            expanded_len = int(input_ids.shape[1] - graph.shape[0] + graph.numel())
            if getattr(model.config, "mm_use_graph_special_token", False):
                expanded_len += int(graph.shape[0] * (getattr(model.config, "use_hop", 0) + 2))
            gidx = get_expanded_graph_key_query_indices(
                input_ids_1d=input_ids[0],
                attention_mask_1d=torch.ones_like(input_ids[0]),
                graphs=graph.cuda(),
                prompt_len=expanded_len,
                keep_pad_tokens=True,
                mm_use_graph_special_token=getattr(model.config, "mm_use_graph_special_token", False),
                use_hop=getattr(model.config, "use_hop", None),
                sample_neighbor_size=getattr(model.config, "sample_neighbor_size", None),
            )
            if gidx is not None:
                graph_pos_all = gidx[0].detach().cpu().tolist()
                graph_is_pad = gidx[3].detach().cpu().tolist()
                perm = list(range(expanded_len))
                if args.reposition_mode == "front_top2":
                    top2 = list(saved_top2_sink_prompt_positions)
                    others = [p for p in graph_pos_all if p not in top2]
                    new_graph = top2 + others
                    for out_pos, src_pos in zip(graph_pos_all, new_graph):
                        perm[out_pos] = src_pos
                    reposition_applied = True
                elif args.reposition_mode == "front_all":
                    sinks = [p for p in saved_all_sink_prompt_positions if p in set(graph_pos_all)]
                    sink_set = set(sinks)
                    others = [p for p in graph_pos_all if p not in sink_set]
                    new_graph = sinks + others
                    for out_pos, src_pos in zip(graph_pos_all, new_graph):
                        perm[out_pos] = src_pos
                    reposition_applied = True
                else:
                    seed_key = args.reposition_seed ^ (int(idx) if isinstance(idx, int) else hash(str(idx)))
                    rng = random.Random(seed_key)
                    all_sinks = list(saved_all_sink_prompt_positions)
                    sink_set = set(all_sinks)
                    nonsink_nonpad = [p for p, pad in zip(graph_pos_all, graph_is_pad) if (not pad) and (p not in sink_set)]
                    k = min(args.num_swap, len(all_sinks), len(nonsink_nonpad))
                    if k > 0:
                        chosen_sinks = rng.sample(all_sinks, k)
                        chosen_nonsinks = rng.sample(nonsink_nonpad, k)
                        for s, ns in zip(chosen_sinks, chosen_nonsinks):
                            perm[s], perm[ns] = perm[ns], perm[s]
                        reposition_swapped = {"sink_positions": chosen_sinks, "nonsink_positions": chosen_nonsinks}
                        reposition_applied = True
                if reposition_applied:
                    assert sorted(perm) == list(range(expanded_len)), "reposition_perm is not a valid permutation"
                    reposition_perm = torch.tensor(perm, dtype=torch.long).cuda()

        redistribute_applied = False
        redistribute_source_prompt_pos = None
        redistribute_nonsink_positions = []
        if args.redistribute:
            clear_redistribution_state()
            expanded_len = int(input_ids.shape[1] - graph.shape[0] + graph.numel())
            if getattr(model.config, "mm_use_graph_special_token", False):
                expanded_len += int(graph.shape[0] * (getattr(model.config, "use_hop", 0) + 2))
            gidx = get_expanded_graph_key_query_indices(
                input_ids_1d=input_ids[0],
                attention_mask_1d=torch.ones_like(input_ids[0]),
                graphs=graph.cuda(),
                prompt_len=expanded_len,
                keep_pad_tokens=True,
                mm_use_graph_special_token=getattr(model.config, "mm_use_graph_special_token", False),
                use_hop=getattr(model.config, "use_hop", None),
                sample_neighbor_size=getattr(model.config, "sample_neighbor_size", None),
            )
            if gidx is not None and len(saved_all_sink_prompt_positions) > 0:
                all_graph_pos = gidx[0].detach().cpu().tolist()
                all_graph_is_pad = gidx[3].detach().cpu().tolist()
                sink_set = set(int(p) for p in saved_all_sink_prompt_positions)
                redistribute_nonsink_positions = [
                    int(p) for p, pad in zip(all_graph_pos, all_graph_is_pad)
                    if (not pad) and (int(p) not in sink_set)
                ]
                if args.redistribute_direction == "src_to_sinks":
                    if args.redistribute_source_idx < len(all_graph_pos):
                        redistribute_source_prompt_pos = int(all_graph_pos[args.redistribute_source_idx])
                        set_redistribution_state(
                            direction="src_to_sinks",
                            fraction=args.redistribute_fraction,
                            sink_indices=list(saved_all_sink_prompt_positions),
                            source_idx=redistribute_source_prompt_pos,
                            layer_filter=redistribute_layer_filter,
                        )
                        redistribute_applied = True
                else:
                    if len(redistribute_nonsink_positions) > 0:
                        set_redistribution_state(
                            direction=args.redistribute_direction,
                            fraction=args.redistribute_fraction,
                            sink_indices=list(saved_all_sink_prompt_positions),
                            nonsink_indices=redistribute_nonsink_positions,
                            layer_filter=redistribute_layer_filter,
                        )
                        redistribute_applied = True

        steering_handles = []
        if args.steering_mode != "none":
            expanded_len = int(input_ids.shape[1] - graph.shape[0] + graph.numel())
            if getattr(model.config, "mm_use_graph_special_token", False):
                expanded_len += int(graph.shape[0] * (getattr(model.config, "use_hop", 0) + 2))
            attn_mask_1d = torch.ones_like(input_ids[0])
            steer_idx_info = get_expanded_graph_key_query_indices(
                input_ids_1d=input_ids[0],
                attention_mask_1d=attn_mask_1d,
                graphs=graph.cuda(),
                prompt_len=expanded_len,
                keep_pad_tokens=True,
                mm_use_graph_special_token=getattr(model.config, "mm_use_graph_special_token", False),
                use_hop=getattr(model.config, "use_hop", None),
                sample_neighbor_size=getattr(model.config, "sample_neighbor_size", None),
            )
            if steer_idx_info is not None:
                k_idx, q_idx, _, k_pad = steer_idx_info
                target_pos = select_target_positions(
                    target=args.steering_target,
                    key_idx=k_idx.detach().cpu(),
                    query_idx=q_idx.detach().cpu(),
                    key_is_pad=k_pad.detach().cpu(),
                    attention_mask_1d=attn_mask_1d.detach().cpu(),
                )
                if target_pos.numel() > 0:
                    sink_positions_tensor = None
                    skip_sample = False
                    if args.steering_source == "per_sample":
                        if not saved_all_sink_prompt_positions:
                            skip_sample = True   # no sinks detected → leave forward pass untouched
                        else:
                            sink_positions_tensor = torch.tensor(
                                saved_all_sink_prompt_positions, dtype=torch.long
                            )
                    if not skip_sample:
                        steering_handles = apply_contrastive_steering(
                            model=model,
                            sink_vectors=steering_vectors,
                            layer_indices=steering_layer_indices,
                            target_positions=target_pos,
                            mode=args.steering_mode,
                            strength=args.steering_strength,
                            sink_positions=sink_positions_tensor,
                        )["handles"]

        try:
            with torch.inference_mode():    # when calling model.generate, HF repeatedly calls `prepare_inputs_for_generation()` to build the next-step inputs
                output_ids = model.generate(
                    input_ids,
                    graph_emb=graph_emb.half().cuda(),
                    graph=graph.cuda(),
                    prune_token_positions=prune_token_positions,
                    reposition_perm=reposition_perm,
                    output_hidden_states=True,    # Hidden states from all layers
                    output_attentions=True,       # Attention weights from all layers
                    return_dict_in_generate=True, # Return a dict with all outputs
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,

                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            if not use_existing_sink_records:
                activation_analysis = compute_layerwise_graph_token_hidden_states(
                    generate_outputs=output_ids,
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    graphs=graph.cuda(),
                )

                if activation_analysis["valid"][0]:
                    layer_hidden = activation_analysis["layer_graph_hidden_states"][0]
                    key_is_pad = activation_analysis["key_is_pad"][0]
                    key_idx = activation_analysis["key_idx"][0]

                    if sink_activation_agg_state is None:
                        sink_activation_agg_state = init_sink_activation_agg_state(
                            num_layers=layer_hidden.shape[0],
                            hidden_dim=layer_hidden.shape[2],
                        )

                    sink_activation_agg_state = update_sink_activation_agg_state(
                        state=sink_activation_agg_state,
                        layer_graph_hidden_states=layer_hidden,
                        key_is_pad=key_is_pad,
                    )

                    if activation_topdims_state is None:
                        activation_topdims_state = init_activation_topdims_agg_state(
                            num_layers=layer_hidden.shape[0],
                            hidden_dim=layer_hidden.shape[2],
                        )
                    activation_topdims_state = update_activation_topdims_agg_state(
                        state=activation_topdims_state,
                        layer_graph_hidden_states=layer_hidden,
                    )

                    sinks = detect_sink_graph_tokens(
                        layer_graph_hidden_states=layer_hidden,
                        key_idx=key_idx,
                        key_is_pad=key_is_pad,
                        sink_dims=sink_dims_list,
                        threshold=sink_threshold,
                        ignore_pad_tokens=False
                    )

                    sink_scores = sinks["sink_scores"]
                    sink_token_indices = sinks["sink_token_indices"]

                    if len(sink_token_indices) > 0:
                        sink_token_tensor = torch.tensor(sink_token_indices, dtype=torch.long)

                        if sink_only_activation_topdims_state is None:
                            sink_only_activation_topdims_state = init_activation_topdims_agg_state(
                                num_layers=layer_hidden.shape[0],
                                hidden_dim=layer_hidden.shape[2],
                            )
                        sink_only_activation_topdims_state = update_activation_topdims_agg_state(
                            state=sink_only_activation_topdims_state,
                            layer_graph_hidden_states=layer_hidden,
                            token_indices=sink_token_tensor,
                        )

                        sink_scores = sink_scores[sink_token_tensor]

                        topk = min(2, sink_token_tensor.numel())
                        _, top_idx = torch.topk(sink_scores, k=topk)

                        top2_sink_token_tensor = sink_token_tensor[top_idx]
                        top2_sink_token_indices = top2_sink_token_tensor.tolist()
                        top2_sink_prompt_token_indices = key_idx[top2_sink_token_tensor].tolist()
                        top2_sink_scores = sink_scores[top_idx].tolist()
                        top2_sink_is_pad = key_is_pad[top2_sink_token_tensor].tolist()
                        all_sinks = key_idx[sink_token_indices].tolist()
                    else:
                        top2_sink_token_indices = []
                        top2_sink_prompt_token_indices = []
                        top2_sink_scores = []
                        top2_sink_is_pad = []
                        all_sinks = []

                    for rank, is_pad in enumerate(top2_sink_is_pad):
                        sink_pad_totals[rank] += 1
                        sink_pad_hits[rank] += int(is_pad)

                    all_sink_token_indices.append(top2_sink_token_indices)

                    if args.logit_lens and args.task == "nc":
                        logit_lens_total_seen += 1
                        # Filter on the *top-2* K-axis sinks (matches what's
                        # persisted in sink_records.jsonl as
                        # 'top2_sink_token_indices'). The full above-threshold
                        # set frequently spans many positions on LLaGA NC
                        # (K=111), so equality on it almost never holds even
                        # when the top sinks really are at {4, 5}.
                        sink_set = set(int(i) for i in top2_sink_token_indices)
                        if sink_set == logit_lens_filter_set:
                            ll = compute_logit_lens(
                                layer_graph_hidden_states=layer_hidden,
                                final_norm=model.model.norm,
                                lm_head=model.lm_head,
                                tokenizer=None,
                            )
                            logit_lens_records_ids.append(ll["top1_token_ids"])
                            logit_lens_records_probs.append(ll["top1_probs"])

                    sink_record = {
                        "question_id": idx,
                        "top2_sink_token_indices": top2_sink_token_indices,
                        "top2_sink_prompt_token_indices": top2_sink_prompt_token_indices,
                        "top2_sink_scores": top2_sink_scores,
                        "top2_sink_is_pad": top2_sink_is_pad,
                        "all_sink_indices": all_sinks,
                    }
                    sink_file.write(json.dumps(sink_record) + "\n")
                    sink_file.flush()

            if use_existing_sink_records and len(saved_all_sink_prompt_positions) > 0:
                sink_only_act = compute_layerwise_graph_token_hidden_states(
                    generate_outputs=output_ids,
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    graphs=graph.cuda(),
                )
                if sink_only_act["valid"][0]:
                    so_layer_hidden = sink_only_act["layer_graph_hidden_states"][0]
                    so_key_idx = sink_only_act["key_idx"][0].detach().cpu().tolist()
                    saved_set = {int(p) for p in saved_all_sink_prompt_positions}
                    sink_idx_in_K = torch.tensor(
                        [i for i, pos in enumerate(so_key_idx) if int(pos) in saved_set],
                        dtype=torch.long,
                    )
                    if sink_idx_in_K.numel() > 0:
                        if sink_only_activation_topdims_state is None:
                            sink_only_activation_topdims_state = init_activation_topdims_agg_state(
                                num_layers=so_layer_hidden.shape[0],
                                hidden_dim=so_layer_hidden.shape[2],
                            )
                        sink_only_activation_topdims_state = update_activation_topdims_agg_state(
                            state=sink_only_activation_topdims_state,
                            layer_graph_hidden_states=so_layer_hidden,
                            token_indices=sink_idx_in_K,
                        )

                    if args.logit_lens and args.task == "nc":
                        logit_lens_total_seen += 1
                        # Use the *top-2* K-axis sinks (the same quantity
                        # persisted as 'top2_sink_token_indices' in
                        # sink_records.jsonl). Map saved top-2 prompt
                        # positions back to K via so_key_idx.
                        saved_top2_set = {
                            int(p) for p in saved_top2_sink_prompt_positions
                        }
                        top2_idx_in_K = [
                            i for i, pos in enumerate(so_key_idx)
                            if int(pos) in saved_top2_set
                        ]
                        sink_set = set(top2_idx_in_K)
                        if sink_set == logit_lens_filter_set:
                            ll = compute_logit_lens(
                                layer_graph_hidden_states=so_layer_hidden,
                                final_norm=model.model.norm,
                                lm_head=model.lm_head,
                                tokenizer=None,
                            )
                            logit_lens_records_ids.append(ll["top1_token_ids"])
                            logit_lens_records_probs.append(ll["top1_probs"])

            if reposition_sink_file is not None:
                rep_top2_token_indices = []
                rep_top2_prompt_token_indices = []
                rep_top2_scores = []
                rep_top2_is_pad = []
                rep_all_sinks = []
                rep_activation = compute_layerwise_graph_token_hidden_states(
                    generate_outputs=output_ids,
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    graphs=graph.cuda(),
                )
                if rep_activation["valid"][0]:
                    rep_layer_hidden = rep_activation["layer_graph_hidden_states"][0]
                    rep_key_is_pad = rep_activation["key_is_pad"][0]
                    rep_key_idx = rep_activation["key_idx"][0]
                    if reposition_applied and reposition_perm is not None:
                        # pad mask comes from the original `graph` tensor; after permutation
                        # prompt position p holds the token originally at perm[p], so remap.
                        perm_cpu = reposition_perm.detach().cpu().long()
                        key_idx_cpu = rep_key_idx.detach().cpu().long()
                        full_pad = torch.zeros(perm_cpu.numel(), dtype=torch.bool)
                        full_pad[key_idx_cpu] = rep_key_is_pad.detach().cpu().bool()
                        rep_key_is_pad = full_pad[perm_cpu[key_idx_cpu]]
                    rep_sinks = detect_sink_graph_tokens(
                        layer_graph_hidden_states=rep_layer_hidden,
                        key_idx=rep_key_idx,
                        key_is_pad=rep_key_is_pad,
                        sink_dims=sink_dims_list,
                        threshold=sink_threshold,
                        ignore_pad_tokens=False,
                    )
                    rep_sink_scores = rep_sinks["sink_scores"]
                    rep_sink_token_indices = rep_sinks["sink_token_indices"]
                    if len(rep_sink_token_indices) > 0:
                        rep_tensor = torch.tensor(rep_sink_token_indices, dtype=torch.long)
                        rep_scores_sub = rep_sink_scores[rep_tensor]
                        rep_topk = min(2, rep_tensor.numel())
                        _, rep_top_idx = torch.topk(rep_scores_sub, k=rep_topk)
                        rep_top2_tensor = rep_tensor[rep_top_idx]
                        rep_top2_token_indices = rep_top2_tensor.tolist()
                        rep_top2_prompt_token_indices = rep_key_idx[rep_top2_tensor].tolist()
                        rep_top2_scores = rep_scores_sub[rep_top_idx].tolist()
                        rep_top2_is_pad = rep_key_is_pad[rep_top2_tensor].tolist()
                        rep_all_sinks = rep_key_idx[rep_sink_token_indices].tolist()
                all_reposition_sink_token_indices.append(rep_top2_token_indices)
                reposition_sink_file.write(json.dumps({
                    "question_id": idx,
                    "reposition_mode": args.reposition_mode,
                    "reposition_applied": reposition_applied,
                    "top2_sink_token_indices": rep_top2_token_indices,
                    "top2_sink_prompt_token_indices": rep_top2_prompt_token_indices,
                    "top2_sink_scores": rep_top2_scores,
                    "top2_sink_is_pad": rep_top2_is_pad,
                    "all_sink_indices": rep_all_sinks,
                }) + "\n")
                reposition_sink_file.flush()

            top2_sink_prompt_positions_for_stats = (
                saved_top2_sink_prompt_positions
                if use_existing_sink_records
                else top2_sink_prompt_token_indices
            )
            all_sink_prompt_positions_for_plot = (
                saved_all_sink_prompt_positions
                if use_existing_sink_records
                else all_sinks
            )

            attention_analysis = None
            need_attention_analysis = (
                args.template == "ND"
                or args.attention_probe
                or len(top2_sink_prompt_positions_for_stats) > 0
            )
            if need_attention_analysis:
                attention_analysis = compute_layerwise_query_to_graph_attention(
                    generate_outputs=output_ids,
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    graphs=graph.cuda(),
                    keep_pad_tokens=True,
                    mm_use_graph_special_token=getattr(model.config, 'mm_use_graph_special_token', False),
                    use_hop=getattr(model.config, 'use_hop', None),
                    sample_neighbor_size=getattr(model.config, 'sample_neighbor_size', None),
                )

            if (
                args.attention_probe
                and attention_analysis is not None
                and attention_analysis['valid'][0]
            ):
                # Layer-average up front to keep aggregator memory small.
                lqk = (
                    attention_analysis['layer_query_to_graph'][0]
                    .detach()
                    .cpu()
                    .to(torch.float32)
                )
                all_sample_layer_query_to_graph.append(lqk.mean(dim=0))   # [Q, K]
                all_sample_layer_to_graph_lk.append(lqk.mean(dim=1))      # [L, K]

            # remap_plot_path = None
            # if highest_attention_graph_token_records_path is not None:
            #     if attention_analysis is not None and attention_analysis['valid'][0]:
            #         try:
            #             remap_out = remap_and_plot_node_attention(
            #                 layer_query_to_graph=attention_analysis['layer_query_to_graph'][0],
            #                 graphs=graph,
            #                 key_is_pad=attention_analysis['key_is_pad'][0],
            #                 edge_index=data.edge_index,
            #                 save_path=_build_dataset_remap_plot_path(
            #                     dataset_name=args.dataset,
            #                     template_name=args.template,
            #                     sample_id=idx,
            #                 ),
            #             )
            #             remap_plot_path = remap_out["plot_path"]
            #             num_remap_plots += 1
            #         except Exception as e:
            #             num_remap_plot_skipped += 1
            #             print(f"Failed to save remap plot for sample {idx}: {e}")
            #     else:
            #         num_remap_plot_skipped += 1

                # highest_attention_record = build_highest_attention_graph_token_record(
                #     sample_id=idx,
                #     graphs=graph,
                #     layer_query_to_graph=(
                #         attention_analysis['layer_query_to_graph'][0]
                #         if attention_analysis is not None and attention_analysis['valid'][0]
                #         else None
                #     ),
                #     key_idx=(
                #         attention_analysis['key_idx'][0]
                #         if attention_analysis is not None and attention_analysis['valid'][0]
                #         else None
                #     ),
                #     key_is_pad=(
                #         attention_analysis['key_is_pad'][0]
                #         if attention_analysis is not None and attention_analysis['valid'][0]
                #         else None
                #     ),
                #     edge_index=data.edge_index,
                #     remap_plot_path=remap_plot_path,
                # )
                # append_highest_attention_graph_token_record(
                #     record=highest_attention_record,
                #     save_path=highest_attention_graph_token_records_path,
                # )

            top2_sink_attention_record = build_top2_sink_attention_nonpad_record(
                sample_id=idx,
                graphs=graph,
                top2_sink_prompt_positions=top2_sink_prompt_positions_for_stats,
                layer_query_to_graph=(
                    attention_analysis['layer_query_to_graph'][0]
                    if attention_analysis is not None and attention_analysis['valid'][0]
                    else None
                ),
                key_idx=(
                    attention_analysis['key_idx'][0]
                    if attention_analysis is not None and attention_analysis['valid'][0]
                    else None
                ),
                key_is_pad=(
                    attention_analysis['key_is_pad'][0]
                    if attention_analysis is not None and attention_analysis['valid'][0]
                    else None
                ),
            )
            append_top2_sink_attention_nonpad_record(
                record=top2_sink_attention_record,
                save_path=top2_sink_attention_nonpad_records_path,
            )

            # if args.attention_probe:
            #     if (
            #         attention_analysis is not None
            #         and attention_analysis['valid'][0]
            #         and len(all_sink_prompt_positions_for_plot) > 0
            #     ):
            #         out_path = plot_layeravg_query_graph_attention(
            #             layer_query_to_graph=attention_analysis['layer_query_to_graph'][0],
            #             sample_id=str(idx),
            #             save_path=_build_attention_to_sink_plot_path(
            #                 dataset_name=args.dataset,
            #                 template_name=args.template,
            #                 sample_id=idx,
            #             ),
            #             query_idx=attention_analysis['query_idx'][0],
            #             key_idx=attention_analysis['key_idx'][0],
            #             key_is_pad=attention_analysis['key_is_pad'][0],
            #             sink_prompt_positions=all_sink_prompt_positions_for_plot,
            #             top2_sink_prompt_positions=top2_sink_prompt_positions_for_stats,
            #         )
            #         if out_path is not None:
            #             num_attention_probe_plots += 1
            #         else:
            #             num_attention_probe_skipped += 1
            #     else:
            #         num_attention_probe_skipped += 1

            if do_sink_reoccur and graph_key_idx_all is not None:
                hs0 = output_ids.hidden_states[0][1:]
                pruned_prompt_positions = torch.tensor(
                    sorted({int(p) for p in prune_token_positions}),
                    dtype=torch.long,
                )
                keep_graph_mask = torch.ones_like(graph_key_idx_all, dtype=torch.bool)
                if pruned_prompt_positions.numel() > 0:
                    keep_graph_mask = (
                        graph_key_idx_all.unsqueeze(1) != pruned_prompt_positions.unsqueeze(0)
                    ).all(dim=1)

                if keep_graph_mask.any():
                    remaining_prompt_idx = graph_key_idx_all[keep_graph_mask]
                    remaining_key_is_pad = graph_key_is_pad_all[keep_graph_mask]
                    remaining_graph_idx = torch.arange(graph_key_idx_all.numel(), dtype=torch.long)[keep_graph_mask]
                    shift = (pruned_prompt_positions.unsqueeze(0) < remaining_prompt_idx.unsqueeze(1)).sum(dim=1)
                    reoccur_key_idx = remaining_prompt_idx - shift

                    reoccur_layer_hidden = torch.stack(
                        [
                            layer[0].to(torch.float32).index_select(0, reoccur_key_idx.to(layer[0].device))
                            for layer in hs0
                        ],
                        dim=0,
                    )
                    reoccur_sinks = detect_sink_graph_tokens(
                        layer_graph_hidden_states=reoccur_layer_hidden,
                        key_idx=reoccur_key_idx,
                        key_is_pad=remaining_key_is_pad,
                        sink_dims=sink_dims_list,
                        threshold=sink_threshold * 0.75,
                        ignore_pad_tokens=False,
                    )
                    reoccur_local_idx = reoccur_sinks["sink_token_indices"]
                    if len(reoccur_local_idx) > 0:
                        reoccur_tensor = torch.tensor(reoccur_local_idx, dtype=torch.long)
                        reoccur_sink_token_indices = remaining_graph_idx[reoccur_tensor].tolist()
                        reoccur_sink_prompt_token_indices = reoccur_key_idx[reoccur_tensor].tolist()
                        reoccur_sink_scores = reoccur_sinks["sink_scores"][reoccur_tensor].tolist()

                all_reoccur_sink_token_indices.append(reoccur_sink_token_indices)
                sink_reoccur_file.write(
                    json.dumps(
                        {
                            "question_id": idx,
                            "reoccur_sink_token_indices": reoccur_sink_token_indices,
                            "reoccur_sink_prompt_token_indices": reoccur_sink_prompt_token_indices,
                            "reoccur_sink_scores": reoccur_sink_scores,
                        }
                    )
                    + "\n"
                )
                sink_reoccur_file.flush()

            generated_sequences = output_ids.sequences
            input_token_len = input_ids.shape[1] 
            n_diff_input_output = (input_ids != generated_sequences[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(generated_sequences[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
        except Exception as e:
            print(f"!!!!!!Error!!!!! {e}")
            traceback.print_exc()
            # outputs=""
            raise
        finally:
            if steering_handles:
                remove_contrastive_steering(steering_handles)
                steering_handles = []

        ans_id = shortuuid.uuid()
        answer_record = {
            "question_id": idx,
            "prompt": cur_prompt,
            "graph": line['graph'],
            "text": outputs,
            "gt": line["conversations"][1]["value"],
            "answer_id": ans_id,
        }

        if args.pruning_nonsink:
            answer_record["pruned_nonsink_token_indices"] = selected_nonsink_token_indices
            answer_record["nonsink_excludes"] = "all_sink_indices"
            answer_record["seed"] = args.seed
            answer_record["run_idx"] = args.run_idx
        elif args.pruning:
            answer_record["pruned_sink_prompt_token_indices"] = prune_token_positions
            answer_record["pruning_mode"] = args.pruning_mode
        if args.reposition_mode != "none":
            answer_record["reposition_mode"] = args.reposition_mode
            answer_record["reposition_applied"] = reposition_applied
            answer_record["saved_top2_sink_prompt_positions"] = list(saved_top2_sink_prompt_positions)
            answer_record["saved_all_sink_prompt_positions"] = list(saved_all_sink_prompt_positions)
            if reposition_swapped is not None:
                answer_record["swapped_sink_positions"] = reposition_swapped["sink_positions"]
                answer_record["swapped_nonsink_positions"] = reposition_swapped["nonsink_positions"]
                answer_record["num_swap"] = args.num_swap
                answer_record["reposition_seed"] = args.reposition_seed
                answer_record["run_idx"] = args.run_idx
        if args.redistribute:
            answer_record["redistribute"] = True
            answer_record["redistribute_direction"] = args.redistribute_direction
            answer_record["redistribute_applied"] = redistribute_applied
            answer_record["redistribute_fraction"] = args.redistribute_fraction
            answer_record["redistribute_sink_prompt_positions"] = list(saved_all_sink_prompt_positions)
            answer_record["redistribute_layers"] = args.redistribute_layers
            if args.redistribute_direction == "src_to_sinks":
                answer_record["redistribute_source_idx"] = args.redistribute_source_idx
                answer_record["redistribute_source_prompt_pos"] = redistribute_source_prompt_pos
            else:
                answer_record["redistribute_nonsink_prompt_positions"] = redistribute_nonsink_positions
            answer_record["run_idx"] = args.run_idx
        if do_sink_reoccur:   # After pruning existing sinks, will new sink tokens emerge?
            answer_record["reoccur_sink_token_indices"] = reoccur_sink_token_indices
            answer_record["reoccur_sink_prompt_token_indices"] = reoccur_sink_prompt_token_indices
        answer_record["top2_sink_token_indices"] = top2_sink_token_indices
        answer_record["top2_sink_prompt_token_indices"] = top2_sink_prompt_token_indices
        answer_record["top2_sink_is_pad"] = top2_sink_is_pad
        ans_file.write(json.dumps(answer_record) + "\n")
        ans_file.flush()

    # RQ1: Identifying and Plotting Sink Dimensions
    if activation_topdims_state is not None:
        topdims_stats = finalize_activation_topdims_agg_state(activation_topdims_state, topk=3)
        topdims_plot_path, topdims_avg_plot_path, top_sink_dims = plot_topdims_mean_activation_curve(
            aggregated=topdims_stats,
            save_path=f"{analysis_dir}/sink_dim_mean_activation.png",
            sink_threshold=15.0,
            layer_index=-2,
            use_abs=True
        )
        print(f"Saved layer-specific sink-dimension activation plot to {topdims_plot_path}")
        print(f"Saved layer-averaged sink-dimension activation plot to {topdims_avg_plot_path}")
        print(f"Sink dimensions above threshold (token-averaged at the chosen layer): {top_sink_dims}")
        np.save(
            os.path.join(rq_arrays_dir, "mean_per_dim_all.npy"),
            topdims_stats["mean_all"].detach().cpu().to(torch.float32).numpy(),
        )
        np.save(
            os.path.join(rq_arrays_dir, "mean_per_dim_all_signed.npy"),
            topdims_stats["mean_all_signed"].detach().cpu().to(torch.float32).numpy(),
        )

    if sink_only_activation_topdims_state is not None:
        sink_only_topdims_stats = finalize_activation_topdims_agg_state(
            sink_only_activation_topdims_state, topk=3
        )
        sink_only_plot_path, sink_only_avg_plot_path, sink_only_top_dims = plot_topdims_mean_activation_curve(
            aggregated=sink_only_topdims_stats,
            save_path=f"{analysis_dir}/sink_only_dim_mean_activation.png",
            sink_threshold=15.0,
            layer_index=-2,
            use_abs=True,
        )
        print(f"Saved sink-only layer-specific activation plot to {sink_only_plot_path}")
        print(f"Saved sink-only layer-averaged activation plot to {sink_only_avg_plot_path}")
        print(f"Sink dimensions above threshold (sink-only, token-averaged at chosen layer): {sink_only_top_dims}")
        np.save(
            os.path.join(rq_arrays_dir, "mean_per_dim_sink_only.npy"),
            sink_only_topdims_stats["mean_all"].detach().cpu().to(torch.float32).numpy(),
        )
        np.save(
            os.path.join(rq_arrays_dir, "mean_per_dim_sink_only_signed.npy"),
            sink_only_topdims_stats["mean_all_signed"].detach().cpu().to(torch.float32).numpy(),
        )


    if args.logit_lens and args.task == "nc":
        if len(logit_lens_records_ids) > 0:
            agg = aggregate_logit_lens(
                top1_token_ids_list=logit_lens_records_ids,
                top1_probs_list=logit_lens_records_probs,
                tokenizer=tokenizer,
            )
            ll_dir = os.path.join(analysis_dir, "logit_lens")
            ll_path = plot_logit_lens_heatmap(
                top1_strings=agg["top1_strings"],
                top1_probs=agg["top1_probs"],
                save_path=os.path.join(ll_dir, "logit_lens.png"),
                row_indices=list(LOGIT_LENS_DISPLAY_POSITIONS),
                sink_indices=list(LOGIT_LENS_SINK_POSITIONS),
            )
            print(
                f"Logit-lens heatmap saved to {ll_path} "
                f"(kept {agg['num_samples']}/{logit_lens_total_seen} samples with sinks "
                f"== {sorted(logit_lens_filter_set)}, "
                f"mean agreement {float(agg['agreement'].mean()):.3f})"
            )
        else:
            print(
                f"Logit-lens: 0/{logit_lens_total_seen} samples matched the "
                f"sinks == {sorted(logit_lens_filter_set)} filter — heatmap skipped."
            )

    if not args.pruning and len(all_sink_token_indices) > 0:
        sink_hist_path = plot_sink_token_index_histogram(
            all_sink_token_indices=all_sink_token_indices,
            num_graph_tokens=num_graph_tokens,
            save_path=f"{analysis_dir}/sink_distribution.png",
        )
        print(f"Saved sink-token histogram to {sink_hist_path}")
    if len(all_reoccur_sink_token_indices) > 0:
        sink_reoccur_hist_path = plot_sink_token_index_histogram(
            all_sink_token_indices=all_reoccur_sink_token_indices,
            num_graph_tokens=num_graph_tokens,
            save_path=f"{analysis_dir}/sink_reoccur_distribution.png",
        )
        print(f"Saved reoccur sink-token histogram to {sink_reoccur_hist_path}")

        baseline_source = (
            args.sink_records_path
            if getattr(args, "sink_records_path", None) and os.path.exists(args.sink_records_path)
            else sink_records_path
        )
        baseline_top2_indices = []
        if os.path.exists(baseline_source):
            with open(baseline_source, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    baseline_top2_indices.append(
                        [int(i) for i in rec.get("top2_sink_token_indices", [])]
                    )
        if len(baseline_top2_indices) > 0:
            shift_path = plot_sink_distribution_shift(
                baseline_sink_token_indices=baseline_top2_indices,
                post_sink_token_indices=all_reoccur_sink_token_indices,
                num_graph_tokens=num_graph_tokens,
                save_path=f"{analysis_dir}/sink_distribution_shift.png",
                title=f"{args.dataset}/{args.template}: sink position distribution — baseline vs after pruning",
            )
            print(f"Saved sink distribution shift overlay to {shift_path}")
        else:
            print(
                f"Skipped sink distribution shift overlay: no baseline records at {baseline_source}"
            )
    if reposition_tag is not None and len(all_reposition_sink_token_indices) > 0:
        reposition_sink_hist_path = plot_sink_token_index_histogram(
            all_sink_token_indices=all_reposition_sink_token_indices,
            num_graph_tokens=num_graph_tokens,
            save_path=f"{analysis_dir}/reposition_{reposition_tag}_sink_distribution.png",
        )
        print(f"Saved reposition sink-token histogram to {reposition_sink_hist_path}")
    for rank, total in enumerate(sink_pad_totals, start=1):
        if total > 0:
            pct = 100.0 * sink_pad_hits[rank - 1] / total
            print(f"Top-{rank} sink token is a graph pad token: {pct:.2f}% ({sink_pad_hits[rank - 1]}/{total})")
    aggregated_top2_sink_attention = aggregate_top2_sink_attention_nonpad_from_jsonl(
        records_path=top2_sink_attention_nonpad_records_path,
    )
    if not args.pruning and aggregated_top2_sink_attention['n_valid_samples'] > 0:
        top2_sink_attention_plot_path = plot_top2_sink_attention_vs_nonpad_percentage(
            aggregated=aggregated_top2_sink_attention,
            save_path=f'{analysis_dir}/top2_sink_attention_vs_nonpad_percentage.png',
        )
        print(
            'Saved top-2 sink attention vs non-padded graph-token % plot to '
            f'{top2_sink_attention_plot_path} '
            f'(valid={aggregated_top2_sink_attention["n_valid_samples"]}/{aggregated_top2_sink_attention["n_samples"]})'
        )
    else:
        print('No valid samples for top-2 sink attention vs non-padded graph-token % plot.')

    if args.attention_probe and len(all_sample_layer_to_graph_lk) > 0:
        layer_vs_graph_aggregated = aggregate_layer_vs_graph_attention(
            sample_layer_to_graph=all_sample_layer_to_graph_lk,
        )
        layer_vs_graph_plot_path = plot_layer_vs_graph_attention_heatmap(
            aggregated=layer_vs_graph_aggregated,
            save_path=f"{analysis_dir}/cross_attention_layer_vs_graph_heatmap.png",
            title=(
                f"{args.dataset}: per-layer mean attention to graph tokens "
                f"(n={layer_vs_graph_aggregated['n_samples']} samples, head-avg, query-avg, sample-avg)"
            ),
        )
        print(f"Saved layer-vs-graph attention heatmap to {layer_vs_graph_plot_path}")
        np.save(
            os.path.join(rq_arrays_dir, "layer_to_graph.npy"),
            layer_vs_graph_aggregated["mean_layer_to_graph"].detach().cpu().to(torch.float32).numpy(),
        )

    if highest_attention_graph_token_records_path is not None:
        aggregated_highest_attention = aggregate_highest_attention_graph_token_from_jsonl(
            records_path=highest_attention_graph_token_records_path,
        )
        highest_attention_summary_path = (
            f"{analysis_dir}/highest_attention_graph_token_summary.json"
        )
        highest_attention_degree_plot_path = None
        if aggregated_highest_attention["n_valid_samples"] > 0:
            highest_attention_degree_plot_path = plot_highest_attention_degree_vs_attention(
                aggregated=aggregated_highest_attention,
                save_path=(
                    f"{analysis_dir}/"
                    "highest_attention_degree_vs_attention.png"
                ),
            )

        highest_attention_summary = {
            "num_samples": aggregated_highest_attention["n_samples"],
            "num_valid_samples": aggregated_highest_attention["n_valid_samples"],
            "hop_counts": aggregated_highest_attention["hop_counts"],
            "hop_percentages": aggregated_highest_attention["hop_percentages"],
            "avg_highest_node_degree": aggregated_highest_attention["avg_highest_node_degree"],
            "avg_center_node_degree": aggregated_highest_attention["avg_center_node_degree"],
            "avg_degree_gap_highest_minus_center": (
                aggregated_highest_attention["avg_degree_gap_highest_minus_center"]
            ),
            "highest_attention_graph_token_records_path": (
                highest_attention_graph_token_records_path
            ),
            "highest_attention_degree_plot_path": highest_attention_degree_plot_path,
        }
        with open(highest_attention_summary_path, "w", encoding="utf-8") as f:
            json.dump(highest_attention_summary, f, indent=2)

        print(
            "Highest-attention graph-token hop percentages: "
            f"center {aggregated_highest_attention['hop_percentages']['center']:.2f}%, "
            f"one_hop {aggregated_highest_attention['hop_percentages']['one_hop']:.2f}%, "
            f"two_hop {aggregated_highest_attention['hop_percentages']['two_hop']:.2f}%, "
            f"rest {aggregated_highest_attention['hop_percentages']['rest']:.2f}% "
            f"(valid={aggregated_highest_attention['n_valid_samples']}/"
            f"{aggregated_highest_attention['n_samples']})"
        )
        print(
            "Average node degree comparison: "
            f"highest-attention node {aggregated_highest_attention['avg_highest_node_degree']:.2f} "
            f"vs center node {aggregated_highest_attention['avg_center_node_degree']:.2f}"
        )
        print(f"Saved highest-attention summary to {highest_attention_summary_path}")
        if highest_attention_degree_plot_path is not None:
            print(
                "Saved highest-attention node degree vs attention plot to "
                f"{highest_attention_degree_plot_path}"
            )
        if num_remap_plots > 0 or num_remap_plot_skipped > 0:
            print(
                "Node remap plots: "
                f"saved {num_remap_plots}, skipped {num_remap_plot_skipped} samples."
            )

    if args.attention_probe and len(all_sample_layer_query_to_graph) > 0:
        dataset_avg_attention = plot_final_average_cross_attention_for_dataset(
            sample_layer_query_to_graph=all_sample_layer_query_to_graph,
            save_path=(
                f"{analysis_dir}/"
                "dataset_avg_layer_avg_attention.png"
            ),
        )
        print(
            "Saved dataset-level layer-avg cross-attention heatmap to "
            f"{dataset_avg_attention['plot_path']} "
            f"(valid={dataset_avg_attention['n_valid_samples']}/"
            f"{dataset_avg_attention['n_samples']})"
        )
        np.save(
            os.path.join(rq_arrays_dir, "query_to_graph.npy"),
            dataset_avg_attention["avg_query_to_graph"].detach().cpu().to(torch.float32).numpy(),
        )

    if args.attention_probe and (num_attention_probe_plots > 0 or num_attention_probe_skipped > 0):
        print(
            'Attention probes: '
            f'all-graph saved {num_attention_probe_plots}, '
            f'skipped {num_attention_probe_skipped} samples.'
        )


    # Plot pre-graph activation plots
    # if pregraph_activation_agg_state is not None:
    #     pregraph_activation_stats = finalize_pregraph_activation_agg_state(
    #         pregraph_activation_agg_state
    #     )
    #     pregraph_activation_plot_paths = plot_pregraph_activation_curves(
    #         aggregated=pregraph_activation_stats,
    #         save_dir=f"{analysis_dir}/pregraph_activation_probes_aggregated",
    #     )
    if sink_file is not None:
        sink_file.close()
    if sink_reoccur_file is not None:
        sink_reoccur_file.close()
    if reposition_sink_file is not None:
        reposition_sink_file.close()
    ans_file.close()
    if args.redistribute:
        uninstall_redistribution(model)
    if rename_answers_file_to is not None and answers_file != rename_answers_file_to:
        os.replace(answers_file, rename_answers_file_to)
        print(f"Renamed answers file to {rename_answers_file_to}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--pretrained_embedding_type", type=str, default="sbert")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=5)
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="If set, evaluate at most this many samples (applied after --start/--end). "
                             "Useful for quick runs on large test sets like ogbn-arxiv.")
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--mm_use_graph_start_end",default=False, action="store_true")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--template", type=str, default="ND")
    parser.add_argument("--num_prune", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sink_records_path", type=str, default=None)
    parser.add_argument("--sink_dims", type=str, default="1512,2298,2533",
                        help="Comma-separated dim indices used by detect_sink_graph_tokens. "
                             "Default is calibrated for NC; for LP try '2533,2789,363'.")
    parser.add_argument("--sink_threshold", type=float, default=20.0,
                        help="Token-level sink threshold (max RMSNorm value across --sink_dims "
                             "at layer -2). Default 20.0 is calibrated for NC; LP layer-(-2) "
                             "magnitudes top out near 5, so try 3.0.")
    parser.add_argument("--pruning", action="store_true")
    parser.add_argument("--pruning_nonsink", action="store_true")
    parser.add_argument("--pruning_mode", type=str, default="top2", choices=["top2", "all"])
    parser.add_argument("--sink_reoccur", action="store_true")
    parser.add_argument("--attention_probe", action="store_true")
    parser.add_argument("--reposition_mode", type=str, default="none",
                        choices=["none", "front_top2", "front_all", "swap_sink_nonsink"])
    parser.add_argument("--num_swap", type=int, default=2)
    parser.add_argument("--reposition_seed", type=int, default=42)
    parser.add_argument("--run_idx", type=int, default=1,
                        help="Run index used in output filename (run1, run2, ...). "
                             "Multiple runs with the same seed reveal inference nondeterminism.")
    parser.add_argument("--use_existing_sink_records", action="store_true")
    parser.add_argument("--redistribute", action="store_true",
                        help="Enable attention-mass redistribution. See --redistribute_direction.")
    parser.add_argument("--redistribute_direction", type=str, default="src_to_sinks",
                        choices=["src_to_sinks", "sinks_to_top_nonsink", "sinks_to_nonsink_even",
                                 "sinks_to_nonsink_value_sim"],
                        help="Where the redistributed mass flows. "
                             "'src_to_sinks': shave source col by fraction p, spread to sinks (original). "
                             "'sinks_to_top_nonsink': shave each sink col by fraction p, dump removed mass into per-cell argmax over non-sink graph tokens. "
                             "'sinks_to_nonsink_even': shave each sink col by fraction p, spread removed mass evenly across non-sink graph tokens. "
                             "'sinks_to_nonsink_value_sim': shave each sink col by p, softmax-weight removed mass across non-sink graph tokens by V_j . Delta_i / ||V_j||.")
    parser.add_argument("--redistribute_fraction", type=float, default=0.5,
                        help="Fraction p in [0, 1] of each source/sink column's attention to redistribute. "
                             "p=0 is a no-op; p=1 fully zeros the source/sink columns.")
    parser.add_argument("--redistribute_source_idx", type=int, default=11,
                        help="Only used for direction=src_to_sinks: index into key_idx of the source token.")
    parser.add_argument("--redistribute_layers", type=str, default="all",
                        help='Layers to apply redistribution to. "all", a comma list "8,9,10", or a range "0-7".')
    # Contrastive steering against structural sinks (RQ4).
    parser.add_argument("--steering_mode", type=str, default="none",
                        choices=["none", "subtract", "project"])
    parser.add_argument("--steering_layers", type=str, default="-2",
                        help='Decoder layers to hook. "all", a comma list "8,9,10" '
                             '(negatives allowed: "-1,-2"), or a non-negative range "0-7". '
                             'Default "-2" = second-to-last only.')
    parser.add_argument("--steering_strength", type=float, default=1.0,
                        help="gamma (subtract) or alpha in [0,1] (project).")
    parser.add_argument("--steering_target", type=str, default="query",
                        choices=["query", "graph", "both", "all_nonpad"])
    parser.add_argument("--steering_vector_path", type=str, default=None,
                        help="Path to [num_layers, hidden_dim] npy. Default: "
                             "analysis/<dataset>_<template>/rq_arrays/mean_per_dim_sink_only_signed.npy")
    parser.add_argument("--steering_use_abs", action="store_true",
                        help="Use _sink_only.npy (|RMSNorm| view) instead of the signed file.")
    parser.add_argument("--logit_lens", action="store_true",
                        help="Save a logit-lens heatmap (top-1 token per layer x graph token) "
                             "for NC samples whose detected sink graph-token indices are "
                             f"exactly {list(LOGIT_LENS_SINK_POSITIONS)}. Heatmap displays "
                             f"graph tokens {list(LOGIT_LENS_DISPLAY_POSITIONS)}; rows in "
                             "the sink set are marked with an extra <s> tag. Modal-token "
                             "aggregation across qualifying samples; cell colour is the mean "
                             "top-1 prob restricted to the modal token.")
    parser.add_argument("--steering_source", type=str, default="global",
                        choices=["global", "per_sample"],
                        help="'global': use precomputed dataset-mean sink direction from the npy. "
                             "'per_sample': for each sample, derive s online from the saved "
                             "sink-token positions' hidden states at the steered layer. "
                             "per_sample requires --use_existing_sink_records.")
    args = parser.parse_args()

    eval_model(args)
