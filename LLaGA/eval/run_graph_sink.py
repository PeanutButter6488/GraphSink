from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
for path in (str(THIS_DIR), str(REPO_ROOT), str(WORKSPACE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from glm_sink import (  # noqa: E402
    ArtifactWriter,
    GraphModelAdapter,
    GraphSinkPipeline,
    PruningExperiment,
    PruningSpec,
    SinkConfig,
    TraceExample,
    load_sink_record_index,
    load_spike_summary,
)
from utils.activation_probes import compute_layerwise_graph_token_hidden_states  # noqa: E402
from utils.attention_probes import compute_layerwise_query_to_graph_attention  # noqa: E402
from utils.constants import DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN, GRAPH_TOKEN_INDEX  # noqa: E402
from utils.utils import disable_torch_init, get_model_name_from_path, tokenizer_graph_token  # noqa: E402


class LLaGAAdapter(GraphModelAdapter):
    def __init__(
        self,
        artifact_root: str,
        dataset_tag: str,
        sink_dims: Optional[List[int]] = None,
        threshold: float = 20.0,
    ):
        self._artifact_root = artifact_root
        self._dataset_tag = dataset_tag
        self._sink_dims = sink_dims if sink_dims is not None else [1512, 2298, 2533]
        self._threshold = threshold

    def trace_batch(
        self,
        *,
        sample_ids: List[int],
        generate_outputs,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graphs: torch.Tensor,
        with_attentions: bool,
        metadata: Optional[List[Dict[str, int]]] = None,
    ) -> List[TraceExample]:
        activation = compute_layerwise_graph_token_hidden_states(
            generate_outputs=generate_outputs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            graphs=graphs,
            keep_pad_tokens=True,
        )
        attention = None
        if with_attentions:
            attention = compute_layerwise_query_to_graph_attention(
                generate_outputs=generate_outputs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                graphs=graphs,
                keep_pad_tokens=True,
            )

        examples: List[TraceExample] = []
        for batch_idx, sample_id in enumerate(sample_ids):
            if not activation["valid"][batch_idx]:
                continue
            examples.append(
                TraceExample(
                    sample_id=sample_id,
                    layer_graph_hidden_states=activation["layer_graph_hidden_states"][batch_idx].detach().cpu(),
                    graph_token_positions=activation["key_idx"][batch_idx].detach().cpu(),
                    graph_token_is_pad=activation["key_is_pad"][batch_idx].detach().cpu(),
                    layer_query_to_graph_attention=(
                        attention["layer_query_to_graph"][batch_idx].detach().cpu()
                        if attention is not None and attention["valid"][batch_idx]
                        else None
                    ),
                    query_token_positions=(
                        attention["query_idx"][batch_idx].detach().cpu()
                        if attention is not None and attention["valid"][batch_idx]
                        else None
                    ),
                    metadata=(metadata or [{}])[batch_idx],
                )
            )
        return examples

    def get_decoder_layers(self, model) -> List[torch.nn.Module]:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
            return list(model.model.decoder.layers)
        raise ValueError("Could not locate decoder layers on the LLaGA model.")

    def default_sink_config(self) -> SinkConfig:
        return SinkConfig(
            model_name="LLaGA",
            dataset_tag=self._dataset_tag,
            hidden_dim=4096,
            expected_graph_tokens=None,
            explicit_sink_dims=list(self._sink_dims),
            num_sink_dims=len(self._sink_dims),
            threshold=float(self._threshold),
            activation_topk_per_layer=5,
            allow_padded_sinks=True,
            artifact_root=self._artifact_root,
        )

def _parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _default_output_root() -> str:
    return str(WORKSPACE_ROOT)


def _build_graph_emb(args, *, graph_line, data, pretrained_emb, structure_emb, index):
    from eval_pretrain import SMALL_DATASETS
    from torch_geometric.utils import add_self_loops, degree, k_hop_subgraph, remove_self_loops

    if not isinstance(graph_line[0], list):
        graph_line = [graph_line]

    if args.template == "ND":
        graph = torch.LongTensor(graph_line)
        mask = graph != DEFAULT_GRAPH_PAD_ID
        masked_graph_emb = pretrained_emb[graph[mask]]
        num_graphs, num_tokens, hidden_dim = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
        graph_emb = torch.zeros((num_graphs, num_tokens, hidden_dim))
        graph_emb[mask] = masked_graph_emb
        if structure_emb is not None:
            graph_emb = torch.cat([graph_emb, structure_emb.unsqueeze(0).expand(num_graphs, -1, -1)], dim=-1)
        return graph, graph_emb

    if args.template != "HO":
        raise ValueError(f"Unsupported template: {args.template}")

    if args.dataset in SMALL_DATASETS and args.task == "lp":
        from eval_pretrain import MP

        mp = MP()
        center_nodes = []
        adjusted = []
        for graph_nodes in graph_line:
            center_id = graph_nodes[0]
            adjusted.append([center_id] * (args.use_hop + 1))
            center_nodes.append(center_id)
        graph = torch.LongTensor(adjusted)
        center_id = graph[:, 0]
        graph_embs = [pretrained_emb[center_id].cuda()]
        subset, edge_index, mapping, _ = k_hop_subgraph(center_nodes, args.use_hop, data.edge_index, relabel_nodes=True)
        local_edge_mask = ((edge_index[0] == mapping[0]) & (edge_index[1] == mapping[1])) | (
            (edge_index[0] == mapping[1]) & (edge_index[1] == mapping[0])
        )
        edge_index = edge_index[:, ~local_edge_mask]
        local_x = pretrained_emb[subset].cuda()
        num_nodes = subset.shape[0]
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index)
        edge_index = edge_index.cuda()
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=pretrained_emb.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        for _ in range(args.use_hop):
            local_x = mp.propagate(edge_index, x=local_x, norm=norm)
            graph_embs.append(local_x[mapping])
        return graph, torch.stack(graph_embs, dim=1)

    adjusted = []
    for graph_nodes in graph_line:
        center_id = graph_nodes[0]
        adjusted.append([center_id] * (args.use_hop + 1))
    graph = torch.LongTensor(adjusted)
    center_id = graph[:, 0]
    graph_emb = torch.stack([emb[index[center_id]] for emb in pretrained_emb], dim=1)
    return graph, graph_emb


def _load_llaga_context(args: argparse.Namespace):
    from eval_pretrain import (
        load_pretrain_embedding_graph,
        load_pretrain_embedding_hop,
        load_pretrain_embedding_hop_lp,
        resolve_eval_data_dir,
        resolve_eval_paths,
    )
    from model.builder import load_pretrained_model

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        cache_dir=args.cache_dir,
    )
    model = model.to(torch.float16).cuda()

    data_dir = resolve_eval_data_dir(args)
    prompt_file, data_path = resolve_eval_paths(args, data_dir)
    data = torch.load(data_path, weights_only=False)
    questions = [json.loads(line) for line in open(prompt_file, "r", encoding="utf-8")]

    if args.start >= 0:
        end = len(questions) if args.end < 0 else args.end
        questions = questions[args.start:end]
    elif args.end > 0:
        questions = questions[: args.end]

    index = None
    if args.template == "ND":
        pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
        structure_emb = torch.load(os.path.join(str(REPO_ROOT), "dataset", f"laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt"))
    elif args.template == "HO":
        num_nodes = data.num_nodes
        if args.dataset in SMALL_DATASETS and args.task == "lp":
            pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
        elif args.task == "lp":
            pretrained_emb, mask = load_pretrain_embedding_hop_lp(data_dir, args.pretrained_embedding_type, args.use_hop)
            index = torch.full([num_nodes], fill_value=num_nodes + 1, dtype=torch.long)
            index[mask] = torch.arange(mask.sum())
        else:
            mask = torch.full([num_nodes], fill_value=False, dtype=torch.bool)
            for question in questions:
                qid = question["id"]
                if args.task == "lp":
                    mask[qid[0]] = True
                    mask[qid[1]] = True
                else:
                    mask[qid] = True
            pretrained_emb = load_pretrain_embedding_hop(data_dir, args.pretrained_embedding_type, args.use_hop, mask)
            index = torch.full([num_nodes], fill_value=num_nodes + 1, dtype=torch.long)
            index[mask] = torch.arange(mask.sum())
        structure_emb = None
    else:
        raise ValueError(f"Unsupported template: {args.template}")

    return tokenizer, model, data, questions, pretrained_emb, structure_emb, index


def _iter_trace_examples(
    args: argparse.Namespace,
    adapter: LLaGAAdapter,
    *,
    with_attentions: bool,
) -> Iterable[TraceExample]:
    from eval_pretrain import resolve_nc_prompt
    from utils.conversation import conv_templates

    tokenizer, model, data, questions, pretrained_emb, structure_emb, index = _load_llaga_context(args)

    for row_index, line in enumerate(tqdm(questions, desc=args.stage)):
        sample_id = int(line["id"])
        if args.task in {"nd", "nda"}:
            qs = f"Please briefly describe the center node of {DEFAULT_GRAPH_TOKEN}."
        elif args.task == "nc":
            qs = resolve_nc_prompt(args, line)
        else:
            raise ValueError(f"Unsupported task for graph-sink runner: {args.task}")

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_graph_token(prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        attention_mask = torch.ones_like(input_ids)
        graph, graph_emb = _build_graph_emb(
            args,
            graph_line=line["graph"],
            data=data,
            pretrained_emb=pretrained_emb,
            structure_emb=structure_emb,
            index=index,
        )

        with torch.inference_mode():
            generate_outputs = model.generate(
                input_ids,
                graph_emb=graph_emb.half().cuda(),
                graph=graph.cuda(),
                output_hidden_states=True,
                output_attentions=with_attentions,
                return_dict_in_generate=True,
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        examples = adapter.trace_batch(
            sample_ids=[sample_id],
            generate_outputs=generate_outputs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            graphs=graph.cuda(),
            with_attentions=with_attentions,
            metadata=[{"source_row_index": row_index}],
        )
        for example in examples:
            yield example


def _run_pruning(
    args: argparse.Namespace,
    adapter: LLaGAAdapter,
    config: SinkConfig,
    sink_dims: List[int],
) -> str:
    from eval_pretrain import resolve_nc_prompt
    from utils.conversation import conv_templates

    writer = ArtifactWriter(
        config.artifact_root,
        model_name=config.model_name,
        dataset_tag=config.dataset_tag,
    )
    records_path = args.sink_records_path or writer.sink_records_path
    record_index = load_sink_record_index(records_path)
    experiment = None

    tokenizer, model, data, questions, pretrained_emb, structure_emb, index = _load_llaga_context(args)
    experiment = PruningExperiment(adapter.get_decoder_layers(model))
    spec = PruningSpec(mode=args.prune_mode, selection=args.selection, k=args.k, seed=args.seed)
    meta_rows: List[Dict[str, object]] = []

    for row_index, line in enumerate(tqdm(questions, desc="prune")):
        sample_id = int(line["id"])
        record = record_index["by_sample_id"].get(str(sample_id))
        if record is None:
            record = record_index["by_source_row_index"].get(str(row_index))
        if record is None:
            meta_rows.append({"sample_id": sample_id, "source_row_index": row_index, "skipped": "missing_record"})
            continue

        plan = experiment.build_plan(
            sink_record=record,
            spec=spec,
            sink_dims=sink_dims,
            hidden_dim=config.hidden_dim,
        )
        meta_rows.append({"sample_id": sample_id, "source_row_index": row_index, **plan})

        if args.task in {"nd", "nda"}:
            qs = f"Please briefly describe the center node of {DEFAULT_GRAPH_TOKEN}."
        elif args.task == "nc":
            qs = resolve_nc_prompt(args, line)
        else:
            raise ValueError(f"Unsupported task for graph-sink runner: {args.task}")

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_graph_token(prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        graph, graph_emb = _build_graph_emb(
            args,
            graph_line=line["graph"],
            data=data,
            pretrained_emb=pretrained_emb,
            structure_emb=structure_emb,
            index=index,
        )

        handles = experiment.attach([plan])
        try:
            with torch.inference_mode():
                model.generate(
                    input_ids,
                    graph_emb=graph_emb.half().cuda(),
                    graph=graph.cuda(),
                    do_sample=False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
        finally:
            experiment.remove(handles)

    writer.save_json(
        writer.pruning_meta_path(spec.mode),
        {
            "model_name": config.model_name,
            "records_path": records_path,
            "spec": {
                "mode": spec.mode,
                "selection": spec.selection,
                "k": spec.k,
                "seed": spec.seed,
            },
            "num_samples": len(meta_rows),
            "plans": meta_rows,
        },
    )
    return writer.pruning_meta_path(spec.mode)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["discover_dims", "detect_tokens", "discover_spikes", "detect_and_analyze", "prune"],
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--pretrained_embedding_type", type=str, default="sbert")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=5)
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--template", type=str, default="ND")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--run_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--sink_dims", type=str, default=None)
    parser.add_argument("--auto_sink_dims", action="store_true")
    parser.add_argument("--num_sink_dims", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--activation_topk_per_layer", type=int, default=5)
    parser.add_argument("--spike_summary_path", type=str, default=None)
    parser.add_argument("--sink_records_path", type=str, default=None)
    parser.add_argument("--prune_mode", choices=["sink_token", "control_nonsink_token", "sink_dim", "control_nonsink_dim"], default="sink_token")
    parser.add_argument("--selection", choices=["all", "topk"], default="all")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    output_root = args.output_root or args.run_root or _default_output_root()
    dataset_tag = f"{args.dataset}_{args.template}"

    adapter = LLaGAAdapter(
        artifact_root=output_root,
        dataset_tag=dataset_tag,
        sink_dims=_parse_int_list(args.sink_dims),
        threshold=args.threshold,
    )
    config = adapter.default_sink_config()
    config.activation_topk_per_layer = int(args.activation_topk_per_layer)
    config.num_sink_dims = int(args.num_sink_dims)
    if args.auto_sink_dims:
        config.explicit_sink_dims = None
    elif _parse_int_list(args.sink_dims) is not None:
        config.explicit_sink_dims = _parse_int_list(args.sink_dims)

    writer = ArtifactWriter(
        config.artifact_root,
        model_name=config.model_name,
        dataset_tag=config.dataset_tag,
    )
    pipeline = GraphSinkPipeline(config, writer)

    if args.stage in {"discover_dims", "discover_spikes"}:
        summary = pipeline.discover_dimensions(_iter_trace_examples(args, adapter, with_attentions=False))
        print(f"Saved sink-dimension summary to {writer.dimension_summary_path}")
        print(f"Saved sink-dimension plot to {writer.dimension_plot_path}")
        print(f"Selected sink dims: {summary.selected_sink_dims} ({summary.selection_source})")
        return

    if args.stage in {"detect_tokens", "detect_and_analyze"}:
        spike_summary = None
        summary_path = args.spike_summary_path or writer.dimension_summary_path
        if os.path.exists(summary_path):
            spike_summary = load_spike_summary(summary_path)
        outputs = pipeline.detect_tokens(
            _iter_trace_examples(args, adapter, with_attentions=True),
            dimension_summary=spike_summary,
        )
        print(f"Saved sink records to {outputs['sink_records_path']}")
        print(f"Saved sink attention records to {outputs['sink_attention_records_path']}")
        if outputs["dataset_attention_heatmap_path"] is not None:
            print(f"Saved dataset attention heatmap to {outputs['dataset_attention_heatmap_path']}")
        return

    sink_dims = config.explicit_sink_dims or []
    summary_path = args.spike_summary_path or writer.dimension_summary_path
    if not sink_dims and os.path.exists(summary_path):
        sink_dims = load_spike_summary(summary_path).selected_sink_dims
    meta_path = _run_pruning(args, adapter, config, sink_dims)
    print(f"Saved pruning metadata to {meta_path}")


if __name__ == "__main__":
    main()
