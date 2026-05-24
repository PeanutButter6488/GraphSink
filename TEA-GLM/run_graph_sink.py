from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from tqdm import tqdm
from transformers import LlamaTokenizer


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
WORKSPACE_ROOT = REPO_ROOT.parent
UTILS_DIR = REPO_ROOT / "utils"
for path in (str(REPO_ROOT), str(UTILS_DIR), str(WORKSPACE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import Config  # noqa: E402
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
from activation import compute_layerwise_graph_token_hidden_states  # noqa: E402


class TeaGlmAdapter(GraphModelAdapter):
    def __init__(
        self,
        artifact_root: str,
        dataset_tag: str,
        num_tokens: int,
        sink_dims: Optional[List[int]] = None,
        threshold: float = 15.0,
    ):
        self._artifact_root = artifact_root
        self._dataset_tag = dataset_tag
        self._num_tokens = num_tokens
        self._sink_dims = sink_dims if sink_dims is not None else [1512, 2533, 3431]
        self._threshold = threshold

    def trace_batch(
        self,
        *,
        sample_ids: List[int],
        outputs,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        is_node: torch.Tensor,
        tokenizer: Optional[LlamaTokenizer] = None,
        source_row_indices: Optional[List[int]] = None,
        with_attentions: bool,
    ) -> List[TraceExample]:
        activation = compute_layerwise_graph_token_hidden_states(
            hidden_states=outputs.hidden_states,
            is_node=is_node,
            expected_k=self._num_tokens,
        )

        examples: List[TraceExample] = []
        for batch_idx, sample_id in enumerate(sample_ids):
            if not activation["valid"][batch_idx]:
                continue

            layer_qk = None
            query_pos = None
            if with_attentions and tokenizer is not None:
                key_pos = activation["graph_token_positions"][batch_idx].detach().cpu().to(torch.long)
                valid = attention_mask[batch_idx].bool()
                query_mask = valid.clone()
                if key_pos.numel() > 0:
                    query_mask[: int(key_pos[-1].item()) + 1] = False
                query_pos = torch.nonzero(query_mask, as_tuple=False).reshape(-1).cpu()
                if query_pos.numel() > 0 and outputs.attentions is not None:
                    layer_mats = []
                    for layer_attn in outputs.attentions:
                        sub = (
                            layer_attn[batch_idx]
                            .detach()
                            .to(torch.float32)
                            .index_select(1, query_pos.to(layer_attn.device))
                            .index_select(2, key_pos.to(layer_attn.device))
                            .mean(dim=0)
                            .cpu()
                        )
                        layer_mats.append(sub)
                    layer_qk = torch.stack(layer_mats, dim=0)

            graph_positions = activation["graph_token_positions"][batch_idx].detach().cpu()
            graph_pad = torch.zeros(graph_positions.numel(), dtype=torch.bool)
            metadata = {"source_row_index": int((source_row_indices or sample_ids)[batch_idx])}
            examples.append(
                TraceExample(
                    sample_id=sample_id,
                    layer_graph_hidden_states=activation["layer_graph_hidden_states"][batch_idx].detach().cpu(),
                    graph_token_positions=graph_positions,
                    graph_token_is_pad=graph_pad,
                    layer_query_to_graph_attention=layer_qk,
                    query_token_positions=query_pos,
                    metadata=metadata,
                )
            )
        return examples

    def get_decoder_layers(self, model) -> List[torch.nn.Module]:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
            return list(model.model.decoder.layers)
        raise ValueError("Could not locate decoder layers on the TEA-GLM model.")

    def default_sink_config(self) -> SinkConfig:
        return SinkConfig(
            model_name="TEA-GLM",
            dataset_tag=self._dataset_tag,
            hidden_dim=4096,
            expected_graph_tokens=self._num_tokens,
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


def _to_config(args: argparse.Namespace) -> Config:
    return Config(
        seed=args.seed,
        dataset=args.dataset,
        test_dataset=args.test_dataset,
        project=args.project,
        exp_num=1,
        backbone=args.backbone,
        lora_weights="",
        pretrain_gnn=args.pretrain_gnn,
        graph_pooling=args.graph_pooling,
        prefix=args.prefix,
        suffix=None,
        config_class="LlamaConfig",
        model_class="InstructGLM",
        gt_layers=args.gt_layers,
        num_token=args.num_token,
        head=args.head,
        att_d_model=args.att_d_model,
        gnn_output=args.gnn_output,
        max_text_length=args.max_text_length,
        batch_size=args.batch_size,
        freeze_llama=args.freeze_llama,
        optim="adamw",
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        clip_grad_norm=1.0,
        grad_steps=1,
        lr=1e-3,
        adam_eps=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        epoch=1,
        dropout=args.dropout,
        inference=True,
        best_epoch=args.best_epoch,
        gen_max_length=args.gen_max_length,
        prune_sink_tokens=False,
        sink_record_path="",
        run_perturbation_analysis=False,
    )


def _load_tea_context(args: argparse.Namespace):
    from utils.trainer_base import TrainerBase

    config = _to_config(args)
    tokenizer = LlamaTokenizer.from_pretrained(config.backbone)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_special_tokens({"additional_special_tokens": [f"<Node {i}>" for i in range(1, 110)]})

    trainer = TrainerBase(config, torch.cuda.current_device())
    loader = trainer.create_dataloader(tokenizer)
    first_model, model = trainer.create_model()

    first_model_path = os.path.join(str(REPO_ROOT), "saved_model", "first_model", f"{config.prefix}_fm_{config.dataset}_epoch{config.best_epoch}_end.pth")
    model_path = os.path.join(str(REPO_ROOT), "saved_model", "model", f"{config.prefix}_m_{config.dataset}_epoch{config.best_epoch}_end.pth")

    first_model.load_state_dict(torch.load(first_model_path, map_location="cpu"), strict=False)
    if not config.freeze_llama:
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)

    first_model = first_model.cuda().eval()
    model = model.cuda().eval()
    return tokenizer, loader, first_model, model


def _iter_trace_examples(
    args: argparse.Namespace,
    adapter: TeaGlmAdapter,
    *,
    with_attentions: bool,
) -> Iterable[TraceExample]:
    tokenizer, loader, first_model, model = _load_tea_context(args)
    sample_offset = 0

    for batch in tqdm(loader, desc=args.stage):
        input_ids = batch["input_ids"].cuda()
        is_node = batch["is_node"].cuda()
        attention_mask = batch["attn_mask"].cuda()
        graph = batch["graph"].to(input_ids.device)

        with torch.no_grad():
            embeds = first_model(input_ids=input_ids, is_node=is_node, graph=graph)
            outputs = model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=with_attentions,
                return_dict=True,
            )

        batch_size = input_ids.shape[0]
        sample_ids = list(range(sample_offset, sample_offset + batch_size))
        examples = adapter.trace_batch(
            sample_ids=sample_ids,
            outputs=outputs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            is_node=is_node,
            tokenizer=tokenizer,
            source_row_indices=sample_ids,
            with_attentions=with_attentions,
        )
        sample_offset += batch_size
        for example in examples:
            yield example


def _run_pruning(args: argparse.Namespace, adapter: TeaGlmAdapter, config: SinkConfig, sink_dims: Sequence[int]) -> str:
    writer = ArtifactWriter(
        config.artifact_root,
        model_name=config.model_name,
        dataset_tag=config.dataset_tag,
    )
    records_path = args.sink_records_path or writer.sink_records_path
    record_index = load_sink_record_index(records_path)
    tokenizer, loader, first_model, model = _load_tea_context(args)
    experiment = PruningExperiment(adapter.get_decoder_layers(model))
    spec = PruningSpec(mode=args.prune_mode, selection=args.selection, k=args.k, seed=args.seed)
    meta_rows: List[Dict[str, object]] = []
    sample_offset = 0

    for batch in tqdm(loader, desc="prune"):
        input_ids = batch["input_ids"].cuda()
        is_node = batch["is_node"].cuda()
        attention_mask = batch["attn_mask"].cuda()
        graph = batch["graph"].to(input_ids.device)
        batch_size = input_ids.shape[0]
        sample_ids = list(range(sample_offset, sample_offset + batch_size))
        batch_plan = []

        for local_idx, sample_id in enumerate(sample_ids):
            record = record_index["by_sample_id"].get(str(sample_id))
            if record is None:
                record = record_index["by_source_row_index"].get(str(sample_id))
            if record is None:
                batch_plan.append(
                    {
                        "sample_id": sample_id,
                        "mode": spec.mode,
                        "selection": spec.selection,
                        "k": spec.k,
                        "seed": spec.seed,
                        "selected_token_positions": [],
                        "selected_dim_ids": [],
                        "sink_prompt_positions": [],
                        "graph_token_positions": [],
                    }
                )
                meta_rows.append({"sample_id": sample_id, "source_row_index": sample_id, "skipped": "missing_record"})
                continue

            plan = experiment.build_plan(
                sink_record=record,
                spec=spec,
                sink_dims=sink_dims,
                hidden_dim=config.hidden_dim,
            )
            batch_plan.append(plan)
            meta_rows.append({"sample_id": sample_id, "source_row_index": sample_id, **plan})

        with torch.no_grad():
            embeds = first_model(input_ids=input_ids, is_node=is_node, graph=graph)
            handles = experiment.attach(batch_plan)
            try:
                model.g_step(in_embeds=embeds, attention_mask=attention_mask)
            finally:
                experiment.remove(handles)

        sample_offset += batch_size

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
    parser.add_argument("--dataset", type=str, default="pubmed")
    parser.add_argument("--test_dataset", type=str, default="pubmed")
    parser.add_argument("--project", type=str, default="project_GraphLLM")
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--pretrain_gnn", type=str, required=True)
    parser.add_argument("--graph_pooling", type=str, default="sum")
    parser.add_argument("--prefix", type=str, default="trainable_llama_gnn")
    parser.add_argument("--gt_layers", type=int, default=2)
    parser.add_argument("--num_token", type=int, default=5)
    parser.add_argument("--head", type=int, default=2)
    parser.add_argument("--att_d_model", type=int, default=2048)
    parser.add_argument("--gnn_output", type=int, default=4096)
    parser.add_argument("--max_text_length", type=int, default=2215)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--freeze_llama", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--best_epoch", type=int, default=0)
    parser.add_argument("--gen_max_length", type=int, default=64)
    parser.add_argument("--run_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--sink_dims", type=str, default=None)
    parser.add_argument("--auto_sink_dims", action="store_true")
    parser.add_argument("--num_sink_dims", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=15.0)
    parser.add_argument("--activation_topk_per_layer", type=int, default=5)
    parser.add_argument("--spike_summary_path", type=str, default=None)
    parser.add_argument("--sink_records_path", type=str, default=None)
    parser.add_argument("--prune_mode", choices=["sink_token", "control_nonsink_token", "sink_dim", "control_nonsink_dim"], default="sink_token")
    parser.add_argument("--selection", choices=["all", "topk"], default="all")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    output_root = args.output_root or args.run_root or _default_output_root()
    dataset_tag = args.test_dataset

    adapter = TeaGlmAdapter(
        artifact_root=output_root,
        dataset_tag=dataset_tag,
        num_tokens=args.num_token,
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
