import argparse
import pprint
import yaml
from os import path

module_path = path.dirname(path.abspath(__file__))


def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument("--dataset", type=str, default='HIV')
    parser.add_argument("--test_dataset", type=str, default='pubmed')
    parser.add_argument("--task", type=str, default='nc', choices=['nc', 'lp'],
                        help="Inference task. 'nc' loads {ds}_dataset_{mode}.json; "
                             "'lp' loads {ds}_LP_dataset_{mode}.json and tags output paths "
                             "with _lp so NC and LP runs do not overwrite each other.")
    parser.add_argument("--project", type=str, default="project_GraphLLM")
    parser.add_argument("--exp_num", default=1)

    # Model Config
    parser.add_argument('--backbone', type=str, default='llama-v1-7b')
    parser.add_argument('--lora_weights', type=str, default='')
    parser.add_argument('--pretrain_gnn', type=str, default='')
    parser.add_argument('--graph_pooling', type=str, default='sum')
    parser.add_argument('--prefix', type=str, default='trainable_llama_gnn')
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--config_class', type=str, default='LlamaConfig')
    parser.add_argument('--model_class', type=str, default='InstructGLM')
    parser.add_argument('--gt_layers', type=int, default=2)
    parser.add_argument('--num_token', type=int, default=5)
    parser.add_argument('--head', type=int, default=2)
    parser.add_argument('--att_d_model', type=int, default=2048)
    parser.add_argument('--gnn_output', type=int, default=4096)
    parser.add_argument('--max_text_length', type=int, default=2215)

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--freeze_llama', action='store_true')
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--weight_decay', type=float, default=0.0)   
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default='cosine')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument("--grad_steps", type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--best_epoch', type=int, default=0)

    # Inference
    parser.add_argument('--gen_max_length', type=int, default=64)
    parser.add_argument('--prune_sink_tokens', action='store_true',
                        help='Prune graph sink tokens during inference using saved sink-token records')
    parser.add_argument('--sink_record_path', type=str, default='',
                        help='Optional path to a sink-record JSONL file used for inference-time pruning')
    parser.add_argument('--pruning_mode', type=str, default='top2',
                        choices=['top2', 'all', 'random'],
                        help='top2: prune top-2 sink tokens; all: prune every detected sink; '
                             'random: prune num_prune random non-sink graph tokens (seedable)')
    parser.add_argument('--num_prune', type=int, default=2,
                        help='Number of non-sink graph tokens to prune when --pruning_mode=random')
    parser.add_argument('--append_seed_suffix', action='store_true',
                        help='Append _seed{args.seed} to baseline result filenames so vanilla '
                             'multi-seed sweeps do not overwrite each other. No effect when '
                             '--prune_sink_tokens is set (pruning already encodes the seed).')
    parser.add_argument('--sink_reoccur', action='store_true',
                        help='After pruning all sinks (--pruning_mode=all), run an extra forward '
                             'pass on the pruned sequence to detect whether new sink tokens '
                             're-emerge among the remaining graph tokens. Writes a '
                             '{prefix}{out_suffix}_sink_reoccur_distribution.png histogram over '
                             'original K-space graph-token indices.')
    parser.add_argument('--reposition_mode', type=str, default='none',
                        choices=['none', 'swap_sink_nonsink'],
                        help='Graph-token shuffling experiment. swap_sink_nonsink randomly '
                             'swaps --num_swap sink positions with --num_swap non-sink positions '
                             'per sample. Mutually exclusive with --prune_sink_tokens.')
    parser.add_argument('--num_swap', type=int, default=2,
                        help='Number of (sink, non-sink) pairs to swap when '
                             '--reposition_mode=swap_sink_nonsink')
    parser.add_argument('--reposition_seed', type=int, default=0,
                        help='Seed for the sink/non-sink swap sampler')
    
    # Analysis
    parser.add_argument('--run_perturbation_analysis', action='store_true',
                        help='Enable neighborhood perturbation analysis during evaluation')
    parser.add_argument('--sink_dim_threshold', type=float, default=5.0,
                        help='Threshold on the per-dim averaged |RMSNorm| (or |signed RMSNorm|) '
                             'used to mark sink dimensions on the topdims activation plot. '
                             'Dims whose curve value exceeds this are labeled in bold-italic '
                             'red on the x-axis.')
    parser.add_argument('--logit_lens', action='store_true',
                        help='Save a logit-lens heatmap (top-1 token per layer x graph token) '
                             'for the first test sample only.')

    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    # For interative engironmnet (ex. jupyter)
    else:
        args = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

    return args


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


if __name__ == '__main__':
    args = parse_args(True)
