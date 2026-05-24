import sys
sys.path.append("./")
sys.path.append("./utils")
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

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

SMALL_DATASETS=["pubmed", "cora"]


class MP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


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

def eval_model(args):
    # Model
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loaded from {model_path}. Model Base: {args.model_base}")
    tokenizer, model, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                          cache_dir=args.cache_dir)
    model = model.to(torch.float16).cuda()
    # data_dir=os.path.expanduser(args.data_dir)
    if args.dataset == "arxiv":
        data_dir = "dataset/ogbn-arxiv"
    elif args.dataset == "products":
        data_dir = "dataset/ogbn-products"
    elif args.dataset == "pubmed":
        data_dir = "dataset/pubmed"
    elif args.dataset == "cora":
        data_dir = "dataset/cora"
    else:
        print(f"{args.dataset} not exists")
        raise ValueError
    if args.task in  ["nc", "nd", "nda", "nctext"]:
        if args.template == "HO":
            prompt_file = os.path.join(data_dir, f"sampled_2_10_test.jsonl")
        else:
            prompt_file = os.path.join(data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_test.jsonl")
        data_path = os.path.join(data_dir, f"processed_data.pt")
    elif args.task in ["lp"]:
        if args.template == "HO":
            prompt_file = os.path.join(data_dir, f"edge_sampled_2_10_only_test.jsonl")
        else:
            prompt_file = os.path.join(data_dir, f"edge_sampled_{args.use_hop}_{args.sample_neighbor_size}_only_test.jsonl")
        data_path = os.path.join(data_dir, f"processed_data.pt")
    else:
        raise ValueError

    data = torch.load(data_path, weights_only=False)
    print(f"Load from {prompt_file}\n")
    lines = open(prompt_file, "r").readlines()

    if args.start >= 0:
        if args.end < 0:
            args.end = len(lines)
        lines = lines[args.start:args.end]
    elif args.end > 0:
        lines = lines[:args.end]

    base_answers_file = os.path.expanduser(args.answers_file)
    answers_file = base_answers_file
    rename_answers_file_to = None
    # run_fixed_nonsink_dims = None
    # if args.dim_zeroout_num > 0:
    #     answers_file = add_output_suffix(
    #         base_answers_file,
    #         f"_zeroout_{args.dim_zeroout_target}dim_"
    #         f"{args.dim_zeroout_num}_"
    #         f"seed_{args.dim_zeroout_seed}",
    #     )
    #     args.answers_file = answers_file
    #     print(
    #         f"Dimension zero-out enabled ({args.dim_zeroout_target}). "
    #         f"Saving answers to {answers_file}"
    #     )
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # if "tmp" not in args.answers_file and os.path.exists(answers_file):
    #     line_number = len(open(answers_file, 'r').readlines())
    #     print(f"{args.answers_file} already exists! it has {line_number} lines!!")
        # first_postpad_records_path = f"analysis/{args.dataset}_{args.template}/first_postpad_records.jsonl"
        # if os.path.exists(first_postpad_records_path):
        #     stats = summarize_jsonl(
        #         records_path=first_postpad_records_path,
        #         edge_index=data.edge_index,  # optional, for hop category
        #     )
        #     print("\nFirst-post-pad attention summary")
        #     print(f"  Samples: {stats['num_samples']}")
        #     print(f"  Valid samples: {stats['num_valid_samples']}")
        #     print(f"  Highest cases (post-pad target): {stats['num_highest_cases']}")
        #     print(
        #         "  Pct post-pad target is highest "
        #         f"(all / valid): {stats['pct_target_is_highest_over_all_samples']:.2f}% / "
        #         f"{stats['pct_target_is_highest_over_valid_samples']:.2f}%"
        #     )
        #     print(
        #         "  Hop percentages among post-pad highest cases: "
        #         f"center={stats['hop_percentages_among_highest']['center']:.2f}%, "
        #         f"one_hop={stats['hop_percentages_among_highest']['one_hop']:.2f}%, "
        #         f"two_hop={stats['hop_percentages_among_highest']['two_hop']:.2f}%, "
        #         f"rest={stats['hop_percentages_among_highest']['rest']:.2f}%"
        #     )
        #     print(
        #         "  12th token highest "
        #         f"(all / samples-with-12th): {stats['pct_twelfth_token_is_highest_over_all_samples']:.2f}% / "
        #         f"{stats['pct_twelfth_token_is_highest_over_samples_with_twelfth_token']:.2f}%"
        #     )
        #     print(
        #         "  12th token matches post-pad target "
        #         f"(all / samples-with-12th): {stats['pct_twelfth_token_matches_target_over_all_samples']:.2f}% / "
        #         f"{stats['pct_twelfth_token_matches_target_over_samples_with_twelfth_token']:.2f}%"
        #     )
    #     if line_number >= len(lines):
    #         return
    #     lines = lines[line_number:]
    #     ans_file = open(answers_file, "a")
    # else:
    #     ans_file = open(answers_file, "w")

    ans_file = open(answers_file, "w")

    questions = [json.loads(q) for q in lines]
    #questions = questions[:500]

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

    # Initialization Steps Occur Here
    # n_bins = 50
    # agg_state = init_graph_agg_state(Lg_max=256, n_bins=n_bins, device=torch.device("cuda"))
    # activation_agg_state = None
    # activation_topdims_state = None
    sink_activation_agg_state = None
    all_sample_layer_query_to_graph = []
    pregraph_activation_agg_state = None  # for activation probing on pre-graph tokens
    all_sample_generated_key_is_pad = []

    # Main Loop
    for line in tqdm(questions):
        idx = line["id"]
        if args.task in ["nd", "nda"]:
            qs=f"Please briefly describe the center node of {DEFAULT_GRAPH_TOKEN}."
        elif args.task == "nc":
            if args.dataset == "products":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent products sold in Amazon, and edges between products indicate they are purchased together. We need to classify the center node into 47 classes: Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, Baby Products, label 25, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & D&#233;cor, #508510, please tell me which class the center node belongs to?"
            else:
                qs = line["conversations"][0]['value']
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
                for _ in range(args.use_hop):
                    local_x = mp.propagate(edge_index, x=local_x, norm=norm)
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

        try:
            with torch.inference_mode():    # when calling model.generate, HF repeatedly calls `prepare_inputs_for_generation()` to build the next-step inputs
                output_ids = model.generate(
                    input_ids,
                    graph_emb=graph_emb.half().cuda(),
                    graph=graph.cuda(),
                    output_hidden_states=True,    # Hidden states from all layers
                    output_attentions=True,       # Attention weights from all layers
                    return_dict_in_generate=True, # Return a dict with all outputs
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            pruned_generate_outputs = output_ids
            dim_pruning_info = None
            activation_analysis = None

            # activation_analysis = compute_layerwise_graph_token_hidden_states(
            #     generate_outputs=output_ids,
            #     input_ids=input_ids,
            #     attention_mask=torch.ones_like(input_ids),
            #     graphs=graph.cuda(),
            # )

            if activation_analysis:
                layer_hidden = activation_analysis["layer_graph_hidden_states"][0]
                key_is_pad = activation_analysis["key_is_pad"][0]
                key_idx = activation_analysis["key_idx"][0]
                
                # Locate per-layer sink dimension
                spike_dims_per_layer = find_spike_dims(
                    hidden_states=layer_hidden,
                    key_is_pad=key_is_pad,
                    k=args.dim_zeroout_num,
                )
                #print("Sink Dimension Number:", spike_dims_per_layer)

                # Sink pruning
                if args.dim_zeroout_target == "sink":
                    dim_pruning_info = sink_dim_pruning(
                        model=model,
                        spike_dims_per_layer=spike_dims_per_layer,
                        graph_token_positions=key_idx,
                    )
                    dim_pruning_info["target"] = "sink"
                else:
                    dim_pruning_info = nonsink_dim_pruning(
                        model=model,
                        spike_dims_per_layer=spike_dims_per_layer,
                        hidden_dim=layer_hidden.shape[-1],
                        graph_token_positions=key_idx,
                        seed=args.dim_zeroout_seed,
                    )
                    dim_pruning_info["target"] = "nonsink"

                dim_pruning_info["num_zeroout_per_layer"] = args.dim_zeroout_num
                dim_pruning_info["selected_dims"] = dim_pruning_info["selected_dims_per_layer"]

                # try:
                #     with torch.inference_mode():
                #         pruned_generate_outputs = model.generate(
                #             input_ids,
                #             graph_emb=graph_emb.half().cuda(),
                #             graph=graph.cuda(),
                #             output_hidden_states=False,
                #             output_attentions=False,
                #             return_dict_in_generate=True,
                #             do_sample=True,
                #             temperature=args.temperature,
                #             top_p=args.top_p,
                #             num_beams=args.num_beams,
                #             max_new_tokens=1024,
                #             use_cache=True,
                #         )
                # finally:
                #     remove_dim_pruning(dim_pruning_info["handles"])

            # The outputs, a tuple of three entries:
            # First: output_ids
            # Second: attention scores
            #  - # of generation steps
            #    - # of attention layers, each with shape:
            #      - [batch size, num attention heads, query tokens, key tokens]
            # Third: activation values
            #  - # of generation steps
            #    - # of attention layers + input embeddings in front
            #      - [batch size, key tokens, dimension size]

            # analysis = analyze_generation_attention_llaga(
            #     generate_outputs=output_ids,                 # returned dict-like object from generate()
            #     input_ids=input_ids,                         # [1, T_text]
            #     attention_mask=torch.ones_like(input_ids),    # if you do not have one, this is fine for your current prompts
            #     graphs=graph.cuda(),                          # [G, Lg] where G is # of <graph> placeholders
            #     mm_use_graph_special_token=getattr(model.config, "mm_use_graph_special_token", False),
            #     use_hop=getattr(model.config, "use_hop", None),
            #     sample_neighbor_size=getattr(model.config, "sample_neighbor_size", None),
            # )

            # if analysis["valid"][0]:
            #     # graphs is [G,Lg]; importance_all is [G*Lg]
            #     imp_all = analysis["graph_importance_all"][0]
            #     agg_state = update_graph_agg_state(
            #         state=agg_state,
            #         graph_importance_all=imp_all,
            #         graphs=graph.cuda(),
            #         n_bins=n_bins,
            #     )

            # Calculate Attention Scores

            # analysis = analyze_and_plot_sample_attention_layeravg(
            #     generate_outputs=output_ids,
            #     input_ids=input_ids,
            #     attention_mask=torch.ones_like(input_ids),
            #     graphs=graph.cuda(),
            #     sample_id=str(idx),
            #     save_dir=f"analysis/{args.dataset}_{args.template}/attention_probes_avg_padded",
            #     plotting=False
            # )

            # # Remap to graph
            # layer_qk = analysis["layer_query_to_graph"][0]   # [L,Q,K]
            # key_is_pad = analysis["key_is_pad"][0]           # [K]
            # all_sample_layer_query_to_graph.append(layer_qk.detach().cpu())

            # # graph: same graph tensor used in generate (shape [G,Lg])
            # out = remap_and_plot_node_attention(
            #     layer_query_to_graph=layer_qk,
            #     graphs=graph,
            #     key_is_pad=key_is_pad,
            #     edge_index=data.edge_index,  # optional
            #     save_path=f"analysis/{args.dataset}_{args.template}/node_remap/sample_{idx}.png",
            # )

            # rec = build_first_postpad_sample_record(
            #     sample_id=idx,
            #     layer_query_to_graph=layer_qk,  # [L,Q,K]
            #     graphs=graph,  # [G,Lg]
            #     key_is_pad=key_is_pad,
            #     graph_emb=graph_emb,
            # )
            # append_first_postpad_sample_record(
            #     record=rec,
            #     save_path=f"analysis/{args.dataset}_{args.template}/first_postpad_records.jsonl",
            # )

            # Activation Probing Analysis: sink token vs rest_graph vs pad tokens
            # if activation_analysis is None:
            #     activation_analysis = compute_layerwise_graph_token_hidden_states(
            #         generate_outputs=output_ids,
            #         input_ids=input_ids,
            #         attention_mask=torch.ones_like(input_ids),
            #         graphs=graph.cuda(),
            #     )

            # if activation_analysis["valid"][0]:
            #     layer_hidden = activation_analysis["layer_graph_hidden_states"][0]
            #     key_is_pad = activation_analysis["key_is_pad"][0]
            #     if sink_activation_agg_state is None:
            #         sink_activation_agg_state = init_sink_activation_agg_state(
            #             num_layers=layer_hidden.shape[0],
            #             hidden_dim=layer_hidden.shape[2],
            #         )
            #     sink_activation_agg_state = update_sink_activation_agg_state(
            #         state=sink_activation_agg_state,
            #         layer_graph_hidden_states=layer_hidden,
            #         key_is_pad=key_is_pad,
            #     )

            # Second Activation Analysis
            # pregraph_analysis = compute_layerwise_pregraph_token_hidden_states(
            #     generate_outputs=output_ids,
            #     input_ids=input_ids,
            #     attention_mask=torch.ones_like(input_ids),
            #     graphs=graph.cuda(),
            # )

            # layer_first_hidden = pregraph_analysis["layer_first_token_hidden_states"][0]
            # layer_pregraph_hidden = pregraph_analysis["layer_pregraph_hidden_states"][0]

            # if pregraph_activation_agg_state is None:
            #     pregraph_activation_agg_state = init_pregraph_activation_agg_state(
            #         num_layers=layer_first_hidden.shape[0],
            #         hidden_dim=layer_first_hidden.shape[1],
            #     )

            # pregraph_activation_agg_state = update_pregraph_activation_agg_state(
            #     state=pregraph_activation_agg_state,
            #     layer_first_token_hidden_states=layer_first_hidden,
            #     layer_pregraph_hidden_states=layer_pregraph_hidden,
            # )


            ##### Attention Weight analysis on Generated Texts

            # gen_attention_analysis = compute_layerwise_generated_query_to_graph_attention(
            #     generate_outputs=output_ids,
            #     input_ids=input_ids,
            #     attention_mask=torch.ones_like(input_ids),
            #     graphs=graph.cuda(),
            #     keep_pad_tokens=True,
            #     mm_use_graph_special_token=getattr(model.config, "mm_use_graph_special_token", False),
            #     use_hop=getattr(model.config, "use_hop", None),
            #     sample_neighbor_size=getattr(model.config, "sample_neighbor_size", None),
            # )

            # if gen_attention_analysis["valid"][0]:
            #     layer_qk = gen_attention_analysis["layer_generated_query_to_graph"][0]   # [L,S,K]
            #     key_is_pad = gen_attention_analysis["key_is_pad"][0]                     # [K]
            #     key_idx = gen_attention_analysis["key_idx"][0]                           # [K]

            #     all_sample_layer_query_to_graph.append(layer_qk.detach().cpu())
            #     all_sample_generated_key_is_pad.append(key_is_pad.detach().cpu())

                # Optional: save the same JSONL record format as before.
                # This works because build_first_postpad_sample_record accepts [L,Q,K],
                # and here S plays the role of Q.
                # rec = build_first_postpad_sample_record(
                #     sample_id=idx,
                #     layer_query_to_graph=layer_qk,   # [L,S,K]
                #     graphs=graph,                    # [G,Lg]
                #     key_is_pad=key_is_pad,
                #     graph_emb=graph_emb,
                # )
                # append_first_postpad_sample_record(
                #     record=rec,
                #     save_path=f"analysis/{args.dataset}_{args.template}/first_postpad_records_generated.jsonl",
                # )

            generated_sequences = pruned_generate_outputs.sequences
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
            outputs=""
            dim_pruning_info = None

        ans_id = shortuuid.uuid()
        answer_record = {
            "question_id": idx,
            "prompt": cur_prompt,
            "graph": line['graph'],
            "text": outputs,
            "gt":line["conversations"][1]['value'],
            "answer_id": ans_id,
        }
        # if dim_pruning_info is not None:
        #     answer_record["dim_zeroout_num"] = int(args.dim_zeroout_num)
        #     answer_record["dim_zeroout_target"] = dim_pruning_info["target"]
        #     answer_record["dim_zeroout_num_candidates"] = int(dim_pruning_info["num_candidates"])
        #     answer_record["dim_zeroout_num_sink_dims"] = int(dim_pruning_info["num_sink_dims"])
        #     answer_record["dim_zeroout_selected_dims"] = dim_pruning_info["selected_dims"]
        ans_file.write(json.dumps(answer_record) + "\n")
        ans_file.flush()
        # first_postpad_records_path = f"analysis/{args.dataset}_{args.template}/first_postpad_records.jsonl"
    
        # Plot aggregated attention value
        # agg = plot_final_average_cross_attention_for_dataset(
        #     sample_layer_query_to_graph=all_sample_layer_query_to_graph,  # list of [L,Q,K]
        #     save_path=f"analysis/{args.dataset}_{args.template}/dataset_avg_cross_attention_generated_tokens.png",
        # )
        # generated_attn_agg = aggregate_generated_attention_by_graph_position(
        #     sample_layer_generated_query_to_graph=all_sample_layer_query_to_graph,
        #     sample_key_is_pad=all_sample_generated_key_is_pad,
        # )
        # generated_attn_plot_path = plot_generated_attention_histogram(
        #     aggregated=generated_attn_agg,
        #     save_path=f"analysis/{args.dataset}_{args.template}/dataset_avg_generated_attention_histogram.png",
        #     pad_threshold=0.5,
        # )
    #     stats = summarize_jsonl(
    #         records_path=first_postpad_records_path,
    #         edge_index=data.edge_index,  # optional, for hop category
    #     )
    #     print("\nFirst-post-pad attention summary")
    #     print(f"  Samples: {stats['num_samples']}")
    #     print(f"  Valid samples: {stats['num_valid_samples']}")
    #     print(f"  Highest cases (post-pad target): {stats['num_highest_cases']}")
    #     print(
    #         "  Pct post-pad target is highest "
    #         f"(all / valid): {stats['pct_target_is_highest_over_all_samples']:.2f}% / "
    #         f"{stats['pct_target_is_highest_over_valid_samples']:.2f}%"
    #     )
    #     print(
    #         "  Hop percentages among post-pad highest cases: "
    #         f"center={stats['hop_percentages_among_highest']['center']:.2f}%, "
    #         f"one_hop={stats['hop_percentages_among_highest']['one_hop']:.2f}%, "
    #         f"two_hop={stats['hop_percentages_among_highest']['two_hop']:.2f}%, "
    #         f"rest={stats['hop_percentages_among_highest']['rest']:.2f}%"
    #     )
    #     print(
    #         "  12th token highest "
    #         f"(all / samples-with-12th): {stats['pct_twelfth_token_is_highest_over_all_samples']:.2f}% / "
    #         f"{stats['pct_twelfth_token_is_highest_over_samples_with_twelfth_token']:.2f}%"
    #     )
    #     print(
    #         "  12th token matches post-pad target "
    #         f"(all / samples-with-12th): {stats['pct_twelfth_token_matches_target_over_all_samples']:.2f}% / "
    #         f"{stats['pct_twelfth_token_matches_target_over_samples_with_twelfth_token']:.2f}%"
    #     )
    #     print(
    #         "  Avg cosine(post-pad token, center token): "
    #         f"{stats['avg_postpad_center_cosine_similarity']:.4f} "
    #         f"(valid cosine samples={stats['num_postpad_center_cosine_valid_samples']})"
    #     )
    # if activation_agg_state is not None:
    #     activation_stats = finalize_activation_agg_state(activation_agg_state)
    #     activation_plot_paths = plot_aggregated_activation_curves(
    #         aggregated=activation_stats,
    #         save_dir=f"analysis/{args.dataset}_{args.template}/activation_probes_aggregated",
    #     )
    #     print("\nActivation probe summary")
    #     print(f"  Valid activation samples: {activation_stats['num_valid_samples']}")
    #     print(f"  Saved aggregated activation plots: {len(activation_plot_paths)}")

    # if activation_topdims_state is not None:
    #     activation_topdims_stats = finalize_activation_topdims_agg_state(
    #         activation_topdims_state,
    #         topk=5,
    #     )
    #     activation_topdims_path = save_activation_topdims_summary(
    #         aggregated=activation_topdims_stats,
    #         save_path=f"analysis/{args.dataset}_{args.template}/activation_topdims.json",
    #         dataset_name=args.dataset,
    #         view_name=args.template,
    #     )
    #     print("\nActivation probe summary")
    #     print(f"  Valid activation samples: {activation_topdims_stats['num_valid_samples']}")
    #     print(f"  Saved top-dimension summary: {activation_topdims_path}")
    #     if activation_topdims_stats["top_dims_by_layer"]:
    #         print(f"  Layer 0 top-5 dims: {activation_topdims_stats['top_dims_by_layer'][0]}")

    # if sink_activation_agg_state is not None:
    #     sink_activation_stats = finalize_sink_activation_agg_state(sink_activation_agg_state)
    #     sink_activation_plot_paths = plot_sink_vs_rest_activation_curves(
    #         aggregated=sink_activation_stats,
    #         save_dir=f"analysis/{args.dataset}_{args.template}/sink_activation_probes_aggregated",
    #     )
    #     print("\nActivation probe summary")
    #     print(f"  Valid sink-activation samples: {sink_activation_stats['num_valid_samples']}")
    #     print(f"  Saved sink/rest_graph/pad activation plots: {len(sink_activation_plot_paths)}")


    # Plot pre-graph activation plots
    if pregraph_activation_agg_state is not None:
        pregraph_activation_stats = finalize_pregraph_activation_agg_state(
            pregraph_activation_agg_state
        )
        pregraph_activation_plot_paths = plot_pregraph_activation_curves(
            aggregated=pregraph_activation_stats,
            save_dir=f"analysis/{args.dataset}_{args.template}/pregraph_activation_probes_aggregated",
        )
    ans_file.close()
    if rename_answers_file_to is not None and answers_file != rename_answers_file_to:
        os.replace(answers_file, rename_answers_file_to)
        print(f"Renamed answers file to {rename_answers_file_to}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    # parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--pretrained_embedding_type", type=str, default="sbert")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=5)
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--mm_use_graph_start_end",default=False, action="store_true")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--template", type=str, default="ND")
    args = parser.parse_args()

    eval_model(args)
