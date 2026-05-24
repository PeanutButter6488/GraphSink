import numpy as np
import torch
import json
import random, os
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from torch_geometric.data import Batch, Data
from typing import Dict, Any, Optional, Literal, Sequence, List, Tuple, Union


# Sanity checks and Experiments, organized here (12.15.2025)

def init_dim_stats_storage(hidden_dim=4096):
    """
    Creates storage for running sums to calculate the mean later.
    Memory efficient: Only stores one vector per layer.
    """
    return {
        # Structure: layer_idx -> {'sum_tensor': Tensor[4096], 'count': int}
        2:  {'sum': torch.zeros(hidden_dim), 'count': 0},
        10: {'sum': torch.zeros(hidden_dim), 'count': 0},
        16: {'sum': torch.zeros(hidden_dim), 'count': 0},
        20: {'sum': torch.zeros(hidden_dim), 'count': 0},
        22: {'sum': torch.zeros(hidden_dim), 'count': 0},
        24: {'sum': torch.zeros(hidden_dim), 'count': 0},
        26: {'sum': torch.zeros(hidden_dim), 'count': 0},
        28: {'sum': torch.zeros(hidden_dim), 'count': 0},
        30: {'sum': torch.zeros(hidden_dim), 'count': 0},
        31: {'sum': torch.zeros(hidden_dim), 'count': 0}, # Final Layer
    }

def init_dataset_attn_storage():
    target_layers = [2, 10, 16, 20, 22, 24, 26, 28, 30, 31]
    storage = {}
    for layer in target_layers:
        storage[layer] = {
            'text_max_peaks': [],   
            'text_avg_scores': [],  # <--- NEW: Track Text Mean
            'graph_max_peaks': [],  
            'graph_avg_scores': [], 
            'mass_ratios': []       
        }
    return storage

def collect_dim_stats(storage, hidden_states, is_node):
    """
    Aggregates element-wise sums for graph tokens.
    """
    # 1. Identify Graph Tokens Indices
    # graph_indices = (is_node == 1).nonzero(as_tuple=False)
    # Actually, we can do this faster with masking since we are summing everything
    
    mask = (is_node == 1).unsqueeze(-1) # [Batch, Seq, 1]
    num_graph_tokens = mask.sum().item()

    if num_graph_tokens == 0:
        return

    # 2. Iterate Layers
    target_layers = list(storage.keys())
    
    for layer_idx in target_layers:
        # Get hidden states [Batch, Seq, Dim]
        hidden = hidden_states[layer_idx].detach().cpu()
        
        # Zero out non-graph tokens
        graph_tokens_only = hidden * mask.cpu()
        
        # Sum across Batch and Sequence
        # Result shape: [Dim] (e.g., 4096)
        batch_sum = graph_tokens_only.sum(dim=(0, 1))
        
        # Update Storage
        storage[layer_idx]['sum'] += batch_sum
        storage[layer_idx]['count'] += num_graph_tokens

def plot_average_dimension_activations(storage, base_save_dir):
    """
    Plots the Mean Activation Vector for each layer.
    X-Axis: Dimension (0-4096)
    Y-Axis: Average Activation Value
    """
    save_dir = os.path.join(base_save_dir, "aggregated_dim_analysis")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"\n[Analysis] Generating average activation plots for {len(storage)} layers...")

    for layer_idx, data in storage.items():
        total_sum = data['sum']
        count = data['count']
        
        if count == 0:
            continue
            
        # 1. Calculate Mean Vector
        mean_vector = (total_sum / count).numpy()
        x_axis = np.arange(len(mean_vector))
        
        # 2. Plotting
        plt.figure(figsize=(12, 6))
        
        # Plot the average line
        plt.plot(x_axis, mean_vector, color='blue', linewidth=0.8, alpha=0.9, label='Mean Activation')
        
        # 3. Highlight Systematic Peaks
        # Find the top 3 dimensions that are consistently high on average
        # Use abs() to find magnitude peaks (negative or positive)
        sorted_indices = np.argsort(np.abs(mean_vector))[::-1]
        top_indices = sorted_indices[:3]
        
        for rank, idx in enumerate(top_indices):
            val = mean_vector[idx]
            color = 'red' if rank == 0 else 'orange'
            
            # Draw vertical line
            plt.axvline(x=idx, color=color, linestyle='--', alpha=0.6)
            
            # Add label
            plt.text(idx, val, f" Dim {idx}\n Avg: {val:.2f}", 
                     color=color, fontweight='bold', ha='center', va='bottom', fontsize=9)

        # Labels and Style
        plt.title(f"Layer {layer_idx} - Average Graph Token", fontsize=14)
        plt.xlabel("Hidden Dimension Index (0-4096)")
        plt.ylabel("Average Activation Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"layer_{layer_idx}_avg_dim_profile.png"))
        plt.close()
        
    print(f"[Analysis] Average dimension plots saved to {save_dir}")


def probe_individual_graph_tokens(model, inputs_embeds, attention_mask, is_node, step, base_save_dir):
    """
    Generates an individual activation plot for EVERY single graph token in the batch.
    """
    save_dir = os.path.join(base_save_dir, f"step_{step}_individual_tokens")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. Run Forward Pass
    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
    
    hidden_states = outputs.hidden_states
    
    # 2. Identify "Interesting" Layers to save time (Input, Middle, Output)
    # Change this to 'range(len(hidden_states))' to plot ALL 33 layers
    target_layers = [2, 10]
    
    # 3. Identify Graph Token Indices
    # shape: (Num_Graph_Tokens, 2) -> [[batch_idx, seq_idx], [batch_idx, seq_idx]...]
    graph_indices = (is_node == 1).nonzero(as_tuple=False)
    
    print(f"[Analysis] Found {len(graph_indices)} graph tokens in this batch. Generating individual plots...")

    for layer_idx in target_layers:
        hidden = hidden_states[layer_idx].cpu() # Move to CPU once
        
        # Iterate over every single graph token found in the batch
        for i, (b_idx, s_idx) in enumerate(graph_indices):
            # Extract the vector for this specific token
            # Shape: [4096]
            token_vector = hidden[b_idx, s_idx, :].float().numpy()
            
            # --- Plotting ---
            plt.figure(figsize=(10, 6))
            x_axis = np.arange(len(token_vector))
            
            plt.plot(x_axis, token_vector, color='red', alpha=0.8, linewidth=0.8)
            
            # Title with detailed info
            plt.title(f"Layer {layer_idx} | Batch {b_idx} | Token_Pos {s_idx} (Graph Token #{i})")
            plt.xlabel("Hidden Dimension (0-4096)")
            plt.ylabel("Activation Value")
            plt.grid(True, alpha=0.3)
            
            # --- Auto-Highlight Top Peaks ---
            # Sort to find the massive outliers
            sorted_indices = np.argsort(token_vector)[::-1]
            top_1 = sorted_indices[0]
            top_2 = sorted_indices[1]
            
            # Annotate Top 1
            plt.axvline(x=top_1, color='orange', linestyle='--')
            plt.text(top_1, token_vector[top_1], f" Peak: {token_vector[top_1]:.1f}\n (Dim {top_1})", 
                     color='orange', fontweight='bold')
            
            # Annotate Top 2
            if token_vector[top_2] > 10: # Only if it's significant
                plt.axvline(x=top_2, color='orange', linestyle=':')
                plt.text(top_2, token_vector[top_2], f" 2nd: {token_vector[top_2]:.1f}\n (Dim {top_2})", 
                         color='orange')

            # --- Save ---
            # Filename: layer_31_batch_0_token_sequence_index_15.png
            fname = f"layer_{layer_idx}_b{b_idx}_seq{s_idx}.png"
            plt.savefig(os.path.join(save_dir, fname))
            plt.close()
            
    print(f"[Analysis] Saved individual token plots to {save_dir}")


def collect_dataset_attn_stats(storage, outputs, is_node):
    attentions = outputs.attentions 
    target_layers = list(storage.keys())
    
    for layer_idx in target_layers:
        attn_layer = attentions[layer_idx].detach().cpu()
        batch_size = attn_layer.shape[0]
        
        batch_text_maxes = []
        batch_text_means = [] # <--- NEW
        batch_graph_maxes = []
        batch_graph_means = []
        batch_mass_ratios = []
        
        for b in range(batch_size):
            # Ensure masks are on CPU
            graph_mask = (is_node[b] == 1).cpu()
            text_mask = (is_node[b] == 0).cpu()
            
            # A. Graph Tokens
            if graph_mask.sum() > 0:
                graph_cols = attn_layer[b, :, :, graph_mask] 
                batch_graph_maxes.append(graph_cols.max().item())
                batch_graph_means.append(graph_cols.mean().item())

            # B. Text Tokens
            if text_mask.sum() > 0:
                text_cols = attn_layer[b, :, :, text_mask]
                batch_text_maxes.append(text_cols.max().item())
                batch_text_means.append(text_cols.mean().item()) # <--- NEW
            
            # C. Mass Ratio
            if text_mask.sum() > 0 and graph_mask.sum() > 0:
                text_looking_at_graph = attn_layer[b][:, text_mask, :][:, :, graph_mask]
                ratio = text_looking_at_graph.sum(dim=-1).mean().item()
                batch_mass_ratios.append(ratio)
        
        # Update Global Storage
        if batch_text_maxes: storage[layer_idx]['text_max_peaks'].extend(batch_text_maxes)
        if batch_text_means: storage[layer_idx]['text_avg_scores'].extend(batch_text_means) # <--- NEW
        if batch_graph_maxes: storage[layer_idx]['graph_max_peaks'].extend(batch_graph_maxes)
        if batch_graph_means: storage[layer_idx]['graph_avg_scores'].extend(batch_graph_means)
        if batch_mass_ratios: storage[layer_idx]['mass_ratios'].extend(batch_mass_ratios)


def plot_dataset_attention_summary(storage, base_save_dir):
    save_dir = os.path.join(base_save_dir, "dataset_attention_analysis")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"\n[Analysis] Generating dataset-level attention plots...")

    layers = sorted(storage.keys())
    
    text_max_means = []
    text_avg_means = [] # <--- NEW
    graph_max_means = []
    graph_avg_means = []
    mass_ratio_means = []
    mass_ratio_stds = []

    for layer in layers:
        data = storage[layer]
        
        # Compute Global Averages
        t_max = np.mean(data['text_max_peaks']) if data['text_max_peaks'] else 0
        t_avg = np.mean(data['text_avg_scores']) if data['text_avg_scores'] else 0 # <--- NEW
        g_max = np.mean(data['graph_max_peaks']) if data['graph_max_peaks'] else 0
        g_avg = np.mean(data['graph_avg_scores']) if data['graph_avg_scores'] else 0
        
        m_ratio = np.mean(data['mass_ratios']) if data['mass_ratios'] else 0
        m_std = np.std(data['mass_ratios']) if data['mass_ratios'] else 0
        
        text_max_means.append(t_max)
        text_avg_means.append(t_avg) # <--- NEW
        graph_max_means.append(g_max)
        graph_avg_means.append(g_avg)
        mass_ratio_means.append(m_ratio)
        mass_ratio_stds.append(m_std)

    # --- PLOT 1: Dataset-Wide Competition (Updated with 4 bars) ---
    x = np.arange(len(layers))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Text Bars (Grays)
    rects1 = ax.bar(x - 1.5*width, text_max_means, width, label='Text Max (Peak)', color='dimgray')
    rects2 = ax.bar(x - 0.5*width, text_avg_means, width, label='Text Avg (Baseline)', color='lightgray') # <--- NEW
    
    # Graph Bars (Colors)
    rects3 = ax.bar(x + 0.5*width, graph_max_means, width, label='Graph Max (Peak)', color='orange')
    rects4 = ax.bar(x + 1.5*width, graph_avg_means, width, label='Graph Avg (General)', color='blue')

    ax.set_ylabel('Attention Score')
    ax.set_xlabel('Layer')
    ax.set_title('Attention Score Distributions on Graph & Text')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.savefig(os.path.join(save_dir, "dataset_competition_stats.png"))
    plt.close()

    # --- PLOT 2: Relationship (Same as before) ---
    plt.figure(figsize=(10, 6))
    plt.plot(layers, mass_ratio_means, marker='o', color='green', linewidth=2, label='Mean Mass Ratio')
    plt.fill_between(layers, 
                     np.array(mass_ratio_means) - np.array(mass_ratio_stds),
                     np.array(mass_ratio_means) + np.array(mass_ratio_stds),
                     color='green', alpha=0.2, label='Std Dev')
    
    plt.title("Dataset-Wide Relationship: Text Attention to Graph Tokens")
    plt.xlabel("Layer Index")
    plt.ylabel("Attention Mass Allocation (0.0 - 1.0)")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    for i, val in enumerate(mass_ratio_means):
        plt.text(layers[i], val + 0.05, f"{val:.1%}", ha='center', color='darkgreen', fontweight='bold')

    plt.savefig(os.path.join(save_dir, "dataset_mass_ratio_trend.png"))
    plt.close()
    
    print(f"[Analysis] Dataset plots saved to {save_dir}")


def probe_graph_tokens(model, inputs_embeds, attention_mask, is_node, step, base_save_dir):
    """
    Plots ALL graph tokens for a given layer side-by-side in one horizontal figure.
    CRITICAL: Uses a SHARED Y-AXIS to allow true magnitude comparison.
    """
    # 1. Setup
    save_dir = os.path.join(base_save_dir, f"step_{step}_parallel_comparison")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. Forward Pass
    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )
    hidden_states = outputs.hidden_states
    
    # 3. Layers to analyze
    target_layers = [2, 10, 20, 22, 24, 26, 28, 30, 32] # Add 31 to see the final output
    
    # 4. Identify Graph Tokens
    # graph_indices = [[batch_idx, seq_idx], ...]
    graph_indices = (is_node == 1).nonzero(as_tuple=False)
    num_tokens = len(graph_indices)
    
    print(f"[Analysis] Found {num_tokens} graph tokens. Generating parallel plots...")

    if num_tokens == 0:
        return

    # 5. Loop per Layer
    for layer_idx in target_layers:
        hidden = hidden_states[layer_idx].cpu()
        
        # --- A. Collect Data First (To find Global Y-Limits) ---
        vectors = []
        titles = []
        all_values = [] # To find min/max
        
        for i, (b_idx, s_idx) in enumerate(graph_indices):
            vec = hidden[b_idx, s_idx, :].float().numpy()
            vectors.append(vec)
            all_values.append(vec)
            titles.append(f"Batch {b_idx}\nPos {s_idx}")

        # Determine Global Y-Limits for this layer
        # This ensures that a token with value 100,000 doesn't hide a token with value 5
        all_concat = np.concatenate(all_values)
        y_min = all_concat.min()
        y_max = all_concat.max()
        
        # Add a 10% buffer to the limits
        y_range = y_max - y_min
        y_lims = (y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        # --- B. Plotting Side-by-Side ---
        # Create a figure with N subplots (1 Row, N Columns)
        # Width = 3 inches per token, Height = 5 inches
        fig, axes = plt.subplots(1, num_tokens, figsize=(num_tokens * 3.5, 5), sharey=True)
        
        # Handle case where there is only 1 graph token (axes is not a list)
        if num_tokens == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            vec = vectors[i]
            x_axis = np.arange(len(vec))
            
            # Plot
            ax.plot(x_axis, vec, color='red', alpha=0.9, linewidth=0.8)
            ax.set_title(titles[i], fontsize=10)
            ax.set_xlabel("Dim")
            ax.grid(True, alpha=0.3)
            
            # Force the shared scale
            ax.set_ylim(y_lims)
            
            # Only label Y-axis on the first plot to reduce clutter
            if i == 0:
                ax.set_ylabel("Activation Value")
            
            # --- Auto-Highlight Peak (Only the #1 peak) ---
            sorted_indices = np.argsort(vec)[::-1]
            top_1 = sorted_indices[0]
            val_1 = vec[top_1]
            
            # Draw line and text
            ax.axvline(x=top_1, color='orange', linestyle='--', alpha=0.5)
            # Make text readable (alternating height or fixed)
            ax.text(top_1, val_1, f"dim:{top_1}; val:{val_1:.2f}", color='orange', fontweight='bold', 
                    ha='center', va='bottom', fontsize=7)

        plt.tight_layout()
        plt.suptitle(f"Layer {layer_idx} - Parallel Graph Token Activations", y=1.02, fontsize=14)
        
        # Save
        filename = f"layer_{layer_idx}_parallel_comparison.png"
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
        plt.close()

    print(f"[Analysis] Saved parallel comparison plots to {save_dir}")

#==============================================================================================

def probe_graph_attention_stats(outputs, is_node, step, base_save_dir):
    """
    Analyzes Attention Scores RECEIVED by Graph Tokens (Incoming Attention).
    Checks if Graph Tokens are acting as 'Sinks' (absorbing massive attention).
    
    Input:
      outputs: The model output object (containing .attentions)
      is_node: Boolean mask [Batch, Seq]
    """
    # 1. Setup
    save_dir = os.path.join(base_save_dir, f"step_{step}_attention_stats")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    attentions = outputs.attentions # Tuple of 32 tensors: (Batch, Heads, Seq, Seq)
    
    # 2. Layers to analyze (Early, Middle, Late)
    target_layers = [2, 10, 20, 24, 26, 30, 31]
    
    # 3. Identify Graph Token Indices
    # graph_indices = [[batch_idx, seq_idx], ...]
    graph_indices = (is_node == 1).nonzero(as_tuple=False)
    num_tokens = len(graph_indices)
    
    print(f"\n[Analysis Step {step}] Analyzing Attention Scores for {num_tokens} graph tokens...")

    if num_tokens == 0:
        return

    # 4. Loop per Layer
    for layer_idx in target_layers:
        # Get Attention Tensor for this layer: [Batch, Heads, Seq(Query), Seq(Key)]
        # We want to check columns (Key) corresponding to graph tokens
        attn_layer = attentions[layer_idx].detach().cpu()
        
        print(f"\n--- Layer {layer_idx} Attention Stats ---")
        
        max_scores_per_token = []
        avg_scores_per_token = []
        labels = []

        # Iterate over each graph token to see who is looking at it
        for i, (b_idx, s_idx) in enumerate(graph_indices):
            # Extract the COLUMN for this graph token
            # Shape: [Heads, Seq_Len] 
            # Meaning: "How much did every token (in every head) attend to ME?"
            token_incoming_attn = attn_layer[b_idx, :, :, s_idx] 
            
            # 1. Global Max for this token (The single strongest link)
            max_val = token_incoming_attn.max().item()
            
            # 2. Average Attention (General importance)
            mean_val = token_incoming_attn.mean().item()
            
            max_scores_per_token.append(max_val)
            avg_scores_per_token.append(mean_val)
            labels.append(f"B{b_idx}\nP{s_idx}")
            
            # Print high-value alerts
            if max_val > 0.5:
                 print(f"  > ALERT: Graph Token (Batch {b_idx}, Pos {s_idx}) received MAX attention: {max_val:.4f} (Sink Behavior?)")

        # --- Statistics Summary ---
        global_max = max(max_scores_per_token)
        global_avg = sum(avg_scores_per_token) / len(avg_scores_per_token)
        print(f"  Global Max Received Attention: {global_max:.4f}")
        print(f"  Average Received Attention:    {global_avg:.4f}")

        # --- Plotting: Max Attention Received by Each Graph Token ---
        plt.figure(figsize=(num_tokens * 0.8 + 4, 5)) # Dynamic width
        x_pos = np.arange(len(max_scores_per_token))
        
        # Bar Plot
        bars = plt.bar(x_pos, max_scores_per_token, color='purple', alpha=0.7)
        
        # Add Reference Line (Standard Attention is 1/Seq_Len approx 0.001, Strong is >0.1)
        #plt.axhline(y=0.1, color='gray', linestyle='--', label='Significant Threshold (0.1)')
        #plt.axhline(y=0.9, color='red', linestyle='--', label='Sink Threshold (0.9)')
        
        plt.title(f"Layer {layer_idx}: Max Attention Received by Graph Tokens")
        plt.ylabel("Max Attention Score (0.0 - 1.0)")
        plt.xlabel("Graph Token Index")
        plt.xticks(x_pos, labels, fontsize=8)
        plt.ylim(0, 1.1) # Attention is always 0-1
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"layer_{layer_idx}_max_attention.png"))
        plt.close()

    print(f"[Analysis] Attention stats saved to {save_dir}")



##################### Calculating Attention Inflows for Graph Tokens (2.3) ########################
@torch.no_grad()
def init_attention_inflow_stats() -> Dict[str, Any]:
    """
    Initializes an empty stats container.
    Tensors are created lazily on the first update call because L/H are only known
    after seeing outputs.attentions.
    """
    return {
        "initialized": False,
        "num_layers": None,
        "num_heads": None,

        # --- Three graph-token inflow variants (per layer/head) ---
        # 1) all tokens (valid queries) -> graph keys (includes graph/text + self if query==key at graph positions)
        "graph_inflow_all_sum_lh": None,         # [L, H]
        # 2) all tokens -> graph keys, but EXCLUDING self-attention when query==key at graph positions
        "graph_inflow_no_self_sum_lh": None,     # [L, H]
        # 3) text-only queries -> graph keys (queries are non-graph tokens)
        "graph_inflow_text_only_sum_lh": None,   # [L, H]

        # Denominators for "mean per graph-key token"
        "graph_key_count": 0,    # number of (valid) graph key positions aggregated

        # Optional bookkeeping
        "example_count": 0,
        "token_count_sum": 0,    # sum of valid query counts across examples
    }


@torch.no_grad()
def init_position_inflow_stats(seq_len: int) -> Dict[str, Any]:
    """
    Accumulators for per-position inflow aggregated across batches.
    Stores sums in float64 on CPU for stability.
    """
    return {
        "seq_len": int(seq_len),
        "initialized": False,
        "num_layers": None,
        "num_heads": None,

        # Sum of mean inflow per key position, averaged over heads+layers for each example,
        # then summed across examples. Shape: [T]
        "pos_inflow_sum": torch.zeros(seq_len, dtype=torch.float64, device="cpu"),

        # Counts: how many examples contributed to each position (valid tokens only). Shape: [T]
        "pos_valid_count": torch.zeros(seq_len, dtype=torch.float64, device="cpu"),

        # Optional: inflow restricted to text queries only (non-graph queries), still key-position indexed
        "pos_inflow_sum_text_queries": torch.zeros(seq_len, dtype=torch.float64, device="cpu"),
        "pos_valid_count_text_queries": torch.zeros(seq_len, dtype=torch.float64, device="cpu"),

        # Optional metadata
        "example_count": 0,
    }

@torch.no_grad()
def update_attention_inflow_stats(
    stats: Dict[str, Any],
    outputs,
    is_node: torch.Tensor,        # [B, T] bool
    attention_mask: torch.Tensor, # [B, T] {0,1}
    *,
    normalize_by_query_count: bool = True,
    apply_key_mask: bool = True,
) -> Dict[str, Any]:
    """
    Records three inflow variants for GRAPH TOKENS (as keys), per layer/head. Note that each stored value is an average value:

    (1) all->graph:
        all valid queries (text + graph) -> graph keys, INCLUDING self-attention.

    (2) all->graph (no self):
        all valid queries (text + graph) -> graph keys, EXCLUDING self-attention
        at graph positions (i=j where the position is a graph token).

    (3) text->graph:
        only text queries (non-graph, valid) -> graph keys.

    Notes:
    - The model's causal mask is already baked into attention weights A.
    - We also mask padding using attention_mask.
    - Means are computed later as (sum inflow) / (number of graph key positions aggregated).
    """
    attns = outputs.attentions
    if attns is None:
        raise ValueError("outputs.attentions is None. Ensure output_attentions=True in the model call.")

    if is_node.dtype != torch.bool:
        is_node = is_node.bool()
    valid = attention_mask.bool()

    # Graph-token key mask (only valid, non-pad positions)
    node_mask = is_node & valid  # [B, T]
    graph_key_count_batch = int(node_mask.sum().item())
    if graph_key_count_batch == 0:
        # Still update bookkeeping counts (optional)
        stats["example_count"] += int(valid.shape[0])
        stats["token_count_sum"] += int(valid.sum(dim=1).sum().item())
        return stats

    B, T = valid.shape
    L = len(attns)
    H = attns[0].shape[1]

    if not stats.get("initialized", False):
        stats["initialized"] = True
        stats["num_layers"] = L
        stats["num_heads"] = H
        stats["graph_inflow_all_sum_lh"] = torch.zeros((L, H), dtype=torch.float64, device="cpu")
        stats["graph_inflow_no_self_sum_lh"] = torch.zeros((L, H), dtype=torch.float64, device="cpu")
        stats["graph_inflow_text_only_sum_lh"] = torch.zeros((L, H), dtype=torch.float64, device="cpu")

    stats["graph_key_count"] += graph_key_count_batch
    stats["example_count"] += int(B)

    # Query counts for normalization (left padding is naturally handled via `valid`)
    q_count_all = valid.sum(dim=1).clamp(min=1)  # [B]
    stats["token_count_sum"] += int(q_count_all.sum().item())

    # Masks broadcastable to [B, 1, T, T]
    q_mask_all = valid[:, None, :, None]            # [B, 1, T, 1]
    q_mask_text = (valid & (~is_node))[:, None, :, None]  # [B, 1, T, 1] text-only queries
    k_mask = valid[:, None, None, :]               # [B, 1, 1, T]

    # Key-selection mask for graph tokens
    node_k = node_mask[:, None, None, :]           # [B, 1, 1, T]

    # Self-attention mask at graph positions: a [B, 1, T, T] tensor that is 1 only on (i==j) where position is a graph token
    # Start with identity [T, T], then broadcast to batch/head dims and AND with node positions.
    eye = torch.eye(T, device=valid.device, dtype=torch.bool)       # [T, T]
    node_diag = (node_mask[:, None, :] & eye[None, :, :])           # [B, T, T]  True only on graph-token diagonal
    node_diag = node_diag[:, None, :, :]                             # [B, 1, T, T]

    for l in range(L):
        A = attns[l].float()  # [B, H, T, T]

        # Apply padding masks (queries always; keys optionally)
        A_all = A * q_mask_all
        A_text = A * q_mask_text
        if apply_key_mask:
            A_all = A_all * k_mask
            A_text = A_text * k_mask

        # Variant (1): all -> graph (including self)
        inflow_all = A_all.sum(dim=2)  # sum over queries i -> [B, H, T]
        if normalize_by_query_count:
            inflow_all = inflow_all / q_count_all[:, None, None].to(inflow_all.dtype)

        graph_inflow_all_bh = (inflow_all * node_mask[:, None, :].to(inflow_all.dtype)).sum(dim=2)  # [B, H]

        # Variant (2): all -> graph, excluding self-attention at graph positions
        # Subtract the diagonal attention mass at graph positions from A_all before summing.
        # node_diag is [B,1,T,T]; broadcast over heads automatically.
        A_no_self = A_all.masked_fill(node_diag, 0.0)
        inflow_no_self = A_no_self.sum(dim=2)  # [B, H, T]
        if normalize_by_query_count:
            inflow_no_self = inflow_no_self / q_count_all[:, None, None].to(inflow_no_self.dtype)

        graph_inflow_no_self_bh = (inflow_no_self * node_mask[:, None, :].to(inflow_no_self.dtype)).sum(dim=2)  # [B, H]

        # Variant (3): text-only -> graph
        inflow_text = A_text.sum(dim=2)  # [B, H, T]
        if normalize_by_query_count:
            # Normalize by number of valid TEXT queries (not all queries)
            q_count_text = (valid & (~is_node)).sum(dim=1).clamp(min=1)  # [B]
            inflow_text = inflow_text / q_count_text[:, None, None].to(inflow_text.dtype)

        graph_inflow_text_bh = (inflow_text * node_mask[:, None, :].to(inflow_text.dtype)).sum(dim=2)  # [B, H]

        # Accumulate over batch -> [H], move to CPU float64
        stats["graph_inflow_all_sum_lh"][l] += graph_inflow_all_bh.sum(dim=0).double().cpu()
        stats["graph_inflow_no_self_sum_lh"][l] += graph_inflow_no_self_bh.sum(dim=0).double().cpu()
        stats["graph_inflow_text_only_sum_lh"][l] += graph_inflow_text_bh.sum(dim=0).double().cpu()

    return stats

def finalize_attention_inflow_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts accumulated sums into mean inflow per graph-key token (per layer/head)
    for each of the three variants.
    """
    if not stats.get("initialized", False):
        raise ValueError("Stats have not been initialized. Call update_attention_inflow_stats first.")

    graph_keys = max(int(stats["graph_key_count"]), 1)

    mean_all_lh = stats["graph_inflow_all_sum_lh"] / graph_keys
    mean_no_self_lh = stats["graph_inflow_no_self_sum_lh"] / graph_keys
    mean_text_only_lh = stats["graph_inflow_text_only_sum_lh"] / graph_keys

    return {
        "num_layers": int(stats["num_layers"]),
        "num_heads": int(stats["num_heads"]),
        "graph_key_count": int(stats["graph_key_count"]),
        "example_count": int(stats["example_count"]),
        "token_count_sum": int(stats["token_count_sum"]),

        # Means per layer/head
        "mean_graph_inflow_all_lh": mean_all_lh,                # all->graph, includes self
        "mean_graph_inflow_no_self_lh": mean_no_self_lh,        # all->graph, excludes self at graph positions
        "mean_graph_inflow_text_only_lh": mean_text_only_lh,    # text->graph only
    }

def save_attention_inflow_final(final: Dict[str, Any], save_path: str) -> None:
    """
    Save `final` to JSON. Tensors become lists via .tolist().
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    payload = {
        "num_layers": final["num_layers"],
        "num_heads": final["num_heads"],
        "graph_key_count": final["graph_key_count"],
        "example_count": final["example_count"],
        "token_count_sum": final["token_count_sum"],

        "mean_graph_inflow_all_lh": final["mean_graph_inflow_all_lh"].tolist(),
        "mean_graph_inflow_no_self_lh": final["mean_graph_inflow_no_self_lh"].tolist(),
        "mean_graph_inflow_text_only_lh": final["mean_graph_inflow_text_only_lh"].tolist(),

        "tensor_shapes": {
            "mean_graph_inflow_all_lh": list(final["mean_graph_inflow_all_lh"].shape),
            "mean_graph_inflow_no_self_lh": list(final["mean_graph_inflow_no_self_lh"].shape),
            "mean_graph_inflow_text_only_lh": list(final["mean_graph_inflow_text_only_lh"].shape),
        },
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@torch.no_grad()
def update_position_inflow_stats(
    stats: Dict[str, Any],
    outputs,
    attention_mask: torch.Tensor,     # [B, T]
    is_node: Optional[torch.Tensor] = None,  # [B, T] bool, optional
    *,
    normalize_by_query_count: bool = True,
    apply_key_mask: bool = True,
    include_text_query_variant: bool = True,
) -> Dict[str, Any]:
    """
    Computes per-position attention inflow:
      inflow[j] = sum_i A[i, j]
    then averages across heads and layers and accumulates across examples.

    Returns mean inflow per key POSITION index (0..T-1), not per token id.
    This is robust for left padding, because we only count valid positions via attention_mask.

    If include_text_query_variant=True and is_node is provided, also computes inflow when
    queries are restricted to TEXT positions (valid & ~is_node).
    """
    attns = outputs.attentions
    if attns is None:
        raise ValueError("outputs.attentions is None. Ensure output_attentions=True.")

    valid = attention_mask.bool()
    B, T = valid.shape
    if T != stats["seq_len"]:
        raise ValueError(f"seq_len mismatch: stats has {stats['seq_len']} but got {T}")

    L = len(attns)
    H = attns[0].shape[1]
    if not stats["initialized"]:
        stats["initialized"] = True
        stats["num_layers"] = L
        stats["num_heads"] = H

    # Masks broadcastable to [B, 1, T, T]
    q_mask_all = valid[:, None, :, None]       # [B, 1, T, 1]
    k_mask = valid[:, None, None, :]          # [B, 1, 1, T]

    q_count_all = valid.sum(dim=1).clamp(min=1)  # [B]

    # Accumulate per-example per-position inflow averaged over layers+heads.
    # We'll build a [B, T] tensor "inflow_mean_bT" and add into global sums.
    inflow_mean_bT = torch.zeros((B, T), dtype=torch.float32, device=valid.device)

    for l in range(L):
        A = attns[l].float()  # [B, H, T, T]
        A = A * q_mask_all
        if apply_key_mask:
            A = A * k_mask

        inflow = A.sum(dim=2)  # [B, H, T] sum over queries
        if normalize_by_query_count:
            inflow = inflow / q_count_all[:, None, None].to(inflow.dtype)

        # average over heads -> [B, T]
        inflow_hmean = inflow.mean(dim=1)
        inflow_mean_bT += inflow_hmean

    # average over layers -> [B, T]
    inflow_mean_bT = inflow_mean_bT / float(L)

    # Accumulate sums/counts per position (valid positions only)
    # Move to CPU float64 accumulators
    stats["pos_inflow_sum"] += (inflow_mean_bT * valid.to(inflow_mean_bT.dtype)).sum(dim=0).double().cpu()
    stats["pos_valid_count"] += valid.sum(dim=0).double().cpu()

    # Optional: text-query-only variant (queries restricted to valid & ~is_node)
    if include_text_query_variant and (is_node is not None):
        if is_node.dtype != torch.bool:
            is_node = is_node.bool()
        text_q = valid & (~is_node)
        q_mask_text = text_q[:, None, :, None]  # [B, 1, T, 1]
        q_count_text = text_q.sum(dim=1).clamp(min=1)  # [B]

        inflow_text_mean_bT = torch.zeros((B, T), dtype=torch.float32, device=valid.device)
        for l in range(L):
            A = attns[l].float()
            A = A * q_mask_text
            if apply_key_mask:
                A = A * k_mask
            inflow = A.sum(dim=2)  # [B, H, T]
            if normalize_by_query_count:
                inflow = inflow / q_count_text[:, None, None].to(inflow.dtype)
            inflow_text_mean_bT += inflow.mean(dim=1)

        inflow_text_mean_bT = inflow_text_mean_bT / float(L)

        stats["pos_inflow_sum_text_queries"] += (inflow_text_mean_bT * valid.to(inflow_text_mean_bT.dtype)).sum(dim=0).double().cpu()
        stats["pos_valid_count_text_queries"] += valid.sum(dim=0).double().cpu()

    stats["example_count"] += int(B)
    return stats

def finalize_position_inflow_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns mean inflow per position index, plus argmax (the "biggest sink position").
    """
    denom = stats["pos_valid_count"].clamp(min=1.0)
    mean_inflow_pos = stats["pos_inflow_sum"] / denom  # [T] float64 on CPU

    max_pos = int(torch.argmax(mean_inflow_pos).item())
    max_val = float(mean_inflow_pos[max_pos].item())

    out = {
        "seq_len": stats["seq_len"],
        "num_layers": stats["num_layers"],
        "num_heads": stats["num_heads"],
        "example_count": stats["example_count"],
        "mean_inflow_pos": mean_inflow_pos,   # tensor [T]
        "max_inflow_pos": max_pos,
        "max_inflow_value": max_val,
    }

    # Text-query-only variant if it was accumulated
    denom_t = stats["pos_valid_count_text_queries"].clamp(min=1.0)
    mean_inflow_pos_textq = stats["pos_inflow_sum_text_queries"] / denom_t
    out["mean_inflow_pos_text_queries"] = mean_inflow_pos_textq
    out["max_inflow_pos_text_queries"] = int(torch.argmax(mean_inflow_pos_textq).item())
    out["max_inflow_value_text_queries"] = float(mean_inflow_pos_textq[out["max_inflow_pos_text_queries"]].item())

    return out

def save_position_inflow_final(final: Dict[str, Any], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload = {
        "seq_len": final["seq_len"],
        "num_layers": final["num_layers"],
        "num_heads": final["num_heads"],
        "example_count": final["example_count"],
        "max_inflow_pos": final["max_inflow_pos"],
        "max_inflow_value": final["max_inflow_value"],
        "max_inflow_pos_text_queries": final["max_inflow_pos_text_queries"],
        "max_inflow_value_text_queries": final["max_inflow_value_text_queries"],
        "mean_inflow_pos": final["mean_inflow_pos"].tolist(),
        "mean_inflow_pos_text_queries": final["mean_inflow_pos_text_queries"].tolist(),
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@torch.no_grad()
def graph_attention_matrix(
    outputs,
    input_ids: torch.Tensor,            # [B, T]
    attention_mask: torch.Tensor,       # [B, T]
    is_node: torch.Tensor,              # [B, T] bool
    tokenizer,
    *,
    example_idx: int = 0,
    layer_agg: str = "mean",            # "mean" or "last"
    head_agg: str = "mean",             # "mean" or "none" (if "none", return [H, Q, K])
    max_text_rows: int = 60,            # cap rows for readability
) -> Dict[str, Any]:
    """
    Builds an attention matrix for a single example:
      rows = text query tokens (subset, in sequence order)
      cols = graph key tokens (in sequence order)
      values = attention weight averaged over heads/layers (unless head_agg="none").

    Returns dict with:
      - attn: Tensor [Q, K] or [H, Q, K]
      - query_pos: list of query positions (length Q)
      - key_pos: list of key positions (length K)
      - query_tokens: list of decoded query tokens
      - key_tokens: list of decoded graph tokens
    """
    attns = outputs.attentions
    if attns is None:
        raise ValueError("outputs.attentions is None. Use output_attentions=True.")
    if is_node.dtype != torch.bool:
        is_node = is_node.bool()

    valid = attention_mask.bool()
    b = example_idx
    T = input_ids.shape[1]

    # positions
    key_pos = torch.nonzero(valid[b] & is_node[b], as_tuple=False).squeeze(1)   # [K]
    query_pos = torch.nonzero(valid[b] & (~is_node[b]), as_tuple=False).squeeze(1)  # [Q_all]

    if key_pos.numel() == 0 or query_pos.numel() == 0:
        return {
            "attn": None,
            "query_pos": [],
            "key_pos": [],
            "query_tokens": [],
            "key_tokens": [],
        }

    # Optionally cap number of rows (take a window from the end is often more interesting in left padding)
    if query_pos.numel() > max_text_rows:
        query_pos = query_pos[-max_text_rows:]

    # choose layers
    if layer_agg == "last":
        layers = [len(attns) - 1]
    elif layer_agg == "mean":
        layers = list(range(len(attns)))
    else:
        raise ValueError("layer_agg must be 'mean' or 'last'")

    # accumulate attention
    # Each attns[l] is [B, H, T, T]
    A_accum = None
    for l in layers:
        A = attns[l][b].float()  # [H, T, T]
        # select query->key block: [H, Q, K]
        A_qk = A[:, query_pos, :][:, :, key_pos]
        if A_accum is None:
            A_accum = A_qk
        else:
            A_accum = A_accum + A_qk

    A_accum = A_accum / float(len(layers))  # [H, Q, K]

    if head_agg == "mean":
        attn_mat = A_accum.mean(dim=0)  # [Q, K]
    elif head_agg == "none":
        attn_mat = A_accum              # [H, Q, K]
    else:
        raise ValueError("head_agg must be 'mean' or 'none'")

    # labels
    query_token_ids = input_ids[b, query_pos].tolist()
    key_token_ids = input_ids[b, key_pos].tolist()
    query_tokens = [tokenizer.decode([tid]) for tid in query_token_ids]
    key_tokens = [tokenizer.decode([tid]) for tid in key_token_ids]

    return {
        "attn": attn_mat.detach().cpu(),
        "query_pos": query_pos.detach().cpu().tolist(),
        "key_pos": key_pos.detach().cpu().tolist(),
        "query_tokens": query_tokens,
        "key_tokens": key_tokens,
    }

@torch.no_grad()
def save_graph_attention_heatmaps_for_batch(
    *,
    outputs,
    input_ids: torch.Tensor,            # [B, T]
    attention_mask: torch.Tensor,       # [B, T]
    is_node: torch.Tensor,              # [B, T] bool
    tokenizer,
    save_dir: str,
    step: int,
    dataset_name: str = "",
    layer_agg: str = "mean",            # "mean" or "last"
    head_agg: str = "mean",             # "mean" recommended for plotting
    max_text_rows: int = 60,
    max_y_ticks: int = 30,
    prefix: str = "attn_text_to_graph",
    dpi: int = 150,
) -> None:
    """
    Same as before, but only plots/records TEXT query tokens that occur AFTER the last graph token
    position in the prompt. (Tokens before the graph block cannot attend to it in a causal decoder.)

    IMPORTANT: This function expects graph_attention_matrix(...) to return:
      - 'attn': Tensor [Q, K]
      - 'query_tokens': list[str] length Q
      - 'key_tokens': list[str] length K
      - 'query_pos': list[int] length Q   (absolute positions in the original prompt sequence)
      - 'key_pos': list[int] length K     (absolute positions in the original prompt sequence)
    """
    os.makedirs(save_dir, exist_ok=True)
    B = input_ids.shape[0]

    for b in range(B):
        mat_info: Dict[str, Any] = graph_attention_matrix(
            outputs=outputs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            is_node=is_node,
            tokenizer=tokenizer,
            example_idx=b,
            layer_agg=layer_agg,
            head_agg=head_agg,
            max_text_rows=max_text_rows,
        )

        attn = mat_info.get("attn", None)
        if attn is None:
            continue
        if attn.dim() != 2:
            raise ValueError(
                f"Expected 2D attention matrix [Q,K] for plotting, got shape {tuple(attn.shape)}. "
                f"Use head_agg='mean'."
            )

        query_tokens = mat_info.get("query_tokens", [])
        key_tokens = mat_info.get("key_tokens", [])
        query_pos = mat_info.get("query_pos", None)
        key_pos = mat_info.get("key_pos", None)

        if query_pos is None or key_pos is None:
            raise ValueError(
                "Missing 'query_pos' and/or 'key_pos' in mat_info. "
                "Please modify graph_attention_matrix to return absolute positions."
            )

        Q = len(query_tokens)
        K = len(key_tokens)
        if Q == 0 or K == 0:
            continue

        # ---- Filter: keep only query rows whose absolute position is AFTER the last graph token ----
        key_pos_arr = np.asarray(key_pos, dtype=int)
        last_graph_pos = int(key_pos_arr.max())

        query_pos_arr = np.asarray(query_pos, dtype=int)
        keep_mask = query_pos_arr > last_graph_pos
        keep_idx = np.nonzero(keep_mask)[0]

        if keep_idx.size == 0:
            continue

        attn_np_full = attn.numpy()                 # [Q, K]
        attn_np = attn_np_full[keep_idx, :]         # [Q_after, K]
        query_tokens_after = [query_tokens[i] for i in keep_idx]
        query_pos_after = query_pos_arr[keep_idx]   # absolute positions for reference if needed

        Q_after = len(query_tokens_after)
        if Q_after == 0:
            continue

        # ---------- Plot (with colorbar) ----------
        plt.figure(figsize=(max(6, K * 1.2), 10))
        im = plt.imshow(attn_np, aspect="auto", interpolation="nearest")
        plt.xlim(-0.5, K - 0.5)

        plt.xticks(np.arange(K), key_tokens, rotation=45, ha="right")

        # Y ticks: sparse labeling within the filtered rows
        if Q_after <= max_y_ticks:
            yticks = np.arange(Q_after)
        else:
            yticks = np.linspace(0, Q_after - 1, max_y_ticks).astype(int)

        ylabels = [query_tokens_after[i].replace("\n", "\\n") for i in yticks]
        plt.yticks(yticks, ylabels, fontsize=8)

        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("Attention weight", rotation=90)

        title_bits = []
        if dataset_name:
            title_bits.append(str(dataset_name))
        title_bits.append(f"step={step}")
        title_bits.append(f"ex={b}")
        plt.title(" | ".join(title_bits))

        plt.xlabel("Graph token")
        plt.ylabel("Text token")
        plt.tight_layout()

        base_name = f"{prefix}_{dataset_name}_step{step:06d}_ex{b:02d}" if dataset_name else f"{prefix}_step{step:06d}_ex{b:02d}"
        fig_path = os.path.join(save_dir, base_name + ".png")
        txt_path = os.path.join(save_dir, base_name + ".txt")

        plt.savefig(fig_path, dpi=dpi)
        plt.close()

        # ---------- Sidecar TXT ----------
        topn = min(20, Q_after)
        lines = []
        lines.append(f"dataset={dataset_name}")
        lines.append(f"step={step} ex={b}")
        lines.append(f"Q_full={Q} Q_after={Q_after} K={K}")
        lines.append(f"last_graph_pos={last_graph_pos}")
        lines.append("")

        # Mean attention per graph token across filtered text tokens
        col_means = attn_np.mean(axis=0)  # [K]
        lines.append("=== Mean attention per graph token (avg over TEXT query tokens AFTER graph block) ===")
        for k in range(K):
            key_name = key_tokens[k].replace("\n", "\\n")
            lines.append(f"col={k}  graph_token={key_name}  mean_attn={float(col_means[k]):.8f}")
        lines.append("")

        # Top-20 query tokens for each graph token (within filtered rows)
        for k in range(K):
            key_name = key_tokens[k].replace("\n", "\\n")
            col = attn_np[:, k]  # [Q_after]
            top_rows = np.argsort(col)[::-1][:topn]

            lines.append(f"=== Top {topn} query tokens (AFTER graph) for graph token col {k} ({key_name}) ===")
            for rank, r in enumerate(top_rows, start=1):
                tok = query_tokens_after[r].replace("\n", "\\n").strip()
                if tok == "":
                    tok = "<whitespace>"
                val = float(col[r])
                abs_pos = int(query_pos_after[r])
                lines.append(f"{rank:02d}) row_after={int(r):04d}  abs_pos={abs_pos:04d}  attn={val:.8f}  tok={tok}")
            lines.append("")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


## Acquire and plot per-layer attention weights

@torch.no_grad()
def graph_attention_matrix_per_layer(
    *,
    outputs,
    input_ids: torch.Tensor,           # [B, T]
    attention_mask: torch.Tensor,      # [B, T]
    is_node: torch.Tensor,             # [B, T] bool
    tokenizer,
    example_idx: int = 0,
    filter_after_graph: bool = True,
) -> Dict[str, Any]:
    """
    Returns per-layer attention matrices for ONE example, averaging over heads only.

    For each layer l:
      attn_layers[l] is a [Q, K] matrix where:
        - Q = number of query TEXT tokens (optionally only those AFTER last graph token)
        - K = number of graph tokens (<Node ...>) in the prompt for this example
        - value = mean over heads of attention weight A[q -> k]

    Requires outputs.attentions from HuggingFace forward pass:
      outputs.attentions is length L (layers), each tensor [B, H, T, T].

    Returns dict:
      - attn_layers: Tensor [L, Q, K] on CPU (float32)
      - query_tokens: list[str] length Q
      - key_tokens: list[str] length K
      - query_pos: list[int] length Q (absolute positions)
      - key_pos: list[int] length K (absolute positions)
      - last_graph_pos: int
    """
    b = example_idx

    attns = getattr(outputs, "attentions", None)
    if attns is None:
        raise ValueError("outputs.attentions is None. Make sure output_attentions=True.")

    L = len(attns)

    # Ensure boolean masks
    valid = attention_mask.bool()
    if is_node.dtype is not torch.bool:
        is_node = is_node.bool()

    # Graph token positions (keys)
    key_pos_t = torch.nonzero(valid[b] & is_node[b], as_tuple=False).squeeze(1)  # [K]
    if key_pos_t.numel() == 0:
        return {
            "attn_layers": None,
            "query_tokens": [],
            "key_tokens": [],
            "query_pos": [],
            "key_pos": [],
            "last_graph_pos": -1,
        }

    key_pos = key_pos_t.detach().cpu().tolist()
    last_graph_pos = int(max(key_pos))

    # Candidate text query positions: valid AND not node
    text_pos_t = torch.nonzero(valid[b] & (~is_node[b]), as_tuple=False).squeeze(1)  # [Q_full]

    if filter_after_graph:
        text_pos_t = text_pos_t[text_pos_t > last_graph_pos]

    if text_pos_t.numel() == 0:
        return {
            "attn_layers": None,
            "query_tokens": [],
            "key_tokens": [tokenizer.decode([int(input_ids[b, p].item())]) for p in key_pos],
            "query_pos": [],
            "key_pos": key_pos,
            "last_graph_pos": last_graph_pos,
        }

    query_pos = text_pos_t.detach().cpu().tolist()

    # Decode tokens for labels
    key_tokens = [tokenizer.decode([int(input_ids[b, p].item())]) for p in key_pos]
    query_tokens = [tokenizer.decode([int(input_ids[b, p].item())]) for p in query_pos]

    K = len(key_pos)
    Q = len(query_pos)

    # Build per-layer matrices: [L, Q, K]
    attn_layers = torch.zeros((L, Q, K), dtype=torch.float32)

    key_pos_dev = key_pos_t.to(attns[0].device)
    query_pos_dev = text_pos_t.to(attns[0].device)

    for l in range(L):
        A = attns[l]              # [B, H, T, T]
        A_b = A[b]                # [H, T, T]

        # Take submatrix: heads x queries x keys
        sub = A_b[:, query_pos_dev, :][:, :, key_pos_dev]  # [H, Q, K]  ==> over all heads, gather attention weights from query to keys: i.e., how much text tokens attend to graph tokens
        sub_mean = sub.mean(dim=0)                         # [Q, K]
        attn_layers[l] = sub_mean.detach().cpu().to(torch.float32)

    return {
        "attn_layers": attn_layers,
        "query_tokens": query_tokens,
        "key_tokens": key_tokens,
        "query_pos": query_pos,
        "key_pos": key_pos,
        "last_graph_pos": last_graph_pos,
    }


@torch.no_grad()
def save_graph_attention_heatmaps_per_layer_for_batch(
    *,
    outputs,
    input_ids: torch.Tensor,            # [B, T]
    attention_mask: torch.Tensor,       # [B, T]
    is_node: torch.Tensor,              # [B, T] bool
    tokenizer,
    save_dir: str,
    step: int,
    dataset_name: str = "",
    prefix: str = "attn_text_to_graph_per_layer",
    max_y_ticks: int = 30,
    dpi: int = 150,
    filter_after_graph: bool = True,
) -> None:
    """
    For every example in the batch, saves 32 heatmaps (one per layer) where heads are averaged.

    File naming:
      {prefix}_{dataset_name}_step{step:06d}_ex{b:02d}_layer{l:02d}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    B = input_ids.shape[0]

    for b in range(B):
        info = graph_attention_matrix_per_layer(
            outputs=outputs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            is_node=is_node,
            tokenizer=tokenizer,
            example_idx=b,
            filter_after_graph=filter_after_graph,
        )

        attn_layers = info["attn_layers"]
        if attn_layers is None:
            continue

        query_tokens = info["query_tokens"]
        key_tokens = info["key_tokens"]

        L, Q, K = attn_layers.shape
        if Q == 0 or K == 0:
            continue

        # Precompute y tick indices/labels (same for all layers in this sample)
        if Q <= max_y_ticks:
            yticks = np.arange(Q)
        else:
            yticks = np.linspace(0, Q - 1, max_y_ticks).astype(int)
        ylabels = [query_tokens[i].replace("\n", "\\n") for i in yticks]

        for l in range(L):
            attn_np = attn_layers[l].numpy()  # [Q, K]

            plt.figure(figsize=(max(6, K * 1.2), 10))
            im = plt.imshow(attn_np, aspect="auto", interpolation="nearest")
            plt.xlim(-0.5, K - 0.5)

            plt.xticks(np.arange(K), key_tokens, rotation=45, ha="right")
            plt.yticks(yticks, ylabels, fontsize=8)

            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label("Attention weight", rotation=90)

            title_bits = []
            if dataset_name:
                title_bits.append(str(dataset_name))
            title_bits.append(f"step={step}")
            title_bits.append(f"ex={b}")
            title_bits.append(f"layer={l}")
            plt.title(" | ".join(title_bits))

            plt.xlabel("Graph token")
            plt.ylabel("Text token")
            plt.tight_layout()

            base = f"{prefix}_{dataset_name}_step{step:06d}_ex{b:02d}_layer{l:02d}" if dataset_name else f"{prefix}_step{step:06d}_ex{b:02d}_layer{l:02d}"
            out_path = os.path.join(save_dir, base + ".png")
            plt.savefig(out_path, dpi=dpi)
            plt.close()


############### 2.5. After checking the attention weights from text to graph tokens, next examine activation values ##############

##### Below plots the average graph tokens
@torch.no_grad()
def init_activation_stats_storage(
    num_layers: int,
    hidden_dim: int = 4096,
    device: str = "cpu",
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Storage for running sums to compute mean activation vectors per layer.
    Stores one sum vector per layer + count.

    Returns:
      storage[layer_idx] = {
        'sum': Tensor[hidden_dim],
        'count': int (stored as Python int)
      }

    Note: for HF LLaMA, outputs.hidden_states usually has length num_layers+1
    (embedding output + each transformer layer). Decide whether you want to include layer 0.
    """
    storage = {}
    for l in range(num_layers):
        storage[l] = {
            "sum": torch.zeros(hidden_dim, dtype=torch.float32, device=device),
            "count": 0,
        }
    return storage


@torch.no_grad()
def update_activation_stats_storage(
    storage: Dict[int, Dict[str, torch.Tensor]],
    hidden_states,
    *,
    attention_mask: torch.Tensor,               # [B, T] (1=valid, 0=pad)
    is_node: torch.Tensor,                      # [B, T] bool (True for graph tokens)
    token_group: Literal["graph", "text", "all"] = "graph",
    filter_after_graph: bool = False,
) -> None:
    """
    Updates storage with sums/counts for the requested token group.

    hidden_states: outputs.hidden_states from HF forward, typically:
      tuple length L_total where each element is [B, T, D]
      L_total may be num_layers+1 (includes embeddings at index 0)

    token_group:
      - "graph": only graph tokens (is_node==True)
      - "text": only text tokens (is_node==False)
      - "all": all valid tokens (attention_mask==1)

    filter_after_graph:
      If True, restrict query tokens to positions strictly AFTER the last graph token
      per example. (This is often useful for causal settings.)
      This option is applied for "text" and "all"; for "graph" it has no effect.
    """
    if hidden_states is None:
        raise ValueError("hidden_states is None. Make sure output_hidden_states=True.")

    if is_node.dtype != torch.bool:
        is_node = is_node.bool()
    valid = attention_mask.bool()

    B, T = attention_mask.shape

    # Build a [B, T] mask of which tokens to include
    if token_group == "graph":
        mask = valid & is_node
    elif token_group == "text":
        mask = valid & (~is_node)
    elif token_group == "all":
        mask = valid
    else:
        raise ValueError("token_group must be one of: 'graph', 'text', 'all'.")

    if filter_after_graph and token_group in ("text", "all"):
        # For each example, keep only positions > last_graph_pos
        # If an example has no graph tokens, we keep all valid (for that example).
        mask2 = torch.zeros_like(mask)
        for b in range(B):
            graph_pos = torch.nonzero(valid[b] & is_node[b], as_tuple=False).squeeze(1)
            if graph_pos.numel() == 0:
                mask2[b] = mask[b]
            else:
                last_graph_pos = int(graph_pos.max().item())
                pos = torch.arange(T, device=mask.device)
                after = pos > last_graph_pos
                mask2[b] = mask[b] & after
        mask = mask2

    num_tokens = int(mask.sum().item())
    if num_tokens == 0:
        return

    # hidden_states is a tuple/list. Each element: [B, T, D]
    # We update only layers present in storage keys.
    for layer_idx in list(storage.keys()):
        h = hidden_states[layer_idx]  # [B, T, D]
        # Move to CPU float32 for stable accumulation (optional but recommended)
        h_cpu = h.detach().to(dtype=torch.float32, device="cpu")  # [B, T, D]
        m_cpu = mask.detach().to(device="cpu").unsqueeze(-1)      # [B, T, 1]

        # Zero out excluded tokens, sum over B and T -> [D]
        summed = (h_cpu * m_cpu).sum(dim=(0, 1))                  # [D]

        storage[layer_idx]["sum"] += summed
        storage[layer_idx]["count"] += num_tokens


@torch.no_grad()
def save_mean_activation_plots(
    storage: Dict[int, Dict[str, torch.Tensor]],
    save_dir: str,
    *,
    title_prefix: str = "Mean activation",
    topk_dims: int = 3,
    dpi: int = 150,
) -> None:
    """
    Saves one plot per layer:
      x-axis: hidden dimension index (0..D-1)
      y-axis: mean activation value over selected tokens
    Also marks top-k dimensions by |mean|.

    Output files:
      layer_{layer_idx:02d}_mean_activation.png
    """
    os.makedirs(save_dir, exist_ok=True)

    for layer_idx, data in storage.items():
        count = int(data["count"])
        if count == 0:
            continue

        mean_vec = (data["sum"] / count).cpu().numpy()  # [D]
        D = mean_vec.shape[0]
        x = np.arange(D)

        plt.figure(figsize=(12, 6))
        plt.plot(x, mean_vec, linewidth=0.8, alpha=0.9)  # do not set colors explicitly

        # Mark top-k by magnitude
        if topk_dims > 0:
            top_idx = np.argsort(np.abs(mean_vec))[::-1][:topk_dims]
            for r, idx in enumerate(top_idx):
                val = mean_vec[idx]
                plt.axvline(x=int(idx), linestyle="--", alpha=0.6)
                plt.text(
                    int(idx),
                    float(val),
                    f"dim {int(idx)}\n{float(val):.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.title(f"{title_prefix} | layer={layer_idx} | count={count}")
        plt.xlabel("Hidden dimension index")
        plt.ylabel("Mean activation value")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(save_dir, f"layer_{layer_idx:02d}_mean_activation.png")
        plt.savefig(out_path, dpi=dpi)
        plt.close()


@torch.no_grad()
def build_activation_topdims_summary(
    storage: Dict[int, Dict[str, torch.Tensor]],
    *,
    dataset_name: str = "",
    topk_dims: int = 5,
    sort_by_abs: bool = True,
) -> Dict[str, Any]:
    """
    Convert per-layer mean-activation storage into a compact summary file.

    The summary only keeps the top-k dimension indices for each layer, which is
    enough to aggregate counts across multiple datasets later.
    """
    if topk_dims <= 0:
        raise ValueError("topk_dims must be positive.")

    hidden_dim = 0
    for data in storage.values():
        if "sum" in data:
            hidden_dim = int(data["sum"].numel())
            break

    layer_indices: List[int] = []
    top_dims_by_layer: List[List[int]] = []

    for layer_idx in sorted(storage.keys()):
        data = storage[layer_idx]
        count = int(data["count"])
        if count == 0:
            continue

        mean_vec = (data["sum"] / count).detach().to(dtype=torch.float32, device="cpu")
        hidden_dim = max(hidden_dim, int(mean_vec.numel()))
        k = min(int(topk_dims), int(mean_vec.numel()))
        score_vec = mean_vec.abs() if sort_by_abs else mean_vec
        topk = torch.topk(score_vec, k=k, largest=True)

        layer_indices.append(int(layer_idx))
        top_dims_by_layer.append([int(dim_idx) for dim_idx in topk.indices.tolist()])

    return {
        "dataset_name": str(dataset_name),
        "hidden_dim": int(hidden_dim),
        "topk_dims": int(topk_dims),
        "sort_by_abs": bool(sort_by_abs),
        "layer_indices": layer_indices,
        "top_dims_by_layer": top_dims_by_layer,
    }


def save_activation_topdims_summary(
    *,
    storage: Dict[int, Dict[str, torch.Tensor]],
    save_path: str,
    dataset_name: str = "",
    topk_dims: int = 5,
    sort_by_abs: bool = True,
) -> str:
    """
    Save the per-layer top activation-dimension indices as JSON.
    """
    summary = build_activation_topdims_summary(
        storage,
        dataset_name=dataset_name,
        topk_dims=topk_dims,
        sort_by_abs=sort_by_abs,
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return save_path


def load_activation_topdims_summary(summary_path: str) -> Dict[str, Any]:
    """
    Load a JSON summary saved by save_activation_topdims_summary().
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    return {
        "dataset_name": str(summary.get("dataset_name", "")),
        "hidden_dim": int(summary["hidden_dim"]),
        "topk_dims": int(summary.get("topk_dims", 5)),
        "sort_by_abs": bool(summary.get("sort_by_abs", True)),
        "layer_indices": [int(layer_idx) for layer_idx in summary.get("layer_indices", [])],
        "top_dims_by_layer": [
            [int(dim_idx) for dim_idx in dims]
            for dims in summary.get("top_dims_by_layer", [])
        ],
    }


def plot_activation_topdims_count_aggregate(
    *,
    summary_paths: List[str],
    save_path: str,
    dpi: int = 180,
    annotate_top_n: int = 15,
    annotate_offset: int = 2,
) -> str:
    """
    Aggregate across all provided summary files and plot the frequency of each
    hidden dimension appearing in the per-layer top-k lists.
    """
    if len(summary_paths) == 0:
        raise ValueError("summary_paths must be non-empty.")

    summaries = [load_activation_topdims_summary(summary_path=p) for p in summary_paths]
    hidden_dim = int(max(summary["hidden_dim"] for summary in summaries))

    counts = torch.zeros((hidden_dim,), dtype=torch.long)
    for summary in summaries:
        for dims in summary["top_dims_by_layer"]:
            for dim_idx in dims:
                dim_idx = int(dim_idx)
                if 0 <= dim_idx < hidden_dim:
                    counts[dim_idx] += 1

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)
    x = torch.arange(hidden_dim, dtype=torch.long)
    ax.plot(x.tolist(), counts.tolist(), linewidth=1.2)

    ax.set_title("Top activation-dimension counts (aggregated)")
    ax.set_xlabel("Dimension number")
    ax.set_ylabel("Count among per-layer top-k dimensions")
    ax.grid(True, axis="y", alpha=0.25)

    nonzero = torch.nonzero(counts > 0, as_tuple=False).squeeze(1)
    if annotate_top_n and annotate_top_n > 0 and nonzero.numel() > 0:
        k = min(int(annotate_top_n), int(nonzero.numel()))
        topk = torch.topk(counts, k=k, largest=True)

        for dim_idx, count in zip(topk.indices.tolist(), topk.values.tolist()):
            if count <= 0:
                continue
            ax.annotate(
                str(dim_idx),
                xy=(dim_idx, count),
                xytext=(0, annotate_offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


############### Below examines the individual ones

@torch.no_grad()
def extract_graph_token_activations_per_layer(
    *,
    hidden_states,                       # outputs.hidden_states (tuple of [B, T, D])
    input_ids: torch.Tensor,             # [B, T]
    attention_mask: torch.Tensor,        # [B, T]
    is_node: torch.Tensor,               # [B, T] bool
    tokenizer,
    example_idx: int = 0,
    expected_k: Optional[int] = 5,       # set None to skip check
) -> Dict[str, Any]:
    """
    For one example in the batch, extract per-layer activation vectors for each graph token position.

    Returns:
      - acts: Tensor [L, K, D] on CPU float32
      - key_pos: list[int] absolute positions (sorted)
      - key_tokens: list[str] decoded tokens at those positions (length K)
    """
    if hidden_states is None:
        raise ValueError("hidden_states is None. Make sure output_hidden_states=True in model(...).")

    if is_node.dtype != torch.bool:
        is_node = is_node.bool()

    b = example_idx
    valid = attention_mask.bool()

    # positions of graph tokens for this example
    key_pos_t = torch.nonzero(valid[b] & is_node[b], as_tuple=False).squeeze(1)  # [K]
    if key_pos_t.numel() == 0:
        return {"acts": None, "key_pos": [], "key_tokens": []}

    # sort by position to align Node1..NodeK order in the sequence
    key_pos_t, _ = torch.sort(key_pos_t)
    key_pos = key_pos_t.detach().cpu().tolist()
    K = len(key_pos)

    if expected_k is not None and K != expected_k:
        # Not fatal, but useful for debugging if is_node marks more than the intended 5 tokens
        raise ValueError(f"Expected K={expected_k} graph tokens, but found K={K}. key_pos={key_pos}")

    key_tokens = [tokenizer.decode([int(input_ids[b, p].item())]) for p in key_pos]

    L = len(hidden_states)
    D = hidden_states[0].shape[-1]

    acts = torch.zeros((L, K, D), dtype=torch.float32)
    key_pos_dev = key_pos_t.to(hidden_states[0].device)

    for l in range(L):
        h = hidden_states[l][b]                      # [T, D]
        sel = h.index_select(dim=0, index=key_pos_dev)  # [K, D]
        acts[l] = sel.detach().to(dtype=torch.float32, device="cpu")

    return {"acts": acts, "key_pos": key_pos, "key_tokens": key_tokens}


@torch.no_grad()
def save_graph_token_activation_plots_for_batch(
    *,
    outputs,                              # HF model(...) outputs with hidden_states
    input_ids: torch.Tensor,              # [B, T]
    attention_mask: torch.Tensor,         # [B, T]
    is_node: torch.Tensor,                # [B, T] bool
    tokenizer,
    save_dir: str,
    step: int,
    dataset_name: str = "",
    layer_indices: Optional[Sequence[int]] = None,   # default: all layers in hidden_states
    expected_k: int = 5,
    prefix: str = "act_graph_tokens",
    dpi: int = 150,
    max_dims_to_plot: Optional[int] = None,          # e.g., 4096 or smaller for speed
) -> None:
    """
    For each example in the batch:
      - for each layer (32 or 33 depending on hidden_states), save ONE figure containing
        5 horizontal subplots (one per graph token) plotting activation over dimension index.

    File naming:
      {prefix}_{dataset}_step{step:06d}_ex{b:02d}_layer{l:02d}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise ValueError("outputs.hidden_states is None. Make sure output_hidden_states=True.")

    B = input_ids.shape[0]
    L_total = len(hidden_states)
    D = hidden_states[0].shape[-1]

    if layer_indices is None:
        layer_indices = list(range(L_total))

    # Optionally plot only the first N dims (useful if plots are heavy)
    D_plot = D if (max_dims_to_plot is None) else min(int(max_dims_to_plot), D)
    x = np.arange(D_plot)

    for b in range(B):
        info = extract_graph_token_activations_per_layer(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            is_node=is_node,
            tokenizer=tokenizer,
            example_idx=b,
            expected_k=expected_k,
        )

        acts = info["acts"]
        if acts is None:
            continue

        key_tokens = info["key_tokens"]  # length K (=5)
        K = acts.shape[1]

        for l in layer_indices:
            vecs = acts[l, :, :D_plot].numpy()  # [K, D_plot]

            # 5 subplots horizontally
            fig, axes = plt.subplots(1, K, figsize=(4.2 * K, 3.2), sharey=True)
            if K == 1:
                axes = [axes]

            for k in range(K):
                ax = axes[k]
                ax.plot(x, vecs[k], linewidth=0.7, alpha=0.9)
                # Title: prefer showing which node slot it is; also include decoded token for debugging
                tok = key_tokens[k].replace("\n", "\\n")
                ax.set_title(f"Node {k+1}\n{tok}", fontsize=9)
                ax.set_xlabel("Dim")
                if k == 0:
                    ax.set_ylabel("Activation")
                ax.grid(True, alpha=0.25)

            title_bits = []
            if dataset_name:
                title_bits.append(str(dataset_name))
            title_bits.append(f"step={step}")
            title_bits.append(f"ex={b}")
            title_bits.append(f"layer={l}")
            fig.suptitle(" | ".join(title_bits), fontsize=11)

            fig.tight_layout()
            fig.subplots_adjust(top=0.82)

            base = (
                f"{prefix}_{dataset_name}_step{step:06d}_ex{b:02d}_layer{l:02d}"
                if dataset_name else
                f"{prefix}_step{step:06d}_ex{b:02d}_layer{l:02d}"
            )
            out_path = os.path.join(save_dir, base + ".png")
            fig.savefig(out_path, dpi=dpi)
            plt.close(fig)


######### 2.5, LLM generated output that attends to the graph ###########
@torch.no_grad()
def generate_with_attentions(
    model,
    *,
    inputs_embeds: torch.Tensor,        # [B, T, D]
    attention_mask: torch.Tensor,       # [B, T]
    max_new_tokens: int = 80,
    do_sample: bool = True,
    temperature: float = 0.7,
):
    """
    Generate and return a generation object that includes per-step attentions.
    """
    gen_out = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        return_dict_in_generate=True,
        output_attentions=True,
    )
    return gen_out


@torch.no_grad()
def gen_text_to_graph_attention_per_layer(
    gen_out,
    *,
    prompt_input_ids: torch.Tensor,      # [B, T]
    prompt_attention_mask: torch.Tensor, # [B, T]
    prompt_is_node: torch.Tensor,        # [B, T] bool
    tokenizer,
    example_idx: int = 0,
) -> Dict[str, Any]:
    b = example_idx
    if prompt_is_node.dtype != torch.bool:
        prompt_is_node = prompt_is_node.bool()
    valid = prompt_attention_mask.bool()

    # Graph token positions in the prompt (keys)
    key_pos_t = torch.nonzero(valid[b] & prompt_is_node[b], as_tuple=False).squeeze(1)
    if key_pos_t.numel() == 0:
        return {"attn_layers": None, "gen_tokens": [], "key_tokens": [], "key_pos": []}

    key_pos_t, _ = torch.sort(key_pos_t)
    key_pos = key_pos_t.detach().cpu().tolist()
    key_tokens = [tokenizer.decode([int(prompt_input_ids[b, p].item())]) for p in key_pos]
    K = len(key_pos)

    attns = getattr(gen_out, "attentions", None)
    if attns is None:
        raise ValueError("gen_out.attentions is None. Ensure output_attentions=True in generate().")

    # Number of recorded decoding steps
    S_attn = len(attns)  # <-- use this as the true step count
    if S_attn == 0:
        return {"attn_layers": None, "gen_tokens": [], "key_tokens": key_tokens, "key_pos": key_pos}

    # Determine generated token ids to label rows with, aligned to S_attn
    seq = gen_out.sequences[b].detach().cpu().tolist()
    T_prompt = int(prompt_input_ids.shape[1])

    # Many HF versions return sequences = prompt + generated
    if len(seq) >= T_prompt:
        gen_ids_full = seq[T_prompt:]
    else:
        gen_ids_full = seq

    # Align: keep the last S_attn generated ids (most consistent with how attentions are recorded)
    if len(gen_ids_full) >= S_attn:
        gen_ids = gen_ids_full[-S_attn:]
    else:
        # Rare: fewer ids than attention steps; fall back by truncating steps
        S_attn = len(gen_ids_full)
        attns = attns[:S_attn]
        gen_ids = gen_ids_full

    gen_tokens = [tokenizer.decode([tid]) for tid in gen_ids]

    # attns[s] should be tuple/list of layers
    if not isinstance(attns[0], (tuple, list)):
        raise ValueError("Unexpected gen_out.attentions structure: expected per-step tuple/list of layers.")

    L = len(attns[0])
    out = torch.zeros((L, S_attn, K), dtype=torch.float32)

    for s in range(S_attn):
        layers_s = attns[s]  # length L
        for l in range(L):
            A = layers_s[l]

            if A.dim() == 4:
                # [B, H, 1, cur_len] -> [H, cur_len]
                A_b = A[b, :, 0, :]
            elif A.dim() == 3:
                # [B, H, cur_len] -> [H, cur_len]
                A_b = A[b]
            else:
                raise ValueError(f"Unexpected attention tensor dim={A.dim()} at step={s}, layer={l}")

            kp = key_pos_t.to(A_b.device)

            # Safety: if cur_len is shorter than some key positions (should not happen, but guard anyway)
            cur_len = A_b.shape[-1]
            kp = kp[kp < cur_len]
            if kp.numel() == 0:
                continue

            A_graph = A_b[:, kp]      # [H, K'] where K'<=K if truncated
            v = A_graph.mean(dim=0)   # [K']
            # write into first K' columns
            out[l, s, : v.numel()] = v.detach().cpu()

    return {
        "attn_layers": out,          # [L, S_attn, K]
        "gen_tokens": gen_tokens,    # length S_attn
        "key_tokens": key_tokens,
        "key_pos": key_pos,
    }


@torch.no_grad()
def save_gen_text_to_graph_heatmaps_per_layer_for_batch(
    *,
    model,
    outputs,                           # prompt forward outputs (not strictly needed except embeds)
    embeds: torch.Tensor,              # [B, T, D] from first_model
    input_ids: torch.Tensor,           # [B, T]
    attention_mask: torch.Tensor,      # [B, T]
    is_node: torch.Tensor,             # [B, T] bool
    tokenizer,
    save_dir: str,
    step: int,
    dataset_name: str = "",
    prefix: str = "gen_text_to_graph_per_layer",
    max_new_tokens: int = 80,
    max_y_ticks: int = 30,
    dpi: int = 150,
    do_sample: bool = True,
    temperature: float = 0.7,
    layer_indices: Optional[Sequence[int]] = None,   # default: all layers returned by generate
) -> None:
    """
    For each example in the batch:
      - run generation once for the batch (captures attentions)
      - save per-layer heatmaps of generated token (query) -> graph token (key in prompt)
        Heads averaged; layers not averaged.

    File naming:
      {prefix}_{dataset}_step{step:06d}_ex{b:02d}_layer{l:02d}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1) Generate once for this batch (expensive; gate with step < 2 outside)
    gen_out = generate_with_attentions(
        model,
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )

    B = input_ids.shape[0]

    for b in range(B):
        info = gen_text_to_graph_attention_per_layer(
            gen_out,
            prompt_input_ids=input_ids,
            prompt_attention_mask=attention_mask,
            prompt_is_node=is_node,
            tokenizer=tokenizer,
            example_idx=b,
        )
        attn_layers = info["attn_layers"]
        if attn_layers is None:
            continue

        gen_tokens = info["gen_tokens"]
        key_tokens = info["key_tokens"]

        L, S, K = attn_layers.shape
        if S == 0 or K == 0:
            continue

        if layer_indices is None:
            layer_indices = list(range(L))

        # y ticks for generated tokens
        if S <= max_y_ticks:
            yticks = np.arange(S)
        else:
            yticks = np.linspace(0, S - 1, max_y_ticks).astype(int)
        ylabels = [gen_tokens[i].replace("\n", "\\n") for i in yticks]

        for l in layer_indices:
            mat = attn_layers[l].numpy()  # [S, K]

            plt.figure(figsize=(max(6, K * 1.2), 10))
            im = plt.imshow(mat, aspect="auto", interpolation="nearest")
            plt.xlim(-0.5, K - 0.5)

            plt.xticks(np.arange(K), key_tokens, rotation=45, ha="right")
            plt.yticks(yticks, ylabels, fontsize=8)

            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label("Attention weight (generated query → graph key)", rotation=90)

            title_bits = []
            if dataset_name:
                title_bits.append(str(dataset_name))
            title_bits.append(f"step={step}")
            title_bits.append(f"ex={b}")
            title_bits.append(f"layer={l}")
            title_bits.append(f"gen_len={S}")
            plt.title(" | ".join(title_bits))

            plt.xlabel("Graph token (key in prompt)")
            plt.ylabel("Generated token (query)")
            plt.tight_layout()

            base = (
                f"{prefix}_{dataset_name}_step{step:06d}_ex{b:02d}_layer{l:02d}"
                if dataset_name else
                f"{prefix}_step{step:06d}_ex{b:02d}_layer{l:02d}"
            )
            out_path = os.path.join(save_dir, base + ".png")
            plt.savefig(out_path, dpi=dpi)
            plt.close()


############## 2.6 Trying to remap to original graph ################

def _ensure_node_mapping(node_set: Optional[torch.Tensor], num_nodes: int) -> Optional[List[int]]:
    if node_set is None:
        return None
    if isinstance(node_set, torch.Tensor):
        node_set = node_set.detach().cpu().tolist()
    # If instruction stores global ids for each local node, lengths should match.
    if len(node_set) != num_nodes:
        # Still return what we have; caller can decide how to interpret.
        return node_set
    return node_set

def graph_token_grad_attribution(
    *,
    first_model,                          # your GraphEncoder instance
    graph: Data,                          # PyG Data for ONE sample (not Batch)
    graph_token_idx: int,                 # 0..num_token-1 (e.g., 0..4)
    node_set: Optional[torch.Tensor] = None,  # local->global ids (instruction['node_set'])
    score_mode: str = "l2",               # "l2" or "mean_abs" or "sum"
    lp: Optional[bool] = None,            # override graph.lp if needed
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    Gradient attribution from ONE graph token embedding to input node features graph.x.

    Returns:
      - node_scores_local: Tensor [N] (importance per local node)
      - node_scores_global: list[(global_id, score)] if node_set provided
      - token_vec: Tensor [D] (the selected graph token embedding)
      - scalar_score: float (the scalar used for gradients)
    """
    assert hasattr(first_model, "GT") and hasattr(first_model, "graph_projector"), \
        "first_model must have .GT and .graph_projector (GraphEncoder)."

    # Put GNN/projector in eval mode to avoid dropout noise
    was_training = first_model.training
    first_model.eval()
    first_model.GT.eval()

    device = next(first_model.parameters()).device

    # Prepare graph
    g = graph.to(device)
    if lp is None:
        lp = bool(getattr(g, "lp", False))
    # Ensure batch exists (single-graph case)
    if getattr(g, "batch", None) is None:
        g.batch = torch.zeros(g.x.size(0), dtype=torch.long, device=device)
    # Ensure lp tensor shape [num_graphs]=[1]
    g_lp = torch.tensor([lp], dtype=torch.bool, device=device)

    # Make node features float32 and require grad
    gnn_dtype = next(first_model.GT.parameters()).dtype  # typically torch.bfloat16
    x = g.x.to(dtype=gnn_dtype)
    x = x.detach().clone().requires_grad_(True)

    # Forward through GraphSAGE -> projector -> token slice
    z = first_model.GT(x, g.edge_index, getattr(g, "edge_attr", None), g.batch, g_lp)  # [1, gnn_output]
    p = first_model.graph_projector(z)  # [1, num_token * embed_dim]

    num_token = int(first_model.args.num_token)
    embed_dim = int(first_model.embed_dim)
    assert p.shape[1] == num_token * embed_dim, f"Unexpected projector output dim: {p.shape}"

    tokens = p.view(num_token, embed_dim)  # [num_token, D]
    assert 0 <= graph_token_idx < num_token, f"graph_token_idx must be in [0, {num_token-1}]"
    token_vec = tokens[graph_token_idx]    # [D]

    # Scalar score for attribution
    if score_mode == "l2":
        scalar = torch.norm(token_vec, p=2)
    elif score_mode == "mean_abs":
        scalar = token_vec.abs().mean()
    elif score_mode == "sum":
        scalar = token_vec.sum()
    else:
        raise ValueError("score_mode must be one of: 'l2', 'mean_abs', 'sum'")

    # Compute grads w.r.t. x
    grad_x = torch.autograd.grad(
        outputs=scalar,
        inputs=x,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]  # [N, F]

    # Node importance: L2 norm of gradient per node
    node_scores = torch.norm(grad_x, p=2, dim=1).detach().cpu()  # [N]

    if normalize:
        denom = node_scores.sum().item()
        if denom > 0:
            node_scores = node_scores / denom

    # Map to global ids if provided
    node_set_list = _ensure_node_mapping(node_set, num_nodes=int(x.size(0)))
    node_scores_global = None
    if node_set_list is not None:
        node_scores_global = [(int(node_set_list[i]), float(node_scores[i].item())) for i in range(len(node_scores))]

    # Restore training state
    if was_training:
        first_model.train()
        first_model.GT.train()

    return {
        "node_scores_local": node_scores,                  # Tensor [N]
        "node_scores_global": node_scores_global,          # list or None
        "token_vec": token_vec.detach().cpu(),             # Tensor [D]
        "scalar_score": float(scalar.detach().cpu().item())
    }


def graph_token_grad_attribution_for_batch(
    *,
    first_model,
    batch_graph: Batch,                       # batch['graph']
    graph_token_idx: int,
    node_sets: Optional[List[torch.Tensor]] = None,  # list of node_set per example (variable length)
    score_mode: str = "l2",
    normalize: bool = True,
) -> List[Dict[str, Any]]:
    """
    Runs attribution for each sample in a PyG Batch. Returns a list of dicts (one per sample).
    """
    graphs = batch_graph.to_data_list()
    out = []
    for i, g in enumerate(graphs):
        ns = None
        if node_sets is not None and i < len(node_sets):
            ns = node_sets[i]
        out.append(
            graph_token_grad_attribution(
                first_model=first_model,
                graph=g,
                graph_token_idx=graph_token_idx,
                node_set=ns,
                score_mode=score_mode,
                normalize=normalize,
            )
        )
    return out


############## 2.9. Aggregation of samples through all batches ###########

def init_text2graph_dataset_agg(
    *,
    num_layers: int,
    q_max: int,
    k_graph: int = 5,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Storage for dataset-level aggregation of text->graph attention.

    sum:   [L, Q, K] float32
    count: [Q] int64  (number of samples contributing to each row)
    """
    return {
        "sum": torch.zeros((num_layers, q_max, k_graph), dtype=torch.float32, device=device),
        "count": torch.zeros((q_max,), dtype=torch.int64, device=device),
        "q_max": q_max,
        "k_graph": k_graph,
        "num_layers": num_layers,
        "n_samples_used": 0,
        "n_skipped": 0,
    }


def _get_graph_and_query_indices_for_example(
    *,
    is_node: torch.Tensor,          # [T] bool or {0,1}
    attention_mask: torch.Tensor,   # [T] bool or {0,1}
    expected_k: int = 5,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns:
      key_idx:   [K] indices of graph tokens (sorted)
      query_idx: [Q] indices of text tokens AFTER last graph token (sorted)
    """
    # valid tokens only
    valid = attention_mask.bool()

    node_mask = is_node.bool() & valid
    key_idx = torch.nonzero(node_mask, as_tuple=False).squeeze(-1)

    if key_idx.numel() < expected_k:
        return None

    # If there are more than K node tokens (unlikely), keep the first K in order.
    key_idx = torch.sort(key_idx).values[:expected_k]

    last_k = int(key_idx[-1].item())
    # queries are tokens after the graph-token block
    query_mask = valid.clone()
    query_mask[: last_k + 1] = False
    query_idx = torch.nonzero(query_mask, as_tuple=False).squeeze(-1)

    if query_idx.numel() == 0:
        return None

    return key_idx, query_idx


@torch.no_grad()
def update_text2graph_dataset_agg(
    *,
    storage: Dict[str, Any],
    outputs,                         # HF model outputs with .attentions
    attention_mask: torch.Tensor,    # [B, T]
    is_node: torch.Tensor,           # [B, T]
    expected_k: int = 5,
    head_agg: str = "mean",          # only "mean" supported here
) -> None:
    """
    Updates dataset-level aggregation using the current batch.

    For each example b and each layer l:
      - take head-mean attention A_l[b] -> [T,T]
      - slice rows = tokens after last graph token (query)
      - slice cols = graph token indices (key) (K=5)
      - take first q_len <= q_max rows
      - add to storage["sum"][l, :q_len, :]
    Also updates storage["count"][:q_len] by +1 per example (shared across layers).
    """
    attns = getattr(outputs, "attentions", None)
    if attns is None:
        raise ValueError("outputs.attentions is None. Make sure output_attentions=True.")

    # HuggingFace: [Batch_size, num_heads, seq_length, seq_length]
    L = len(attns)
    if L != storage["num_layers"]:
        # It is okay if storage was initialized with a different L; handle dynamically
        raise ValueError(f"Storage num_layers={storage['num_layers']} but outputs has {L} layers.")

    B, T = attention_mask.shape
    q_max = int(storage["q_max"])

    for b in range(B):
        idxs = _get_graph_and_query_indices_for_example(    # Stores the graph & text indices
            is_node=is_node[b],
            attention_mask=attention_mask[b],
            expected_k=expected_k,
        )
        if idxs is None:
            storage["n_skipped"] += 1
            continue

        key_idx, query_idx = idxs
        q_len = min(int(query_idx.numel()), q_max)
        if q_len <= 0:
            storage["n_skipped"] += 1
            continue

        # Only take the first q_len tokens after the graph block (relative positions 0..q_len-1)
        query_idx = query_idx[:q_len]

        # Update count ONCE per example (not per layer)
        storage["count"][:q_len] += 1
        storage["n_samples_used"] += 1

        for l in range(L):
            A = attns[l]  # [B,H,T,T] typically
            if A.dim() != 4:
                raise ValueError(f"Unexpected attn tensor dim={A.dim()} at layer={l}.")

            # [H,T,T]
            A_b = A[b]

            if head_agg != "mean":
                raise ValueError("Only head_agg='mean' is supported for dataset aggregation.")

            # head-mean -> [T,T]
            A_mean = A_b.mean(dim=0)

            # text->graph: rows=query, cols=keys
            sub = A_mean.index_select(0, query_idx).index_select(1, key_idx)  # [q_len, K]

            # accumulate on CPU float32
            storage["sum"][l, :q_len, :] += sub.detach().to(torch.float32).cpu()
            

def save_text2graph_dataset_heatmaps_with_stats(
    *,
    storage: Dict[str, Any],
    save_dir: str,
    dataset_name: str = "",
    prefix: str = "dataset_text_to_graph",
    dpi: int = 150,
    show_y_ticks: bool = False,
) -> None:
    """
    Saves:
      - per-layer heatmaps: [Q,K]
      - final heatmap: mean over layers
      - a stats .txt: mean/max/median per graph token for each layer + final
    """
    os.makedirs(save_dir, exist_ok=True)

    sum_ = storage["sum"]          # [L,Q,K] float32 (on CPU)
    count = storage["count"]       # [Q] int64 (on CPU)
    L, Q, K = sum_.shape

    denom = count.to(torch.float32).clamp_min(1.0).view(1, Q, 1)  # [1,Q,1]
    mean_per_layer = sum_ / denom                                 # [L,Q,K]
    mean_all_layers = mean_per_layer.mean(dim=0)                  # [Q,K]

    # rows that actually have data
    valid_rows = (count > 0).cpu().numpy()
    valid_idx = np.where(valid_rows)[0]

    xlabels = [f"Node {i}" for i in range(1, K + 1)]

    # ---- helper: compute stats for a [Q,K] matrix ----
    def _stats_for_matrix(mat_qk: np.ndarray) -> Dict[str, np.ndarray]:
        # mat_qk: [Q,K]
        if valid_idx.size == 0:
            # no data case
            return {
                "mean": np.zeros((K,), dtype=np.float32),
                "max": np.zeros((K,), dtype=np.float32),
                "median": np.zeros((K,), dtype=np.float32),
            }
        sub = mat_qk[valid_idx, :]  # [Q_valid,K]
        return {
            "mean": sub.mean(axis=0).astype(np.float32),
            "max": sub.max(axis=0).astype(np.float32),
            "median": np.median(sub, axis=0).astype(np.float32),
        }

    # ---- compute stats per layer + final ----
    per_layer_stats = []
    for l in range(L):
        mat = mean_per_layer[l].cpu().numpy()  # [Q,K]
        per_layer_stats.append(_stats_for_matrix(mat))
    final_stats = _stats_for_matrix(mean_all_layers.cpu().numpy())

    # ---- Save per-layer plots ----
    for l in range(L):
        mat = mean_per_layer[l].cpu().numpy()  # float32 [Q,K]

        plt.figure(figsize=(6.5, 8.0))
        im = plt.imshow(mat, aspect="auto", interpolation="nearest")
        plt.colorbar(im)

        plt.xticks(np.arange(K), xlabels, rotation=45, ha="right")

        if show_y_ticks:
            yticks = np.linspace(0, Q - 1, min(10, Q)).astype(int)
            plt.yticks(yticks, [str(int(y)) for y in yticks])
            plt.ylabel("Relative text position after graph block")
        else:
            plt.yticks([])
            plt.ylabel("Text positions (after graph block)")

        title_bits = []
        if dataset_name:
            title_bits.append(str(dataset_name))
        title_bits.append(f"Layer {l}")
        title_bits.append("mean over dataset")
        plt.title(" | ".join(title_bits))
        plt.xlabel("Graph tokens (key)")

        fname = (
            f"{prefix}_{dataset_name}_layer{l:02d}.png"
            if dataset_name else
            f"{prefix}_layer{l:02d}.png"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname), dpi=dpi)
        plt.close()

    # ---- Save final plot ----
    plt.figure(figsize=(6.5, 8.0))
    im = plt.imshow(mean_all_layers.cpu().numpy(), aspect="auto", interpolation="nearest")
    plt.colorbar(im)

    plt.xticks(np.arange(K), xlabels, rotation=45, ha="right")

    if show_y_ticks:
        yticks = np.linspace(0, Q - 1, min(10, Q)).astype(int)
        plt.yticks(yticks, [str(int(y)) for y in yticks])
        plt.ylabel("Relative text position after graph block")
    else:
        plt.yticks([])
        plt.ylabel("Text positions (after graph block)")

    title_bits = []
    if dataset_name:
        title_bits.append(str(dataset_name))
    title_bits.append("All layers mean")
    title_bits.append("mean over dataset")
    plt.title(" | ".join(title_bits))
    plt.xlabel("Graph tokens (key)")

    fname = (
        f"{prefix}_{dataset_name}_final.png"
        if dataset_name else
        f"{prefix}_final.png"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=dpi)
    plt.close()

    # ---- Save stats file ----
    stats_path = os.path.join(
        save_dir,
        f"{prefix}_{dataset_name}_stats.txt" if dataset_name else f"{prefix}_stats.txt"
    )

    def _fmt_arr(a: np.ndarray) -> str:
        return "  ".join([f"{v:.6f}" for v in a.tolist()])

    with open(stats_path, "w") as f:
        f.write(f"n_samples_used={storage.get('n_samples_used', 'NA')}\n")
        f.write(f"n_skipped={storage.get('n_skipped', 'NA')}\n")
        f.write(f"q_max={storage.get('q_max', 'NA')}\n")
        f.write(f"k_graph={storage.get('k_graph', 'NA')}\n")
        f.write(f"valid_rows={int(valid_idx.size)}/{Q}\n\n")

        f.write("Per-layer stats (across y-axis rows with count>0):\n")
        f.write("Columns correspond to Node1..Node5\n\n")

        for l in range(L):
            st = per_layer_stats[l]
            f.write(f"[Layer {l:02d}]\n")
            f.write(f"  mean   : {_fmt_arr(st['mean'])}\n")
            f.write(f"  max    : {_fmt_arr(st['max'])}\n")
            f.write(f"  median : {_fmt_arr(st['median'])}\n\n")

        f.write("[FINAL (mean over layers)]\n")
        f.write(f"  mean   : {_fmt_arr(final_stats['mean'])}\n")
        f.write(f"  max    : {_fmt_arr(final_stats['max'])}\n")
        f.write(f"  median : {_fmt_arr(final_stats['median'])}\n")



def plot_layerwise_token_stats_one_figure(
    *,
    per_layer_stats: Optional[list] = None,
    stats_txt_path: Optional[str] = None,
    save_dir: str,
    dataset_name: str = "",
    prefix: str = "layerwise_token_stats",
    dpi: int = 150,
) -> str:
    """
    Creates ONE figure showing mean/median/max over layers for each of the 5 graph tokens.

    You can provide either:
      (A) per_layer_stats: list of length L, each element dict with keys 'mean','median','max' (each shape [5])
      OR
      (B) stats_txt_path: the stats file produced by save_text2graph_dataset_heatmaps_with_stats()

    Output: saves PNG and returns its path.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Load data ---
    if per_layer_stats is None:
        if stats_txt_path is None:
            raise ValueError("Provide either per_layer_stats or stats_txt_path.")

        # Minimal parser for the format written by save_text2graph_dataset_heatmaps_with_stats
        means, medians, maxs = [], [], []
        with open(stats_txt_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines()]

        cur_layer = None
        for ln in lines:
            if ln.startswith("[Layer"):
                cur_layer = True
            elif cur_layer and ln.startswith("mean"):
                arr = ln.split(":", 1)[1].strip().split()
                means.append([float(x) for x in arr])
            elif cur_layer and ln.startswith("max"):
                arr = ln.split(":", 1)[1].strip().split()
                maxs.append([float(x) for x in arr])
            elif cur_layer and ln.startswith("median"):
                arr = ln.split(":", 1)[1].strip().split()
                medians.append([float(x) for x in arr])
                # done for this layer block

        mean_mat = np.array(means, dtype=np.float32)     # [L,5]
        median_mat = np.array(medians, dtype=np.float32) # [L,5]
        max_mat = np.array(maxs, dtype=np.float32)       # [L,5]
    else:
        L = len(per_layer_stats)
        mean_mat = np.stack([d["mean"] for d in per_layer_stats], axis=0).astype(np.float32)     # [L,5]
        median_mat = np.stack([d["median"] for d in per_layer_stats], axis=0).astype(np.float32) # [L,5]
        max_mat = np.stack([d["max"] for d in per_layer_stats], axis=0).astype(np.float32)       # [L,5]

    L, K = mean_mat.shape
    if K != 5:
        raise ValueError(f"Expected 5 graph tokens, got {K}.")

    x = np.arange(L)

    # --- Plot: 1x5 small multiples ---
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharey=True)
    if K == 1:
        axes = [axes]

    for k in range(5):
        ax = axes[k]
        ax.plot(x, mean_mat[:, k], linewidth=1.4, label="mean")
        ax.plot(x, median_mat[:, k], linewidth=1.4, label="median")
        ax.plot(x, max_mat[:, k], linewidth=1.4, label="max")

        ax.set_title(f"Graph token {k+1}", fontsize=10)
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.25)

        if k == 0:
            ax.set_ylabel("Attention (text → graph)")

        # keep ticks readable
        ax.set_xticks([0, L // 2, L - 1])

    # Legend once, outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

    title_bits = []
    if dataset_name:
        title_bits.append(str(dataset_name))
    # title_bits.append("Layerwise summary stats (mean / median / max)")
    fig.suptitle(" | ".join(title_bits), y=1.08, fontsize=12)

    fig.tight_layout()

    out_name = f"{prefix}_{dataset_name}.png" if dataset_name else f"{prefix}.png"
    out_path = os.path.join(save_dir, out_name)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return out_path



############ Attention Rollout ###########

def init_text2graph_rollout_dataset_agg(
    *,
    q_max: int,
    k_graph: int = 5,
) -> Dict[str, Any]:
    """
    Stores dataset-level sum/count for the rollout matrix slice:
      rollout[query_positions_after_graph, graph_positions]
    """
    return {
        "q_max": int(q_max),
        "k_graph": int(k_graph),
        "sum": torch.zeros(q_max, k_graph, dtype=torch.float32),  # CPU
        "count": torch.zeros(q_max, dtype=torch.long),            # CPU
        "n_samples_used": 0,
        "n_skipped": 0,
    }

# def _get_graph_and_query_indices_for_example(
#     *,
#     is_node_1d: torch.Tensor,         # [T] bool
#     attention_mask_1d: torch.Tensor,  # [T] bool/int
#     expected_k: int = 5,
# ) -> Optional[tuple]:
#     """
#     Returns:
#       key_idx:   LongTensor[K] graph token indices in the sequence
#       query_idx: LongTensor[Q] indices of tokens AFTER the last graph token (valid tokens only)
#     """
#     # valid positions
#     valid = attention_mask_1d.bool()
#     # graph token positions among valid tokens
#     key_idx = torch.nonzero(is_node_1d & valid, as_tuple=False).view(-1)

#     if key_idx.numel() < expected_k:
#         return None

#     # if there are more than K (should not happen typically), take the last K in sequence order
#     # but usually it is exactly K contiguous slots.
#     key_idx = key_idx[-expected_k:]

#     last_graph_pos = int(key_idx.max().item())
#     # tokens after the graph block, still valid
#     query_mask = valid.clone()
#     query_mask[: last_graph_pos + 1] = False
#     query_idx = torch.nonzero(query_mask, as_tuple=False).view(-1)

#     return key_idx.to(torch.long), query_idx.to(torch.long)

@torch.no_grad()
def update_text2graph_dataset_rollout_agg(
    *,
    storage: Dict[str, Any],
    outputs,                         # HF outputs with .attentions
    attention_mask: torch.Tensor,    # [B, T]
    is_node: torch.Tensor,           # [B, T] bool
    expected_k: int = 5,
    add_identity: bool = True,
    eps: float = 1e-12,
) -> None:
    """
    Dataset aggregation for rollout(text -> graph).

    For each example b:
      - build rollout R over layers: [T,T]
      - slice rows=query after graph block, cols=graph token indices
      - accumulate into storage['sum'] (CPU float32)
      - update storage['count'] for rows that exist
    """
    attns = getattr(outputs, "attentions", None)
    if attns is None:
        raise ValueError("outputs.attentions is None. Make sure output_attentions=True.")

    B, T = attention_mask.shape
    q_max = int(storage["q_max"])
    K = int(storage["k_graph"])

    # attns: tuple length L, each [B,H,T,T]
    L = len(attns)

    for b in range(B):
        idxs = _get_graph_and_query_indices_for_example(    # Stores the graph & text indices
            is_node=is_node[b],
            attention_mask=attention_mask[b],
            expected_k=expected_k,
        )
        if idxs is None:
            storage["n_skipped"] += 1
            continue

        key_idx, query_idx = idxs
        if key_idx.numel() != K:
            storage["n_skipped"] += 1
            continue

        q_len = min(int(query_idx.numel()), q_max)
        if q_len <= 0:
            storage["n_skipped"] += 1
            continue
        query_idx = query_idx[:q_len]

        # ---- rollout init: Identity [T,T] on device ----
        device = attns[0].device
        R = torch.eye(T, device=device, dtype=torch.float32)

        # ---- multiply layer by layer ----
        for l in range(L):
            A = attns[l]  # [B,H,T,T]
            if A.dim() != 4:
                raise ValueError(f"Unexpected attention tensor dim={A.dim()} at layer={l}.")

            A_b = A[b].to(torch.float32)          # [H,T,T]
            A_mean = A_b.mean(dim=0)              # [T,T]

            if add_identity:
                A_mean = A_mean + torch.eye(T, device=device, dtype=torch.float32)

            # row-normalize
            A_mean = A_mean / (A_mean.sum(dim=-1, keepdim=True) + eps)

            # rollout update
            R = R @ A_mean

        # ---- slice text->graph from rollout ----
        sub = R.index_select(0, query_idx).index_select(1, key_idx)  # [q_len,K]

        # ---- min-max normalize to [0,1] (per-sample, on the sliced block) ----
        sub_min = sub.min()
        sub_max = sub.max()
        denom = (sub_max - sub_min).clamp_min(eps)
        sub = (sub - sub_min) / denom

        # ---- accumulate ----
        storage["sum"][:q_len, :] += sub.detach().to(torch.float32).cpu()
        storage["count"][:q_len] += 1
        storage["n_samples_used"] += 1


def save_text2graph_rollout_dataset_heatmap_all_rows(
    *,
    storage: Dict[str, Any],
    save_dir: str,
    dataset_name: str = "",
    prefix: str = "dataset_text_to_graph_rollout",
    dpi: int = 150,
) -> str:
    os.makedirs(save_dir, exist_ok=True)

    sum_ = storage["sum"]      # [Q,K] CPU float32
    count = storage["count"]   # [Q] CPU long
    Q, K = sum_.shape

    # mean per row; if count[r]=0 -> denominator becomes 1, value stays 0 (represents "no samples")
    denom = count.to(torch.float32).clamp_min(1.0).view(Q, 1)
    mean = (sum_ / denom).numpy()  # [Q,K]

    plt.figure(figsize=(6.5, 8.0))
    im = plt.imshow(mean, aspect="auto", interpolation="nearest")
    plt.colorbar(im)

    xlabels = [f"Node {i}" for i in range(1, K + 1)]
    plt.xticks(np.arange(K), xlabels, rotation=45, ha="right")
    plt.yticks([])

    plt.xlabel("Graph tokens")
    plt.ylabel("Text positions after graph block")

    title_bits = []
    if dataset_name:
        title_bits.append(str(dataset_name))
    title_bits.append("Rollout mean over dataset")
    plt.title(" | ".join(title_bits))

    # note = (
    #     f"n_samples_used={storage.get('n_samples_used', 'NA')}, "
    #     f"n_skipped={storage.get('n_skipped', 'NA')}\n"
    #     f"count[0]={int(count[0].item()) if Q>0 else 0}, "
    #     f"count[last]={int(count[-1].item()) if Q>0 else 0}"
    # )
    plt.subplots_adjust(bottom=0.18)
    #plt.gcf().text(0.02, 0.02, note, ha="left", va="bottom", fontsize=9)

    fname = f"{prefix}_{dataset_name}.png" if dataset_name else f"{prefix}.png"
    out_path = os.path.join(save_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

    return out_path


##### 2.16 LLM Gradient Probe #####


import os
from typing import Any, Dict, List, Optional, Sequence

import torch
import numpy as np


def _split_scores_by_ptr(scores: torch.Tensor, ptr: torch.Tensor) -> List[torch.Tensor]:
    """
    scores: [N_total]
    ptr: [B+1]
    returns list length B, each [N_b]
    """
    out = []
    for i in range(ptr.numel() - 1):
        s, e = int(ptr[i].item()), int(ptr[i + 1].item())
        out.append(scores[s:e])
    return out


def _ensure_node_set_list(node_sets, B: int) -> List[Optional[torch.Tensor]]:
    if node_sets is None:
        return [None] * B
    # already list-like
    if isinstance(node_sets, (list, tuple)):
        out = list(node_sets)
        if len(out) < B:
            out = out + [None] * (B - len(out))
        return out[:B]
    # single tensor -> treat as batch size 1
    return [node_sets] + [None] * (B - 1)


def _get_graph_token_positions(is_node_b: torch.Tensor) -> torch.Tensor:
    """
    is_node_b: [T] bool/int
    returns positions [K] (K graph tokens in the prompt sequence)
    """
    if is_node_b.dtype != torch.bool:
        is_node_b = (is_node_b == 1)
    pos = torch.nonzero(is_node_b, as_tuple=False).squeeze(-1)  # [K]
    return pos


def _select_hidden_for_graph_tokens(
    hidden: torch.Tensor,  # [B,T,D]
    is_node: torch.Tensor, # [B,T] bool/int
    expected_k: int = 5,
) -> torch.Tensor:
    """
    Returns h_g: [B,K,D] where K=expected_k.
    Assumes each sample has exactly K graph token slots.
    """
    B, T, D = hidden.shape
    h_out = []
    for b in range(B):
        pos = _get_graph_token_positions(is_node[b])
        if pos.numel() != expected_k:
            raise ValueError(
                f"Example {b}: expected {expected_k} graph token positions, got {pos.numel()}."
            )
        h_out.append(hidden[b, pos, :])  # [K,D]
    return torch.stack(h_out, dim=0)  # [B,K,D]


def _node_importance_from_grad_x(
    grad_x: torch.Tensor,   # [N_total, F]
    ptr: Optional[torch.Tensor],
    normalize: bool = True,
) -> List[torch.Tensor]:
    """
    Returns list of per-sample node scores (local indexing), each [N_b].
    """
    scores = torch.norm(grad_x.float(), p=2, dim=1)  # [N_total]
    if ptr is None:
        s = scores.detach().cpu()
        if normalize and s.sum().item() > 0:
            s = s / s.sum()
        return [s]
    else:
        scores_list = _split_scores_by_ptr(scores.detach().cpu(), ptr.detach().cpu())
        if normalize:
            out = []
            for s in scores_list:
                denom = s.sum().item()
                out.append(s / denom if denom > 0 else s)
            return out
        return scores_list


def llm_hidden_norm_grad_attribution_for_batch(
    *,
    llm_model,
    first_model,                 # GraphEncoder (has GT + graph_projector + embed_tokens insertion logic)
    batch: Dict[str, Any],
    graph_token_idx: int,        # 0..K-1
    node_sets: Optional[List[torch.Tensor]] = None,
    expected_k: int = 5,
    layer_indices: Optional[Sequence[int]] = None,  # e.g., [8,16,24,32] or None=use last layer
    layer_reduce: str = "mean",  # "mean" or "sum" over selected layers
    normalize_nodes: bool = True,
) -> List[Dict[str, Any]]:
    """
    Attribute graph nodes -> ONE graph token slot (graph_token_idx) using hidden-state norm
    AFTER the LLM forward.

    Returns list length B:
      {
        "node_scores_local": Tensor[N_b],
        "node_scores_global": Optional[List[(global_id, score)]],
        "scalar": float,
        "layers_used": list[int],
        "graph_token_idx": int
      }
    """
    llm_model.eval()
    first_model.eval()
    if hasattr(first_model, "GT"):
        first_model.GT.eval()

    input_ids = batch["input_ids"].to(llm_model.device)
    is_node = batch["is_node"].to(llm_model.device)
    attention_mask = batch["attn_mask"].to(llm_model.device)
    g = batch["graph"].to(llm_model.device)

    B = input_ids.shape[0]

    # node_sets mapping
    node_sets_list = _ensure_node_set_list(node_sets if node_sets is not None else batch.get("node_set", None), B)

    # ---- Make a grad-tracked leaf for node features ----
    # Ensure dtype matches GNN weights to avoid Float vs BFloat16 errors.
    gnn_dtype = next(first_model.GT.parameters()).dtype if hasattr(first_model, "GT") else g.x.dtype
    x_leaf = g.x.detach().clone().to(dtype=gnn_dtype).requires_grad_(True)

    # Replace graph.x with leaf
    g.x = x_leaf

    # ---- Forward: GraphEncoder (GNN + projector + insertion) -> LLM ----
    # IMPORTANT: do NOT wrap in torch.no_grad()
    embeds = first_model(input_ids=input_ids, is_node=is_node, graph=g)  # [B,T,D]
    outputs = llm_model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    hidden_states = outputs.hidden_states  # tuple length L_total, each [B,T,D]
    L_total = len(hidden_states)

    # choose layers
    if layer_indices is None:
        layer_indices = [L_total - 1]  # last layer only by default
    layer_indices = [int(l) for l in layer_indices]
    for l in layer_indices:
        if l < -1 or l >= L_total:
            raise ValueError(f"layer index {l} out of range (0..{L_total-1}).")

    # ---- Build scalar: ||h|| for graph token slot k, optionally aggregated over layers ----
    # Get per-layer h_g: [B,K,D], then select token k: [B,D]
    per_layer_vecs = []
    for l in layer_indices:
        h = hidden_states[l]  # [B,T,D]
        h_g = _select_hidden_for_graph_tokens(h, is_node=is_node, expected_k=expected_k)  # [B,K,D]
        v = h_g[:, graph_token_idx, :]  # [B,D]
        per_layer_vecs.append(v)

    # stack -> [num_layers,B,D]
    V = torch.stack(per_layer_vecs, dim=0)

    if layer_reduce == "mean":
        v_red = V.mean(dim=0)  # [B,D]
    elif layer_reduce == "sum":
        v_red = V.sum(dim=0)   # [B,D]
    else:
        raise ValueError("layer_reduce must be 'mean' or 'sum'.")

    # scalar per sample: L2 norm
    scalars = torch.norm(v_red, p=2, dim=1)  # [B]

    # We want per-sample attributions, so do grads one by one (keeps logic simple and correct).
    # This is slower but reliable; you can later vectorize if needed.
    ptr = getattr(g, "ptr", None)
    if ptr is None:
        # Single graph case; make dummy ptr
        ptr = torch.tensor([0, g.x.size(0)], device=llm_model.device, dtype=torch.long)

    results: List[Dict[str, Any]] = []
    for b in range(B):
        # clear old grads
        if x_leaf.grad is not None:
            x_leaf.grad.zero_()

        grad_x = torch.autograd.grad(
            outputs=scalars[b],
            inputs=x_leaf,
            retain_graph=True,   # we reuse forward for other b
            create_graph=False,
            allow_unused=False,
        )[0]  # [N_total,F]

        # node scores per sample
        node_scores_list = _node_importance_from_grad_x(grad_x, ptr=ptr, normalize=normalize_nodes)
        node_scores_local = node_scores_list[b]  # [N_b]

        # map to global ids if provided
        ns = node_sets_list[b]
        node_scores_global = None
        if ns is not None:
            ns_cpu = ns.detach().cpu()
            if ns_cpu.numel() != node_scores_local.numel():
                # If mismatch happens, do not crash; just skip mapping
                node_scores_global = None
            else:
                node_scores_global = [
                    (int(ns_cpu[i].item()), float(node_scores_local[i].item()))
                    for i in range(ns_cpu.numel())
                ]

        results.append({
            "node_scores_local": node_scores_local,
            "node_scores_global": node_scores_global,
            "scalar": float(scalars[b].detach().cpu().item()),
            "layers_used": list(layer_indices),
            "graph_token_idx": int(graph_token_idx),
        })

    return results


def llm_hidden_norm_grad_attribution_all_k_for_batch(
    *,
    llm_model,
    first_model,
    batch: Dict[str, Any],
    expected_k: int = 5,
    layer_indices: Optional[Sequence[int]] = None,
    layer_reduce: str = "mean",
    normalize_nodes: bool = True,
) -> List[List[Dict[str, Any]]]:
    """
    Returns attrs_per_token: length K, each is list length B of dicts.
    """
    out = []
    for k in range(expected_k):
        out.append(
            llm_hidden_norm_grad_attribution_for_batch(
                llm_model=llm_model,
                first_model=first_model,
                batch=batch,
                graph_token_idx=k,
                expected_k=expected_k,
                layer_indices=layer_indices,
                layer_reduce=layer_reduce,
                normalize_nodes=normalize_nodes,
            )
        )
    return out


def save_llm_graph_attribution_panels_for_batch(
    *,
    batch_graph,                          # PyG Batch
    node_sets: Optional[List[torch.Tensor]],
    attrs_per_token: List[List[Dict[str, Any]]],   # shape: [K][B]
    save_dir: str,
    step: int,
    dataset_name: str = "",
    prefix: str = "llm_hiddennorm_attr",
    dpi: int = 150,
) -> None:
    """
    Saves one panel per sample b:
      5 subplots (one per graph token k),
      each subplot: node attribution over local nodes (optionally mapped by node_sets).
    """

    os.makedirs(save_dir, exist_ok=True)

    graphs = batch_graph.to_data_list()
    B = len(graphs)
    K = len(attrs_per_token)

    # Basic shape checks (print helpful info instead of hard skip)
    bad = False
    for k in range(K):
        if len(attrs_per_token[k]) != B:
            print(f"[WARN] attrs_per_token[{k}] has len={len(attrs_per_token[k])}, but B={B}.")
            bad = True
    if bad:
        print("[WARN] attrs_per_token shape mismatch. Expected attrs_per_token[k][b]. Aborting save for this batch.")
        return

    node_sets_list = node_sets if (node_sets is not None) else [None] * B
    if len(node_sets_list) < B:
        node_sets_list = node_sets_list + [None] * (B - len(node_sets_list))

    for b in range(B):
        g = graphs[b]
        num_nodes = int(g.num_nodes)

        fig, axes = plt.subplots(1, K, figsize=(4.2 * K, 3.4), sharey=False)
        if K == 1:
            axes = [axes]

        for k in range(K):
            ax = axes[k]
            r = attrs_per_token[k][b]
            scores = r.get("node_scores_local", None)

            if scores is None:
                ax.set_title(f"Token {k+1}\n(no scores)")
                ax.axis("off")
                continue

            # scores could be on GPU / bfloat16; convert safely
            if isinstance(scores, torch.Tensor):
                scores_cpu = scores.detach().to(torch.float32).cpu().numpy()
            else:
                scores_cpu = np.asarray(scores, dtype=np.float32)

            # If scores cover all nodes in the *subgraph* (local indexing), length should match num_nodes.
            # If not, still plot what we have (but warn).
            if scores_cpu.shape[0] != num_nodes:
                print(f"[WARN] sample {b} token {k}: scores len={scores_cpu.shape[0]} != num_nodes={num_nodes}. Plotting anyway.")

            x = np.arange(scores_cpu.shape[0])
            ax.plot(x, scores_cpu, linewidth=1.0)
            ax.set_title(f"Token {k+1}", fontsize=10)
            ax.set_xlabel("Local node index")
            ax.set_ylabel("Attribution")
            ax.grid(True, alpha=0.25)

        title_bits = []
        if dataset_name:
            title_bits.append(str(dataset_name))
        title_bits.append(f"step={step}")
        title_bits.append(f"ex={b}")
        fig.suptitle(" | ".join(title_bits), fontsize=11)

        fig.tight_layout()
        fig.subplots_adjust(top=0.82)

        fname = (
            f"{prefix}_{dataset_name}_step{step:06d}_ex{b:02d}.png"
            if dataset_name
            else f"{prefix}_step{step:06d}_ex{b:02d}.png"
        )
        fig.savefig(os.path.join(save_dir, fname), dpi=dpi)
        plt.close(fig)


##################### 2.17 Perturbation-based Methods #########################

def insert_graph_embeds_no_inplace(
    text_embeds: torch.Tensor,     # [B,T,D]
    is_node: torch.Tensor,         # [B,T] bool
    node_embeds: torch.Tensor,     # [B,K,D]
) -> torch.Tensor:
    """
    Replace is_node positions in text_embeds with node_embeds (flattened by order),
    without in-place assignment, to preserve autograd stability.
    """
    B, T, D = text_embeds.shape
    if is_node.shape != (B, T):
        raise ValueError(f"is_node shape {tuple(is_node.shape)} != {(B,T)}")
    if node_embeds.dim() != 3 or node_embeds.size(0) != B or node_embeds.size(2) != D:
        raise ValueError(f"node_embeds must be [B,K,D] with D={D}, got {tuple(node_embeds.shape)}")

    # Flatten mask and build replacement tensor of shape [B,T,D]
    out = text_embeds.clone()
    mask = is_node.unsqueeze(-1).to(out.dtype)  # [B,T,1] (0/1)

    # We need to place K embeddings into the True slots in row-major order per sample.
    # Build a [B,T,D] tensor filled with 0, then scatter per-sample.
    repl = torch.zeros_like(out)

    for b in range(B):
        idx = torch.nonzero(is_node[b], as_tuple=False).squeeze(-1)  # [K]
        Kb = idx.numel()
        if Kb != node_embeds.size(1):
            raise ValueError(f"Sample {b}: expected {node_embeds.size(1)} graph slots, got {Kb}.")
        repl[b, idx, :] = node_embeds[b]

    # out = out*(1-mask) + repl*mask
    return out * (1.0 - mask) + repl * mask


def edge_index_to_adj_list(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u].append(v)
            adj[v].append(u)
    return adj

def hop_distances(edge_index: torch.Tensor, num_nodes: int, center: int = 0, max_hop: Optional[int] = None) -> torch.Tensor:
    adj = edge_index_to_adj_list(edge_index, num_nodes)
    dist = [-1] * num_nodes
    dist[center] = 0
    q = [center]
    head = 0
    while head < len(q):
        u = q[head]; head += 1
        du = dist[u]
        if max_hop is not None and du >= max_hop:
            continue
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = du + 1
                q.append(v)
    return torch.tensor(dist, dtype=torch.long)

def mask_features_by_exact_hop(x: torch.Tensor, dist: torch.Tensor, hop: int, include_center: bool = False) -> torch.Tensor:
    m = (dist == int(hop))
    if not include_center:
        m = m & (dist != 0)
    x2 = x.clone()
    x2[m] = 0
    return x2

def mask_features_by_radius(x: torch.Tensor, dist: torch.Tensor, radius: int, include_center: bool = False) -> torch.Tensor:
    m = (dist >= 0) & (dist <= int(radius))
    if not include_center:
        m = m & (dist != 0)
    x2 = x.clone()
    x2[m] = 0
    return x2


def llm_graph_token_grad_attribution_with_mask(
    *,
    llm_model,          # your HF LLM wrapper (model(...) supports inputs_embeds, output_hidden_states)
    first_model,        # GraphEncoder (has .GT, .graph_projector, .embed_tokens / embedding matrix)
    batch: Dict[str, Any],
    token_idx: int,     # 0..4
    layer_idx: int = -1,          # which hidden_states layer to score on
    mask_mode: str = "none",      # "none" | "exact" | "radius"
    mask_hop: int = 1,            # hop to mask (exact) or radius
    center_local_idx: int = 0,
    include_center: bool = False,
    normalize: bool = True,
) -> List[Dict[str, Any]]:
    """
    Returns list length B. Each element contains:
      - node_scores_local: Tensor [N_local]
      - hop_dist: Tensor [N_local]
      - token_pos: int (position in sequence for this graph token)
    """
    llm_model.eval()
    first_model.eval()
    first_model.GT.eval()

    input_ids = batch["input_ids"]          # [B,T]
    is_node = batch["is_node"]              # [B,T] bool
    attention_mask = batch["attn_mask"]     # [B,T]
    batch_graph = batch["graph"]            # PyG Batch
    node_sets = batch.get("node_set", None) # optional list per sample (local->global)

    device = llm_model.device
    input_ids = input_ids.to(device)
    is_node = is_node.to(device)
    attention_mask = attention_mask.to(device)

    graphs = batch_graph.to_data_list()
    B = len(graphs)

    # Precompute text embeddings once (graph placeholder ids are OK only if your embedding table supports them;
    # in your pipeline, first_model(...) already works, so we reuse that logic below and avoid calling embedding here.)
    results: List[Dict[str, Any]] = []

    for b in range(B):
        g = graphs[b].to(device)

        # --- build hop distances on CPU (small subgraphs) ---
        N = int(g.num_nodes)
        dist = hop_distances(g.edge_index, N, center=center_local_idx).to(device)

        # --- masked node features x with grad ---
        gnn_dtype = next(first_model.GT.parameters()).dtype
        x0 = g.x.to(dtype=gnn_dtype)

        if mask_mode == "none":
            x_masked = x0
        elif mask_mode == "exact":
            x_masked = mask_features_by_exact_hop(x0, dist, hop=mask_hop, include_center=include_center)
        elif mask_mode == "radius":
            x_masked = mask_features_by_radius(x0, dist, radius=mask_hop, include_center=include_center)
        else:
            raise ValueError("mask_mode must be one of: none, exact, radius")

        x = x_masked.detach().clone().requires_grad_(True)  # [N,F]

        # --- run GraphSAGE on this single graph, then projector -> [K,D] ---
        if getattr(g, "batch", None) is None:
            g.batch = torch.zeros(N, dtype=torch.long, device=device)
        # lp handling consistent with your GraphSAGE forward signature
        g_lp = getattr(g, "lp", None)
        if g_lp is None or not isinstance(g_lp, torch.Tensor):
            g_lp = torch.tensor([False], dtype=torch.bool, device=device)
        else:
            g_lp = g_lp.to(device)
            if g_lp.dim() == 0:
                g_lp = g_lp.view(1)

        z = first_model.GT(x, g.edge_index, getattr(g, "edge_attr", None), g.batch, g_lp)  # [1, gnn_output]
        p = first_model.graph_projector(z)                                                 # [1, K*D]

        K = int(first_model.args.num_token)
        D = int(first_model.embed_dim)
        tokens = p.view(1, K, D)   # [1,K,D]
        if not (0 <= token_idx < K):
            raise ValueError(f"token_idx must be in [0,{K-1}]")

        # --- build inputs_embeds for this sample using the SAME idea as first_model.forward ---
        # text embeddings table (frozen)
        text_embeds = first_model.embed_tokens[input_ids[b]]  # [T,D] works in your pipeline
        text_embeds = text_embeds.unsqueeze(0)                # [1,T,D]

        # insert graph tokens into is_node positions
        inputs_embeds = insert_graph_embeds_no_inplace(
            text_embeds=text_embeds,
            is_node=is_node[b].unsqueeze(0),
            node_embeds=tokens,   # [1,K,D]
        )

        # find the token positions for graph slots
        pos = torch.nonzero(is_node[b], as_tuple=False).squeeze(-1)  # [K]
        if pos.numel() != K:
            raise ValueError(f"Sample {b}: expected {K} graph slots, got {pos.numel()}")
        token_pos = int(pos[token_idx].item())

        # --- LLM forward (no labels) ---
        out = llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask[b].unsqueeze(0),
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states
        if hs is None:
            raise ValueError("hidden_states is None; set output_hidden_states=True")

        h = hs[layer_idx]          # [1,T,Hdim]
        h_tok = h[0, token_pos, :] # [Hdim]

        # scalar score: L2 norm of hidden state at this graph token position
        score = torch.norm(h_tok.to(torch.float32), p=2)

        # grads w.r.t. x
        grad_x = torch.autograd.grad(
            outputs=score,
            inputs=x,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]  # [N,F]

        node_scores = torch.norm(grad_x.to(torch.float32), p=2, dim=1).detach()  # [N]
        if normalize:
            s = float(node_scores.sum().item())
            if s > 0:
                node_scores = node_scores / s

        # map to global ids if provided
        node_scores_global = None
        if node_sets is not None and b < len(node_sets) and node_sets[b] is not None:
            ns = node_sets[b]
            if isinstance(ns, torch.Tensor):
                ns_list = ns.detach().cpu().tolist()
            else:
                ns_list = list(ns)
            # guard length mismatch
            if len(ns_list) == node_scores.numel():
                node_scores_global = [(int(ns_list[i]), float(node_scores[i].cpu().item())) for i in range(len(ns_list))]

        results.append({
            "node_scores_local": node_scores.detach().cpu(),
            "node_scores_global": node_scores_global,
            "hop_dist": dist.detach().cpu(),
            "token_pos": token_pos,
            "token_idx": token_idx,
            "mask_mode": mask_mode,
            "mask_hop": mask_hop,
            "layer_idx": layer_idx,
        })

    return results


def llm_grad_attribution_sweep_hops(
    *,
    llm_model,
    first_model,
    batch: Dict[str, Any],
    layer_idx: int = -1,
    mask_mode: str = "exact",      # "exact" or "radius"
    hops: Sequence[int] = (0, 1, 2, 3),
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """
    Returns a dict keyed by (token_idx, hop):
      value is list length B of attribution dicts
    hop=0 is treated as no mask.
    """
    out: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}

    for token_idx in range(5):
        for h in hops:
            if h == 0:
                res = llm_graph_token_grad_attribution_with_mask(
                    llm_model=llm_model,
                    first_model=first_model,
                    batch=batch,
                    token_idx=token_idx,
                    layer_idx=layer_idx,
                    mask_mode="none",
                    mask_hop=0,
                )
            else:
                res = llm_graph_token_grad_attribution_with_mask(
                    llm_model=llm_model,
                    first_model=first_model,
                    batch=batch,
                    token_idx=token_idx,
                    layer_idx=layer_idx,
                    mask_mode=mask_mode,
                    mask_hop=h,
                )
            out[(token_idx, h)] = res
    return out


#### plotting #####

def _circle_layout(n: int) -> np.ndarray:
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)

def plot_attr_one_graph(
    *,
    graph,
    hop_dist: torch.Tensor,           # [N]
    node_scores: torch.Tensor,        # [N]
    ax,
    title: str = "",
) -> None:
    N = int(graph.num_nodes)
    pos = _circle_layout(N)

    ei = graph.edge_index.detach().cpu().numpy()
    for u, v in zip(ei[0], ei[1]):
        if 0 <= u < N and 0 <= v < N:
            ax.plot([pos[u,0], pos[v,0]], [pos[u,1], pos[v,1]], linewidth=0.6, alpha=0.25)

    hop = hop_dist.detach().cpu().numpy()
    sc = ax.scatter(pos[:,0], pos[:,1], c=hop, s=60, alpha=0.9)
    # importance as size overlay
    s = node_scores.detach().cpu().numpy().astype(np.float32)
    s = s / (s.max() + 1e-12)
    ax.scatter(pos[:,0], pos[:,1], s=30 + 300*s, alpha=0.35)

    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return sc

def save_llm_mask_panel_for_first_sample(
    *,
    batch_graph,                     # PyG Batch
    sweep: Dict[Tuple[int,int], List[Dict[str, Any]]],
    save_path: str,
    token_indices: Sequence[int] = (0,1,2,3,4),
    hops: Sequence[int] = (0,1,2,3),
    dpi: int = 150,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    graphs = batch_graph.to_data_list()
    g0 = graphs[0]

    nrows = len(token_indices)
    ncols = len(hops)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 2.6*nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    if ncols == 1:
        axes = np.expand_dims(axes, 1)

    last_sc = None
    for r, k in enumerate(token_indices):
        for c, h in enumerate(hops):
            ax = axes[r, c]
            res0 = sweep[(k, h)][0]  # first sample in batch
            sc = plot_attr_one_graph(
                graph=g0,
                hop_dist=res0["hop_dist"],
                node_scores=res0["node_scores_local"],
                ax=ax,
                title=f"Token {k+1} | hop={h}",
            )
            last_sc = sc

    # colorbar for hop distance
    if last_sc is not None:
        cb = fig.colorbar(last_sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        cb.set_label("Hop distance")

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def _to_numpy_1d(x: Union[torch.Tensor, np.ndarray, Sequence[float]]) -> np.ndarray:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        # handle bf16 etc.
        x = x.detach().to(torch.float32).cpu().numpy()
    else:
        x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        x = x.reshape(-1)
    return x


def _build_positions(num_nodes: int, edge_index: Union[torch.Tensor, np.ndarray]) -> Dict[int, Tuple[float, float]]:
    """
    Returns a dict: node_id -> (x,y).
    Prefers a deterministic spring layout if networkx exists, otherwise a circle.
    """
    if isinstance(edge_index, torch.Tensor):
        ei = edge_index.detach().cpu().numpy()
    else:
        ei = np.asarray(edge_index)
    if ei.size == 0:
        # no edges: circle
        theta = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
        return {i: (float(np.cos(t)), float(np.sin(t))) for i, t in enumerate(theta)}

    # ei shape should be [2, E]
    if ei.shape[0] != 2:
        ei = ei.T
        if ei.shape[0] != 2:
            raise ValueError(f"edge_index must be shape [2,E] or [E,2], got {ei.shape}")

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    src = ei[0].tolist()
    dst = ei[1].tolist()
    edges = list(zip(src, dst))
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=0)  # deterministic
    return {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}

    # fallback: circle
    theta = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    return {i: (float(np.cos(t)), float(np.sin(t))) for i, t in enumerate(theta)}


def _extract_graph_from_batch(
    batch_graph,
    example_idx: int,
):
    """
    Supports either:
      - a PyG Batch with .to_data_list()
      - a list of PyG Data objects
      - a single PyG Data (example_idx must be 0)
    """
    if hasattr(batch_graph, "to_data_list"):
        glist = batch_graph.to_data_list()
        return glist[example_idx]
    if isinstance(batch_graph, (list, tuple)):
        return batch_graph[example_idx]
    if example_idx != 0:
        raise ValueError("batch_graph is a single graph, but example_idx != 0")
    return batch_graph


def _get_attr_from_sweep_leaf(leaf: Any) -> Optional[np.ndarray]:
    """
    A leaf is expected to be either:
      - dict with key 'node_scores_local' (Tensor [N])
      - Tensor [N]
      - ndarray [N]
    """
    if leaf is None:
        return None
    if isinstance(leaf, dict):
        if "node_scores_local" in leaf:
            return _to_numpy_1d(leaf["node_scores_local"])
        # common alternative keys (just in case)
        for k in ["node_scores", "scores", "attr", "attribution"]:
            if k in leaf:
                return _to_numpy_1d(leaf[k])
        return None
    return _to_numpy_1d(leaf)


def _get_leaf(sweep: Any, *, k: int, hop: int, b: int) -> Any:
    """
    Tries a few common sweep layouts.
    You can adjust this if your sweep is structured differently.

    Supported patterns:
      1) sweep[k][hop][b]
      2) sweep[hop][k][b]
      3) sweep[(k, hop)][b]
      4) sweep[(hop, k)][b]
      5) sweep["k"][k]["hop"][hop][b]  (unlikely, but harmless to try)
    """
    # 1) sweep[k][hop][b]
    try:
        return sweep[k][hop][b]
    except Exception:
        pass
    # 2) sweep[hop][k][b]
    try:
        return sweep[hop][k][b]
    except Exception:
        pass
    # 3) sweep[(k,hop)][b]
    try:
        return sweep[(k, hop)][b]
    except Exception:
        pass
    # 4) sweep[(hop,k)][b]
    try:
        return sweep[(hop, k)][b]
    except Exception:
        pass
    # 5) dict-ish nested
    try:
        return sweep["k"][k]["hop"][hop][b]
    except Exception:
        pass

    raise ValueError(
        "Could not index into sweep. Expected one of: "
        "sweep[k][hop][b], sweep[hop][k][b], sweep[(k,hop)][b], sweep[(hop,k)][b]. "
        f"Got type={type(sweep)}"
    )


@torch.no_grad()
def save_attr_hop_grid(
    *,
    batch_graph,                         # PyG Batch or list[Data] or Data
    sweep,                               # output of llm_grad_attribution_sweep_hops(...)
    save_path: str,
    hops: Sequence[int] = (0, 1, 2, 3),
    expected_k: int = 5,
    example_idx: int = 0,
    k_indices: Optional[Sequence[int]] = None,    # default: 0..K-1
    node_sets: Optional[Sequence[torch.Tensor]] = None,  # optional mapping local->global ids (not required for plot)
    title_prefix: str = "",
    dpi: int = 150,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotate_hops: bool = True,           # show hop number next to each node
) -> None:
    """
    Creates a grid of subplots where:
      - Rows = hop settings (0/1/2/3)
      - Cols = graph tokens k (0..4)
      - Node color = attribution score (node_scores_local)
      - Hop distance is shown as small integer text on nodes (optional)

    This fixes the issue you saw: previously hop panels were not showing attribution.
    Now attribution is the colormap + colorbar.

    Output: one PNG at `save_path`.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    g = _extract_graph_from_batch(batch_graph, example_idx=example_idx)
    edge_index = getattr(g, "edge_index", None)
    if edge_index is None:
        raise ValueError("graph has no edge_index")
    num_nodes = int(g.num_nodes) if hasattr(g, "num_nodes") else int(g.x.size(0))

    # determine which k's to plot
    if k_indices is None:
        k_indices = list(range(expected_k))
    K_plot = len(k_indices)
    H_plot = len(hops)

    # Build node positions once (same for all panels)
    pos = _build_positions(num_nodes, edge_index)
    xs = np.array([pos[i][0] for i in range(num_nodes)], dtype=np.float32)
    ys = np.array([pos[i][1] for i in range(num_nodes)], dtype=np.float32)

    # Edge list for drawing
    if isinstance(edge_index, torch.Tensor):
        ei = edge_index.detach().cpu().numpy()
    else:
        ei = np.asarray(edge_index)
    if ei.shape[0] != 2:
        ei = ei.T
    src_list = ei[0].tolist() if ei.size else []
    dst_list = ei[1].tolist() if ei.size else []

    # Pre-collect all attrs to set a consistent color scale
    attrs_grid: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
    for hi, hop in enumerate(hops):
        for ki, k in enumerate(k_indices):
            leaf = _get_leaf(sweep, k=k, hop=hop, b=example_idx)
            attr = _get_attr_from_sweep_leaf(leaf)
            if attr is not None and attr.shape[0] != num_nodes:
                raise ValueError(
                    f"attr length mismatch: got {attr.shape[0]} but graph has {num_nodes} nodes "
                    f"(k={k}, hop={hop}, ex={example_idx})."
                )
            attrs_grid[(hop, k)] = attr

    # vmin/vmax
    all_vals = []
    for key, a in attrs_grid.items():
        if a is not None:
            all_vals.append(a)
    if len(all_vals) == 0:
        raise ValueError("No attribution vectors found in sweep for this example.")
    all_concat = np.concatenate(all_vals, axis=0)
    if vmin is None:
        vmin = float(np.min(all_concat))
    if vmax is None:
        vmax = float(np.max(all_concat))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    # Make figure
    fig_w = max(10, 3.4 * K_plot)
    fig_h = max(8, 3.2 * H_plot)
    fig, axes = plt.subplots(H_plot, K_plot, figsize=(fig_w, fig_h), squeeze=False)

    # helper: hop labels (requires distance computation; we can do a quick BFS on CPU)
    hop_labels = None
    if annotate_hops:
        # compute hop distance from node 0 (central node) in the local subgraph convention
        # If your "center" is not local 0, change `center = 0` here.
        center = 0
        # adjacency
        adj = [[] for _ in range(num_nodes)]
        for u, v in zip(src_list, dst_list):
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                adj[u].append(v)
                adj[v].append(u)
        dist = [-1] * num_nodes
        dist[center] = 0
        q = [center]
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        hop_labels = dist  # list[int]

    mappable = None

    for r, hop in enumerate(hops):
        for c, k in enumerate(k_indices):
            ax = axes[r][c]
            ax.set_axis_off()

            # Draw edges
            for u, v in zip(src_list, dst_list):
                ax.plot([xs[u], xs[v]], [ys[u], ys[v]], linewidth=0.8, alpha=0.35)

            attr = attrs_grid[(hop, k)]
            # Node colors: attribution
            sc = ax.scatter(xs, ys, c=attr, s=120, vmin=vmin, vmax=vmax)
            mappable = sc  # last one, for colorbar

            if annotate_hops and hop_labels is not None:
                for i in range(num_nodes):
                    ax.text(xs[i], ys[i], str(hop_labels[i]), fontsize=8, ha="center", va="center")

            # Title per panel
            ax.set_title(f"hop={hop}, token{k+1}", fontsize=10)

    # Colorbar
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Attribution (node_scores_local)", rotation=90)

    # Global title
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=12)
        fig.subplots_adjust(top=0.93)
    else:
        fig.subplots_adjust(top=0.96)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)