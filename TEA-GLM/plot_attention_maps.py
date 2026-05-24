import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch_geometric
from torch_geometric.data import Data

def visualize_standalone(saliency_path, edge_path, batch_path, step_idx, save_dir):
    """
    Visualizes graph attention maps using ONLY saved .npy files.
    No model or dataset loading required.
    """
    # 1. Load Data
    saliency = np.load(saliency_path)
    edge_index = np.load(edge_path)
    
    # Handle Batching
    if os.path.exists(batch_path):
        batch_idx = np.load(batch_path)
        # Mask for the first graph (index 0)
        mask = (batch_idx == 0)
        
        # Filter nodes and scores
        node_indices = np.where(mask)[0]
        saliency = saliency[mask]
        
        # Filter edges: Keep edges where BOTH ends are in graph 0
        u, v = edge_index[0], edge_index[1]
        edge_mask = np.isin(u, node_indices) & np.isin(v, node_indices)
        
        # Adjust edge indices to start at 0 for the new smaller graph
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(node_indices)}
        u_local = [global_to_local[n] for n in u[edge_mask]]
        v_local = [global_to_local[n] for n in v[edge_mask]]
        
        edge_index = np.array([u_local, v_local])
    
    # 2. Construct NetworkX Graph
    data = Data(edge_index=torch.tensor(edge_index, dtype=torch.long), num_nodes=len(saliency))
    G = torch_geometric.utils.to_networkx(data, to_undirected=True)
    
    # 3. Normalize Scores (0 to 1)
    if saliency.max() > saliency.min():
        scores_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    else:
        scores_norm = saliency

    # --- FIX: Explicitly create Figure and Axes ---
    fig, ax = plt.subplots(figsize=(10, 10))
    
    pos = nx.spring_layout(G, seed=42, k=0.15)  # Force-directed layout
    
    # Draw Nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_color=scores_norm, 
        cmap=plt.cm.coolwarm, 
        node_size=150, 
        alpha=0.9,
        ax=ax  # <--- Pass the axes here
    )
    
    # Draw Edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray', ax=ax) # <--- And here
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    
    # <--- FIX: Pass 'ax=ax' so matplotlib knows where to put it
    plt.colorbar(sm, ax=ax, label='Gradient Attention Score', fraction=0.046, pad=0.04)
    
    ax.set_title(f"Step {step_idx} - Graph Attention Map", fontsize=14)
    ax.axis('off')
    
    save_path = os.path.join(save_dir, f"vis_step_{step_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # Close specific figure to avoid memory leaks
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset", type=str, default="arxiv", help="Dataset name folder")
    args = parser.parse_args()

    result_dir = f"./analysis/{args.test_dataset}"
    vis_dir = f"./analysis/{args.test_dataset}/visualizations"
    
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        
    # Loop through saved steps
    for i in range(10): # Check first 10 steps
        saliency_file = os.path.join(result_dir, f"saliency_step_{i}.npy")
        edge_file = os.path.join(result_dir, f"edges_step_{i}.npy")
        batch_file = os.path.join(result_dir, f"batch_idx_step_{i}.npy")
        
        if os.path.exists(saliency_file) and os.path.exists(edge_file):
            print(f"Processing Step {i}...")
            visualize_standalone(saliency_file, edge_file, batch_file, i, vis_dir)
        else:
            # Stop if files don't exist
            if i > 0: break