import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch_geometric
from torch_geometric.data import Data

def visualize_culprits_on_topology(saliency_path, edge_path, batch_path, step_idx, save_dir):
    """
    Visualizes the graph topology, highlighting the 'Culprit Nodes' that caused the sink.
    - Target Node (Index 0): Blue Square (The paper being classified)
    - Culprit Nodes (High Gradient): Red Circles (Size = Magnitude)
    - Other Nodes: Grey dots (Low Gradient)
    """
    # 1. Load Data
    saliency = np.load(saliency_path)
    edge_index = np.load(edge_path)
    
    # Handle Batching (Plot First Graph Only)
    if os.path.exists(batch_path):
        batch_idx = np.load(batch_path)
        mask = (batch_idx == 0) # Filter for first sample in batch
        
        node_indices = np.where(mask)[0]
        saliency = saliency[mask]
        
        # Filter edges for graph 0
        u, v = edge_index[0], edge_index[1]
        edge_mask = np.isin(u, node_indices) & np.isin(v, node_indices)
        
        # Remap indices to 0..N local range
        global_to_local = {g: l for l, g in enumerate(node_indices)}
        u_local = [global_to_local[n] for n in u[edge_mask]]
        v_local = [global_to_local[n] for n in v[edge_mask]]
        edge_index = np.array([u_local, v_local])

    # 2. Setup Graph for NetworkX
    data = Data(edge_index=torch.tensor(edge_index, dtype=torch.long), num_nodes=len(saliency))
    G = torch_geometric.utils.to_networkx(data, to_undirected=True)
    
    # 3. Identify Roles
    # In TEA-GLM/PyG data, Index 0 is typically the Target/Center node
    target_node = 0 
    
    # Normalize Saliency for sizing (0.0 to 1.0)
    if saliency.max() > 0:
        norm_saliency = saliency / saliency.max()
    else:
        norm_saliency = saliency

    # 4. Plotting
    plt.figure(figsize=(12, 12))
    
    # Layout: Spring layout puts connected nodes closer together
    pos = nx.spring_layout(G, seed=42, k=0.3) 
    
    # A. Draw Edges (Background)
    nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color='gray')
    
    # B. Draw "Other" Nodes (Low Saliency)
    # Filter nodes that are NOT target AND have low saliency (< 20% of max)
    low_sal_mask = (norm_saliency < 0.2) & (np.arange(len(saliency)) != target_node)
    low_nodes = np.where(low_sal_mask)[0]
    nx.draw_networkx_nodes(G, pos, nodelist=low_nodes.tolist(), 
                           node_color='#D3D3D3', node_size=50, alpha=0.6, label='Neighbors')

    # C. Draw "Culprit" Nodes (High Saliency)
    # Nodes with > 20% of max saliency
    high_sal_mask = (norm_saliency >= 0.2) & (np.arange(len(saliency)) != target_node)
    high_nodes = np.where(high_sal_mask)[0]
    
    if len(high_nodes) > 0:
        # Scale size by importance (Big Red Circles)
        sizes = [500 * norm_saliency[n] + 100 for n in high_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=high_nodes.tolist(), 
                               node_color='red', node_size=sizes, alpha=0.9, label='Sink Contributors')
        
        # Add labels for top culprits
        # We label the top 5 culprits
        top_indices = saliency.argsort()[::-1][:5] 
        labels = {n: str(n) for n in top_indices if n != target_node}
        nx.draw_networkx_labels(G, pos, labels=labels, font_color='white', font_size=10, font_weight='bold')

    # D. Draw Target Node (Special Blue Square)
    nx.draw_networkx_nodes(G, pos, nodelist=[target_node], 
                           node_color='blue', node_size=600, label='Target Node (0)')
    nx.draw_networkx_labels(G, pos, labels={target_node: '0'}, font_color='white', font_weight='bold')

    plt.legend(scatterpoints=1)
    plt.title(f"Step {step_idx}: Sink Contributors (Red) vs Target (Blue)", fontsize=16)
    plt.axis('off')
    
    # 5. Save
    save_path = os.path.join(save_dir, f"topology_sink_step_{step_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved topology check to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset", type=str, default="arxiv", help="Dataset name folder")
    args = parser.parse_args()

    result_dir = f"./analysis/{args.test_dataset}"
    vis_dir = f"./analysis/{args.test_dataset}/visualizations"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        
    # Process first 10 steps
    for i in range(10):
        # Note: We are loading 'deep_sink' files now (from Layer 31 trace)
        saliency_file = os.path.join(result_dir, f"deep_sink_step_{i}.npy")
        edge_file = os.path.join(result_dir, f"edges_step_{i}.npy")
        batch_file = os.path.join(result_dir, f"batch_idx_step_{i}.npy")
        
        if os.path.exists(saliency_file) and os.path.exists(edge_file):
            print(f"Visualizing Topology for Step {i}...")
            visualize_culprits_on_topology(saliency_file, edge_file, batch_file, i, vis_dir)