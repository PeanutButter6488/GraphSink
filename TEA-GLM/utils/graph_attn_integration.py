"""
Integration module for graph attention redistribution in TEA-GLM.
Provides high-level API to enable/disable redistribution during training.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from .solution_utils import GraphAttnRedistributor


class GraphAttnRedistributionManager:
    """
    Manages the setup and lifecycle of attention redistribution for graph tokens.
    """
    
    def __init__(self, model: nn.Module, num_layers: int):
        """
        Args:
            model: InstructGLM model instance
            num_layers: Number of transformer layers
        """
        self.model = model
        self.num_layers = num_layers
        self.is_enabled = False
        self.current_batch_graph_indices = None
        self.current_batch_selected_heads = None
    
    def setup_for_batch(
        self,
        is_node: torch.Tensor,  # [B, T] or [T]
        selected_heads: Optional[Dict[int, List[int]]] = None,
        p: float = 0.6,
        sink_token_pos: int = 2,
    ):
        """
        Configure redistribution for the current batch.
        
        Args:
            is_node: Boolean tensor marking which tokens are graph nodes
            selected_heads: Dict[layer_idx -> List[head_ids]]. If provided, only these heads
                           are modified. If None, all heads are modified.
            p: Redistribution factor (default 0.6)
            sink_token_pos: 0-based position of sink token within 5 graph tokens (default 2)
        """
        device = is_node.device
        
        # Extract graph token indices from is_node
        # For simplicity, we assume is_node marks the graph tokens
        if is_node.dim() == 2:
            # [B, T] - use first batch
            node_positions = torch.where(is_node[0])[0]
        else:
            # [T]
            node_positions = torch.where(is_node)[0]
        
        # Expect exactly 5 graph tokens
        if node_positions.numel() < 5:
            print(f"Warning: Expected 5 graph tokens, found {node_positions.numel()}")
            return False
        
        # Use first 5 graph tokens
        graph_indices = node_positions[:5].to(device)
        
        # Configure GraphAttnRedistributor
        GraphAttnRedistributor.enable(p=p, sink_token_pos=sink_token_pos)
        GraphAttnRedistributor.set_graph_indices(graph_indices)
        
        if selected_heads is not None:
            GraphAttnRedistributor.set_selected_heads(selected_heads)
        
        self.current_batch_graph_indices = graph_indices
        self.current_batch_selected_heads = selected_heads
        
        return True
    
    def enable(self):
        """Enable attention redistribution."""
        GraphAttnRedistributor.enabled = True
        self.is_enabled = True
        
        # Create wrapper function for the hook
        def redistribute_wrapper(attn_weights, layer_idx):
            return GraphAttnRedistributor.redistribute_attention(
                attn_weights,
                layer_idx,
            )
        
        # Attach hooks to model
        self.model.enable_graph_attn_redistribution(
            redistribute_wrapper,
            layer_indices=None  # Apply to all layers
        )
    
    def disable(self):
        """Disable attention redistribution."""
        GraphAttnRedistributor.enabled = False
        self.is_enabled = False
        self.model.disable_graph_attn_redistribution()
    
    def cleanup(self):
        """Remove all hooks and clean up."""
        GraphAttnRedistributor.enabled = False
        self.model.remove_graph_attn_hooks()
        self.current_batch_graph_indices = None
        self.current_batch_selected_heads = None


def create_redistribution_manager_from_stats(
    model: nn.Module,
    num_layers: int,
    stats: Dict[str, Any],
    den_all_graph_threshold: float = 2.0,
    sink_ratio_threshold: float = 0.2,
    p: float = 0.6,
    sink_token_pos: int = 2,
) -> GraphAttnRedistributionManager:
    """
    Create and configure a redistribution manager from attention statistics.
    
    Automatically identifies which heads should have redistribution applied
    based on the statistics (graph-centric heads with low sink ratio).
    
    Args:
        model: InstructGLM instance
        num_layers: Number of layers
        stats: Dict with 'den_all_graph' and 'sink_ratio' keys
               Each shape [L, H] or [B, L, H]
        den_all_graph_threshold: Heads must have >= this much attention to graph tokens
        sink_ratio_threshold: Heads must have <= this ratio of sink attention
        p: Redistribution factor
        sink_token_pos: 0-based sink token position within 5 graph tokens
    
    Returns:
        Configured GraphAttnRedistributionManager
    """
    from .solution_utils import select_graph_centric_heads
    
    manager = GraphAttnRedistributionManager(model, num_layers)
    
    # Select heads that should have redistribution applied
    head_selection = select_graph_centric_heads(
        stats,
        den_all_graph_threshold=den_all_graph_threshold,
        sink_ratio_threshold=sink_ratio_threshold,
    )
    
    # Convert selected_indices to the format needed
    # If stats is [B, L, H], selected_indices is List[List[List[int]]]
    # We use the first batch's selection
    if isinstance(head_selection["selected_indices"], list):
        if len(head_selection["selected_indices"]) > 0:
            selected_heads = {}
            batch_0_selection = head_selection["selected_indices"][0]
            for layer_idx, head_list in enumerate(batch_0_selection):
                if len(head_list) > 0:
                    selected_heads[layer_idx] = head_list
    else:
        selected_heads = None
    
    # Store in manager but don't enable yet
    # (will be enabled per batch after is_node is provided)
    manager.current_batch_selected_heads = selected_heads
    manager.p = p
    manager.sink_token_pos = sink_token_pos
    
    return manager
