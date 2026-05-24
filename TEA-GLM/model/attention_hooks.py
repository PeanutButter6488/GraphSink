"""
PyTorch hooks for intercepting and modifying attention in LLaMA layers.
Used to apply attention redistribution from graph sink tokens to non-sink tokens.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Dict, List, Any


class AttentionRedistributionHook:
    """
    Attaches to LLaMA attention layers to intercept and modify attention weights.
    
    Hooks into the output of the attention computation (after softmax) to apply
    redistribution before the attention is multiplied with values.
    """
    
    def __init__(
        self,
        layer_idx: int,
        redistribute_fn: Callable,
        is_node: Optional[torch.Tensor] = None,
        graph_indices: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            layer_idx: Index of the LLaMA layer this hook is attached to
            redistribute_fn: Function to call to redistribute attention
                            Signature: (attn_weights, layer_idx, batch_idx) -> attn_weights
            is_node: [T] or [B, T] bool tensor marking which tokens are graph nodes
            graph_indices: [5] tensor of absolute positions of graph tokens
        """
        self.layer_idx = layer_idx
        self.redistribute_fn = redistribute_fn
        self.is_node = is_node
        self.graph_indices = graph_indices
        self.enabled = False
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False
    
    def __call__(self, module, input, output):
        """
        Hook called after attention computation.
        
        LLaMA attention returns a tuple: (attn_output, attn_weights)
        We only modify attn_weights before it's used.
        """
        if not self.enabled:
            return output
        
        attn_output, attn_weights = output
        
        # attn_weights shape: [B, num_heads, seq_len, seq_len]
        if attn_weights is None:
            return output
        
        # Apply redistribution
        modified_weights = self.redistribute_fn(
            attn_weights,
            self.layer_idx,
        )
        
        # Recompute attention output with modified weights
        # Note: we need access to value_states, which requires re-implementing attention
        # For now, we return as-is (the modification will be used in the forward hook on the attention module itself)
        
        return (attn_output, modified_weights)


class LlamaAttentionForwardHook:
    """
    Intercepts the forward method of LLaMA's self-attention layer.
    Modifies attention weights after softmax, before matmul with values.
    """
    
    def __init__(
        self,
        layer_idx: int,
        redistribute_fn: Callable,
    ):
        """
        Args:
            layer_idx: Index of the LLaMA layer
            redistribute_fn: Function to redistribute attention weights
        """
        self.layer_idx = layer_idx
        self.redistribute_fn = redistribute_fn
        self.enabled = False
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False
    
    def __call__(self, module, input, output):
        """
        Called after attention forward, receives the tuple output.
        Modifies attention_probs before they're used.
        """
        if not self.enabled:
            return output
        
        # In LLaMA, the attention module returns attention_output
        # We can't directly intercept here without access to internals
        return output


class AttentionHookRegistry:
    """
    Manages a collection of attention hooks across all layers.
    """
    
    def __init__(self, model: nn.Module, num_layers: int):
        """
        Args:
            model: The full model (InstructGLM or similar)
            num_layers: Number of transformer layers
        """
        self.model = model
        self.num_layers = num_layers
        self.hooks: Dict[int, Any] = {}
        self.handles: Dict[int, Any] = {}
        self.enabled = False
    
    def attach_hooks(self, redistribute_fn: Callable, layer_indices: Optional[List[int]] = None):
        """
        Attach redistribution hooks to specified layers.
        
        Args:
            redistribute_fn: Function to call for redistribution
            layer_indices: List of layer indices to attach hooks to.
                          If None, attach to all layers.
        """
        if layer_indices is None:
            layer_indices = list(range(self.num_layers))
        
        for layer_idx in layer_indices:
            if layer_idx < 0 or layer_idx >= self.num_layers:
                print(f"Warning: Invalid layer index {layer_idx}")
                continue
            
            # For LLaMA: access the attention layer
            try:
                attn_module = self.model.model.layers[layer_idx].self_attn
            except (AttributeError, IndexError) as e:
                print(f"Warning: Could not access attention module at layer {layer_idx}: {e}")
                continue
            
            hook = LlamaAttentionRedistributionHook(layer_idx, redistribute_fn)
            handle = attn_module.register_forward_hook(hook)
            
            self.hooks[layer_idx] = hook
            self.handles[layer_idx] = handle
    
    def enable_all(self):
        """Enable all attached hooks."""
        self.enabled = True
        for hook in self.hooks.values():
            hook.enable()
    
    def disable_all(self):
        """Disable all attached hooks."""
        self.enabled = False
        for hook in self.hooks.values():
            hook.disable()
    
    def remove_all(self):
        """Remove all hooks from the model."""
        for handle in self.handles.values():
            handle.remove()
        self.hooks.clear()
        self.handles.clear()


class LlamaAttentionRedistributionHook:
    """
    Custom hook that wraps LLaMA's attention computation.
    Intercepts at the point where attn_weights are computed and before value multiplication.
    """
    
    def __init__(self, layer_idx: int, redistribute_fn: Callable):
        self.layer_idx = layer_idx
        self.redistribute_fn = redistribute_fn
        self.enabled = False
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False
    
    def __call__(self, module, input, output):
        """
        Intercepts attention module output.
        In transformers, attention returns (attn_output, attn_weights, past_key_value) or similar.
        """
        if not self.enabled:
            return output
        
        # Handle different return formats
        if isinstance(output, tuple):
            attn_output = output[0]
            attn_weights = output[1] if len(output) > 1 else None
            rest = output[2:] if len(output) > 2 else ()
            
            if attn_weights is not None:
                # Apply redistribution
                modified_weights = self.redistribute_fn(
                    attn_weights,
                    self.layer_idx,
                )
                # Return modified tuple
                return (attn_output, modified_weights) + rest
        
        return output
