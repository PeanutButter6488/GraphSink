"""
Attention Sink Analysis Script for Graph-LLM Domain

This script analyzes attention patterns and hidden state activations to identify
"attention sinks" - tokens that receive consistently high attention regardless of
the query, similar to the visual attention sink phenomenon in vision transformers.

The analysis process:
1. Load attention-received maps across evaluation steps
2. Identify tokens with consistently high attention (potential sinks)
3. Separate graph tokens from text tokens
4. For sink tokens, examine hidden state dimensions for abnormal activation
5. Generate visualizations and reports
"""

import numpy as np
import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionSinkAnalyzer:
    def __init__(self, data_dir='./analysis/attn_maps', output_dir='./analysis/sink_analysis'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.attn_received_list = []
        self.hidden_states_list = []
        self.summaries = []
        self.token_ids_list = []
        self.is_node_mask_list = []
        
    def load_data(self):
        """Load all saved attention, hidden state, and summary files."""
        print("Loading saved attention data...")
        
        # Get all step indices
        summary_files = sorted(self.data_dir.glob('attn_summary_step*.npz'))
        step_indices = sorted(set(
            int(f.stem.split('step')[1]) 
            for f in summary_files
        ))
        
        for step_idx in step_indices:
            # Load attention received
            attn_rec_file = self.data_dir / f'attn_received_layer16_step{step_idx}.npy'
            if attn_rec_file.exists():
                self.attn_received_list.append(np.load(attn_rec_file))  # (B, seq_len)
            
            # Load hidden states
            hidden_file = self.data_dir / f'hidden_layer16_step{step_idx}.npy'
            if hidden_file.exists():
                self.hidden_states_list.append(np.load(hidden_file))  # (B, seq_len, hidden_dim)
            
            # Load summary (contains token_ids, is_node_mask, etc.)
            summary_file = self.data_dir / f'attn_summary_step{step_idx}.npz'
            if summary_file.exists():
                summary_data = np.load(summary_file, allow_pickle=True)
                self.summaries.append(summary_data)
                self.token_ids_list.append(summary_data['token_ids'])
                self.is_node_mask_list.append(summary_data['is_node_mask'])
        
        print(f"  Loaded {len(self.attn_received_list)} evaluation steps")
        print(f"  Attention received shapes: {[a.shape for a in self.attn_received_list[:3]]}")
        if self.hidden_states_list:
            print(f"  Hidden states shapes: {[h.shape for h in self.hidden_states_list[:3]]}")
    
    def identify_sink_tokens(self, percentile=90, min_consistent_steps=5):
        """
        Identify tokens that consistently receive high attention.
        
        Args:
            percentile: Attention threshold (e.g., 90th percentile = top 10%)
            min_consistent_steps: Minimum steps where token must be a sink
        
        Returns:
            Dictionary mapping (step_idx, token_pos) -> sink_info
        """
        print(f"\nIdentifying attention sinks (>{percentile}th percentile)...")
        
        sink_candidates = defaultdict(int)  # (step_idx, token_pos) -> count
        sink_details = {}
        
        for step_idx, attn_rec in enumerate(self.attn_received_list):
            # attn_rec shape: (B, seq_len) - we'll use first batch sample
            batch_attn = attn_rec[0]  # (seq_len,)
            
            threshold = np.percentile(batch_attn, percentile)
            sink_positions = np.where(batch_attn > threshold)[0]
            
            for pos in sink_positions:
                key = (step_idx, int(pos))
                sink_candidates[key] += 1
                
                # Store details
                if key not in sink_details:
                    sink_details[key] = {
                        'attn_received': float(batch_attn[pos]),
                        'is_graph_token': bool(self.is_node_mask_list[step_idx][pos]),
                        'token_id': int(self.token_ids_list[step_idx][pos]),
                        'step_idx': step_idx,
                        'position': int(pos),
                    }
        
        print(f"  Found {len(sink_candidates)} unique sink candidates")
        print(f"  Threshold (90th percentile): {np.mean([np.percentile(a[0], percentile) for a in self.attn_received_list]):.4f}")
        
        return sink_details
    
    def analyze_sink_hidden_dimensions(self, sink_details, top_k=20):
        """
        For identified sink tokens, find which hidden dimensions have abnormally high activation.
        
        This mimics the visual attention sink approach: look at the hidden state vectors
        of sink tokens and find dimensions that spike.
        """
        print(f"\nAnalyzing hidden state dimensions for {len(sink_details)} sinks...")
        
        sink_dimensions = defaultdict(lambda: {'graph': [], 'text': []})
        
        for (step_idx, token_pos), sink_info in sink_details.items():
            if step_idx >= len(self.hidden_states_list):
                continue
            
            hidden = self.hidden_states_list[step_idx]  # (B, seq_len, hidden_dim)
            batch_hidden = hidden[0]  # (seq_len, hidden_dim)
            
            if token_pos >= batch_hidden.shape[0]:
                continue
            
            # Get hidden state for this sink token
            sink_hidden = batch_hidden[token_pos]  # (hidden_dim,)
            
            # Compute "spike score" per dimension
            # Method: abs(dimension_value) / RMS(all_dimensions)
            rms = np.sqrt(np.mean(sink_hidden ** 2))
            if rms < 1e-6:
                continue
            
            spike_scores = np.abs(sink_hidden) / rms  # (hidden_dim,)
            
            # Find top spiking dimensions
            top_dims = np.argsort(spike_scores)[-top_k:][::-1]
            top_scores = spike_scores[top_dims]
            
            # Categorize by token type
            token_type = 'graph' if sink_info['is_graph_token'] else 'text'
            
            sink_dimensions[step_idx][token_type].append({
                'position': token_pos,
                'top_dims': top_dims.tolist(),
                'top_scores': top_scores.tolist(),
                'attn_received': sink_info['attn_received'],
                'token_id': sink_info['token_id'],
            })
        
        return sink_dimensions
    
    def compute_aggregate_statistics(self, sink_details, sink_dimensions):
        """Aggregate statistics across all steps and sinks."""
        print("\nComputing aggregate statistics...")
        
        # 1. Count sinks by type
        graph_sinks = sum(1 for s in sink_details.values() if s['is_graph_token'])
        text_sinks = len(sink_details) - graph_sinks
        
        # 2. Collect all top spiking dimensions across all sinks
        all_graph_dims = defaultdict(int)
        all_text_dims = defaultdict(int)
        
        for step_idx, dim_info in sink_dimensions.items():
            for sink in dim_info['graph']:
                for dim, score in zip(sink['top_dims'], sink['top_scores']):
                    all_graph_dims[dim] += 1
            for sink in dim_info['text']:
                for dim, score in zip(sink['top_dims'], sink['top_scores']):
                    all_text_dims[dim] += 1
        
        # Sort by frequency
        top_graph_dims = sorted(all_graph_dims.items(), key=lambda x: x[1], reverse=True)[:20]
        top_text_dims = sorted(all_text_dims.items(), key=lambda x: x[1], reverse=True)[:20]
        
        stats = {
            'total_sinks': len(sink_details),
            'graph_sinks': graph_sinks,
            'text_sinks': text_sinks,
            'top_graph_dims': top_graph_dims,
            'top_text_dims': top_text_dims,
            'total_unique_dims_graph': len(all_graph_dims),
            'total_unique_dims_text': len(all_text_dims),
        }
        
        return stats
    
    def generate_report(self, sink_details, sink_dimensions, stats):
        """Generate a human-readable report."""
        print("\n" + "="*80)
        print("ATTENTION SINK ANALYSIS REPORT")
        print("="*80)
        
        print(f"\n[1] SINK STATISTICS")
        print(f"  Total sink tokens identified: {stats['total_sinks']}")
        print(f"  Graph tokens acting as sinks: {stats['graph_sinks']}")
        print(f"  Text tokens acting as sinks: {stats['text_sinks']}")
        print(f"  Ratio (Graph/Total): {stats['graph_sinks']/stats['total_sinks']:.1%}")
        
        print(f"\n[2] TOP SPIKING DIMENSIONS (Hidden State)")
        print(f"\n  Graph Tokens - Top 20 Dimensions (by frequency across sinks):")
        for rank, (dim_idx, freq) in enumerate(stats['top_graph_dims'][:10], 1):
            print(f"    {rank:2d}. Dimension {dim_idx:4d} (appears in {freq:3d} sinks)")
        
        print(f"\n  Text Tokens - Top 20 Dimensions (by frequency across sinks):")
        for rank, (dim_idx, freq) in enumerate(stats['top_text_dims'][:10], 1):
            print(f"    {rank:2d}. Dimension {dim_idx:4d} (appears in {freq:3d} sinks)")
        
        print(f"\n[3] DIMENSION DIVERSITY")
        print(f"  Unique dimensions spiking in graph sinks: {stats['total_unique_dims_graph']}")
        print(f"  Unique dimensions spiking in text sinks: {stats['total_unique_dims_text']}")
        
        # Sample some sink examples
        print(f"\n[4] EXAMPLE SINKS (first 5)")
        for (step_idx, pos), info in list(sink_details.items())[:5]:
            token_type = "GRAPH" if info['is_graph_token'] else "TEXT"
            print(f"  Step {step_idx}, Position {pos}: {token_type} token")
            print(f"    Attention Received: {info['attn_received']:.4f}")
            if step_idx in sink_dimensions and len(sink_dimensions[step_idx][token_type.lower()]) > 0:
                example_sink = sink_dimensions[step_idx][token_type.lower()][0]
                top_3_dims = example_sink['top_dims'][:3]
                print(f"    Top 3 Spiking Dims: {top_3_dims}")
        
        print("\n" + "="*80)
    
    def save_results(self, sink_details, sink_dimensions, stats):
        """Save analysis results to JSON/NPZ files."""
        print(f"\nSaving results to {self.output_dir}...")
        
        # Save statistics
        stats_json = {
            'total_sinks': int(stats['total_sinks']),
            'graph_sinks': int(stats['graph_sinks']),
            'text_sinks': int(stats['text_sinks']),
            'top_graph_dims': [[int(d), int(f)] for d, f in stats['top_graph_dims']],
            'top_text_dims': [[int(d), int(f)] for d, f in stats['top_text_dims']],
            'total_unique_dims_graph': int(stats['total_unique_dims_graph']),
            'total_unique_dims_text': int(stats['total_unique_dims_text']),
        }
        with open(self.output_dir / 'sink_statistics.json', 'w') as f:
            json.dump(stats_json, f, indent=2)
        
        # Save sink details
        sink_details_json = {}
        for (step_idx, pos), info in sink_details.items():
            key = f"step{step_idx}_pos{pos}"
            sink_details_json[key] = {
                'attn_received': float(info['attn_received']),
                'is_graph_token': bool(info['is_graph_token']),
                'token_id': int(info['token_id']),
                'position': int(info['position']),
            }
        with open(self.output_dir / 'sink_details.json', 'w') as f:
            json.dump(sink_details_json, f, indent=2)
        
        # Save top dimensions as NPZ for easy loading
        top_graph_dims = np.array([d for d, f in stats['top_graph_dims']])
        top_graph_freqs = np.array([f for d, f in stats['top_graph_dims']])
        top_text_dims = np.array([d for d, f in stats['top_text_dims']])
        top_text_freqs = np.array([f for d, f in stats['top_text_dims']])
        
        np.savez(
            self.output_dir / 'sink_dimensions.npz',
            top_graph_dims=top_graph_dims,
            top_graph_freqs=top_graph_freqs,
            top_text_dims=top_text_dims,
            top_text_freqs=top_text_freqs,
        )
        
        print(f"  Saved: sink_statistics.json")
        print(f"  Saved: sink_details.json")
        print(f"  Saved: sink_dimensions.npz")
    
    def visualize_results(self, stats):
        """Create visualizations."""
        print("\nGenerating visualizations...")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Sink type distribution
            ax = axes[0, 0]
            labels = ['Graph Tokens', 'Text Tokens']
            sizes = [stats['graph_sinks'], stats['text_sinks']]
            colors = ['#ff9999', '#66b3ff']
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Distribution of Attention Sinks by Token Type')
            
            # Plot 2: Top dimensions for graph tokens
            ax = axes[0, 1]
            dims = [str(d) for d, _ in stats['top_graph_dims'][:15]]
            freqs = [f for _, f in stats['top_graph_dims'][:15]]
            ax.barh(dims, freqs, color='#ff9999')
            ax.set_xlabel('Frequency (# sinks with this dim spiking)')
            ax.set_title('Top 15 Spiking Dimensions in Graph Token Sinks')
            ax.invert_yaxis()
            
            # Plot 3: Top dimensions for text tokens
            ax = axes[1, 0]
            dims = [str(d) for d, _ in stats['top_text_dims'][:15]]
            freqs = [f for _, f in stats['top_text_dims'][:15]]
            ax.barh(dims, freqs, color='#66b3ff')
            ax.set_xlabel('Frequency (# sinks with this dim spiking)')
            ax.set_title('Top 15 Spiking Dimensions in Text Token Sinks')
            ax.invert_yaxis()
            
            # Plot 4: Summary stats
            ax = axes[1, 1]
            ax.axis('off')
            summary_text = f"""
            SUMMARY STATISTICS
            
            Total Sinks Found: {stats['total_sinks']}
            Graph Tokens: {stats['graph_sinks']} ({stats['graph_sinks']/stats['total_sinks']:.1%})
            Text Tokens: {stats['text_sinks']} ({stats['text_sinks']/stats['total_sinks']:.1%})
            
            Unique Spiking Dims (Graph): {stats['total_unique_dims_graph']}
            Unique Spiking Dims (Text): {stats['total_unique_dims_text']}
            
            Analysis Layer: Layer 2 (deep layer)
            Attention Threshold: 90th percentile
            """
            ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plot_path = self.output_dir / 'sink_analysis_summary_layer16.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: sink_analysis_summary.png")
            plt.close()
            
        except ImportError:
            print("  Matplotlib not available, skipping visualization")
    
    def run_analysis(self):
        """Execute full analysis pipeline."""
        self.load_data()
        sink_details = self.identify_sink_tokens(percentile=90)
        sink_dimensions = self.analyze_sink_hidden_dimensions(sink_details, top_k=20)
        stats = self.compute_aggregate_statistics(sink_details, sink_dimensions)
        self.generate_report(sink_details, sink_dimensions, stats)
        self.save_results(sink_details, sink_dimensions, stats)
        self.visualize_results(stats)
        
        print(f"\n✓ Analysis complete! Results saved to: {self.output_dir}")


if __name__ == '__main__':
    analyzer = AttentionSinkAnalyzer(
        data_dir='./analysis/attn_maps',
        output_dir='./analysis/sink_analysis'
    )
    analyzer.run_analysis()
