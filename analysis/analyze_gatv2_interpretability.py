#!/usr/bin/env python
"""
analyze_gatv2_interpretability.py - GATv2 Interpretability Analysis
==================================================================

Analyzes trained GATv2 models to identify:
1. Most important ROIs (brain regions) for prediction
2. Most important edges (connections) and their strengths
3. How specific nodes and edges contribute to predictions

Outputs:
- Excel files with ROI rankings and importance scores
- Excel files with edge rankings and attention weights
- Visualizations of brain network importance
- Statistical analysis of contribution patterns

Usage:
    python analyze_gatv2_interpretability.py --model_dir results_gatv2_improved
    python analyze_gatv2_interpretability.py --model_dir results_gatv2_interpretable --top_k 20
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import model architectures
# Add training directory to path
import sys
training_dir = Path(__file__).parent.parent / "training" / "gatv2"
sys.path.insert(0, str(training_dir))

try:
    from train_gatv2_improved import ImprovedGATv2Regressor
except Exception as e:
    print(f"Warning: Could not import ImprovedGATv2Regressor: {e}")
    ImprovedGATv2Regressor = None

try:
    from train_gatv2_basic import GATv2Regressor
except Exception as e:
    print(f"Warning: Could not import GATv2Regressor: {e}")
    GATv2Regressor = None


# ----------------------------
# 1. Load Model and Data
# ----------------------------

def load_model_and_data(model_dir, fold_name, device='cpu'):
    """
    Load trained model and test data.

    Args:
        model_dir: Directory containing model checkpoint
        fold_name: Name of fold (e.g., 'graphs_outer1_inner1')
        device: Device to load model on

    Returns:
        model, test_graphs, scaler, config
    """
    model_path = Path(model_dir) / fold_name / "gatv2_best.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load checkpoint
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle both old format (state_dict only) and new format (with config, scaler, etc.)
    if 'model_state_dict' in checkpoint:
        # New format
        config = checkpoint.get('config', {})
        scaler = checkpoint.get('scaler', None)
        state_dict = checkpoint['model_state_dict']
    else:
        # Old format - checkpoint IS the state_dict
        config = {}
        scaler = None
        state_dict = checkpoint
        print("Warning: Old checkpoint format detected (no config/scaler)")

    # Load test data
    fold_data_path = Path(f"../../data/folds_data/{fold_name}.pkl")
    if not fold_data_path.exists():
        # Fallback: try from project root
        fold_data_path = Path(f"data/folds_data/{fold_name}.pkl")

    if not fold_data_path.exists():
        print(f"Error: Data file not found: {fold_name}.pkl")
        raise FileNotFoundError(f"Data file not found: {fold_name}.pkl")

    print(f"Loading data from {fold_data_path}")
    graphs_dict = torch.load(fold_data_path, map_location='cpu', weights_only=False)

    test_2d = graphs_dict['test_graphs']

    # Flatten and filter
    test_graphs = []
    for subj_idx, row in enumerate(test_2d):
        for g in row:
            if hasattr(g, "pad") and bool(g.pad):
                continue
            g.subject_id = torch.tensor([subj_idx], dtype=torch.long)
            test_graphs.append(g)

    # Create model
    in_dim = test_graphs[0].x.size(-1)

    # Try different model architectures - try loading weights to verify compatibility
    model = None
    model_name = None

    # Try ImprovedGATv2Regressor first
    if ImprovedGATv2Regressor is not None:
        try:
            temp_model = ImprovedGATv2Regressor(
                in_dim=in_dim,
                hidden_dim=config.get('hidden_dim', 64),
                n_layers=config.get('n_layers', 3),
                n_heads=config.get('n_heads', 4),
                dropout=config.get('dropout', 0.2),
                edge_dropout=config.get('edge_dropout', 0.1),
            )
            temp_model.load_state_dict(state_dict)
            model = temp_model
            model_name = "ImprovedGATv2Regressor"
        except Exception as e:
            print(f"Could not load with ImprovedGATv2Regressor: {type(e).__name__}")

    # Try GATv2Regressor (basic) if improved didn't work
    if model is None and GATv2Regressor is not None:
        try:
            temp_model = GATv2Regressor(
                in_dim=in_dim,
                hidden_dim=config.get('hidden_dim', 32),
            )
            temp_model.load_state_dict(state_dict)
            model = temp_model
            model_name = "GATv2Regressor"
        except Exception as e:
            print(f"Could not load with GATv2Regressor: {type(e).__name__}")

    if model is None:
        raise RuntimeError("Could not load model - no compatible architecture found")

    # Move to device
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Test graphs: {len(test_graphs)}")

    return model, test_graphs, scaler, config


# ----------------------------
# 2. Extract Attention Weights
# ----------------------------

def extract_edge_attention(model, graphs: List[Data], device='cpu'):
    """
    Extract edge attention weights from all test graphs.

    Args:
        model: Trained GATv2 model
        graphs: List of test graphs
        device: Device

    Returns:
        Dict with edge statistics
    """
    print("\nExtracting edge attention weights...")

    # Aggregate attention weights per edge type
    edge_attention_agg = defaultdict(list)
    all_edge_attentions = []

    for graph in graphs:
        try:
            edge_index, alpha = model.get_edge_attention(graph)

            # Store per-edge attention
            for i in range(edge_index.size(1)):
                src = int(edge_index[0, i])
                dst = int(edge_index[1, i])
                att = float(alpha[i])

                # Create edge key (undirected)
                edge_key = tuple(sorted([src, dst]))
                edge_attention_agg[edge_key].append(att)

                all_edge_attentions.append({
                    'src': src,
                    'dst': dst,
                    'attention': att,
                })
        except Exception as e:
            print(f"Warning: Could not extract attention for graph: {e}")
            continue

    # Compute average attention per edge
    edge_stats = {}
    for edge_key, attentions in edge_attention_agg.items():
        edge_stats[edge_key] = {
            'mean_attention': np.mean(attentions),
            'std_attention': np.std(attentions),
            'n_occurrences': len(attentions),
        }

    print(f"Extracted attention for {len(edge_stats)} unique edges")

    return edge_stats, all_edge_attentions


def extract_node_importance(model, graphs: List[Data], device='cpu'):
    """
    Extract node importance scores from all test graphs.

    Args:
        model: Trained GATv2 model
        graphs: List of test graphs
        device: Device

    Returns:
        Dict with node statistics
    """
    print("\nExtracting node importance scores...")

    # Aggregate node importance scores
    node_importance_agg = defaultdict(list)

    for graph in graphs:
        try:
            node_scores = model.get_node_importance(graph)

            for node_idx, score in enumerate(node_scores):
                node_importance_agg[node_idx].append(float(score))
        except Exception as e:
            print(f"Warning: Could not extract node importance: {e}")
            continue

    # Compute average importance per node
    node_stats = {}
    for node_idx, scores in node_importance_agg.items():
        node_stats[node_idx] = {
            'mean_importance': np.mean(scores),
            'std_importance': np.std(scores),
            'n_occurrences': len(scores),
        }

    print(f"Extracted importance for {len(node_stats)} nodes")

    return node_stats


# ----------------------------
# 3. Analyze and Rank
# ----------------------------

def rank_important_nodes(node_stats: Dict, top_k: int = 20) -> pd.DataFrame:
    """
    Rank nodes by importance.

    Args:
        node_stats: Dict with node statistics
        top_k: Number of top nodes to return

    Returns:
        DataFrame with ranked nodes
    """
    # Convert to DataFrame
    data = []
    for node_idx, stats in node_stats.items():
        data.append({
            'ROI_Index': node_idx,
            'Mean_Importance': stats['mean_importance'],
            'Std_Importance': stats['std_importance'],
            'N_Graphs': stats['n_occurrences'],
        })

    df = pd.DataFrame(data)

    # Sort by importance
    df = df.sort_values('Mean_Importance', ascending=False)

    # Add rank
    df['Rank'] = range(1, len(df) + 1)

    # Reorder columns
    df = df[['Rank', 'ROI_Index', 'Mean_Importance', 'Std_Importance', 'N_Graphs']]

    return df.head(top_k)


def rank_important_edges(edge_stats: Dict, top_k: int = 50) -> pd.DataFrame:
    """
    Rank edges by attention weight.

    Args:
        edge_stats: Dict with edge statistics
        top_k: Number of top edges to return

    Returns:
        DataFrame with ranked edges
    """
    # Convert to DataFrame
    data = []
    for edge_key, stats in edge_stats.items():
        src, dst = edge_key
        data.append({
            'ROI_Source': src,
            'ROI_Target': dst,
            'Mean_Attention': stats['mean_attention'],
            'Std_Attention': stats['std_attention'],
            'N_Graphs': stats['n_occurrences'],
        })

    df = pd.DataFrame(data)

    # Sort by attention
    df = df.sort_values('Mean_Attention', ascending=False)

    # Add rank
    df['Rank'] = range(1, len(df) + 1)

    # Reorder columns
    df = df[['Rank', 'ROI_Source', 'ROI_Target', 'Mean_Attention', 'Std_Attention', 'N_Graphs']]

    return df.head(top_k)


def analyze_node_connectivity(edge_stats: Dict, node_stats: Dict) -> pd.DataFrame:
    """
    Analyze how node importance relates to connectivity.

    Returns:
        DataFrame with node connectivity analysis
    """
    # Count edges per node
    node_edge_count = defaultdict(int)
    node_edge_attention = defaultdict(list)

    for edge_key, stats in edge_stats.items():
        src, dst = edge_key
        node_edge_count[src] += stats['n_occurrences']
        node_edge_count[dst] += stats['n_occurrences']

        node_edge_attention[src].append(stats['mean_attention'])
        node_edge_attention[dst].append(stats['mean_attention'])

    # Create DataFrame
    data = []
    for node_idx in node_stats.keys():
        data.append({
            'ROI_Index': node_idx,
            'Node_Importance': node_stats[node_idx]['mean_importance'],
            'Edge_Count': node_edge_count[node_idx],
            'Mean_Edge_Attention': np.mean(node_edge_attention[node_idx]) if node_edge_attention[node_idx] else 0,
        })

    df = pd.DataFrame(data)
    df = df.sort_values('Node_Importance', ascending=False)

    return df


# ----------------------------
# 4. Visualizations
# ----------------------------

def plot_node_importance(node_df: pd.DataFrame, output_path: str):
    """Plot node importance rankings."""
    plt.figure(figsize=(12, 6))

    # Bar plot
    plt.subplot(1, 2, 1)
    top_20 = node_df.head(20)
    plt.barh(range(len(top_20)), top_20['Mean_Importance'])
    plt.yticks(range(len(top_20)), [f"ROI {int(idx)}" for idx in top_20['ROI_Index']])
    plt.xlabel('Mean Importance Score')
    plt.title('Top 20 Most Important ROIs')
    plt.gca().invert_yaxis()

    # Distribution
    plt.subplot(1, 2, 2)
    all_importances = node_df['Mean_Importance'].values
    plt.hist(all_importances, bins=50, edgecolor='black')
    plt.xlabel('Importance Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Node Importance')
    plt.axvline(np.mean(all_importances), color='r', linestyle='--', label='Mean')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved node importance plot: {output_path}")


def plot_edge_attention(edge_df: pd.DataFrame, output_path: str):
    """Plot edge attention rankings."""
    plt.figure(figsize=(12, 6))

    # Bar plot
    plt.subplot(1, 2, 1)
    top_20 = edge_df.head(20)
    plt.barh(range(len(top_20)),
             top_20['Mean_Attention'])
    plt.yticks(range(len(top_20)),
               [f"{int(row['ROI_Source'])}-{int(row['ROI_Target'])}"
                for _, row in top_20.iterrows()])
    plt.xlabel('Mean Attention Weight')
    plt.title('Top 20 Most Important Edges')
    plt.gca().invert_yaxis()

    # Distribution
    plt.subplot(1, 2, 2)
    all_attentions = edge_df['Mean_Attention'].values
    plt.hist(all_attentions, bins=50, edgecolor='black')
    plt.xlabel('Attention Weight')
    plt.ylabel('Frequency')
    plt.title('Distribution of Edge Attention')
    plt.axvline(np.mean(all_attentions), color='r', linestyle='--', label='Mean')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved edge attention plot: {output_path}")


def plot_connectivity_vs_importance(connectivity_df: pd.DataFrame, output_path: str):
    """Plot relationship between node connectivity and importance."""
    plt.figure(figsize=(10, 6))

    plt.scatter(
        connectivity_df['Edge_Count'],
        connectivity_df['Node_Importance'],
        alpha=0.5,
        s=50
    )

    plt.xlabel('Number of Edges')
    plt.ylabel('Node Importance Score')
    plt.title('Node Connectivity vs. Importance')

    # Add correlation
    corr = np.corrcoef(connectivity_df['Edge_Count'],
                      connectivity_df['Node_Importance'])[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved connectivity vs importance plot: {output_path}")


# ----------------------------
# 5. Main Analysis Pipeline
# ----------------------------

def analyze_fold(model_dir: str, fold_name: str, output_dir: str, top_k: int = 20, device: str = 'cpu'):
    """
    Analyze interpretability for a single fold.

    Args:
        model_dir: Directory containing trained models
        fold_name: Name of fold
        output_dir: Output directory for results
        top_k: Number of top nodes/edges to report
        device: Device to use

    Returns:
        Dict with analysis results
    """
    print(f"\n{'='*70}")
    print(f"Analyzing fold: {fold_name}")
    print(f"{'='*70}")

    # Load model and data
    device = torch.device(device)
    model, test_graphs, scaler, config = load_model_and_data(
        model_dir, fold_name, device
    )

    # Extract attention weights
    edge_stats, all_edge_attentions = extract_edge_attention(model, test_graphs, device)

    # Extract node importance
    node_stats = extract_node_importance(model, test_graphs, device)

    # Rank nodes and edges
    top_nodes_df = rank_important_nodes(node_stats, top_k=top_k)
    top_edges_df = rank_important_edges(edge_stats, top_k=top_k * 2)

    # Connectivity analysis
    connectivity_df = analyze_node_connectivity(edge_stats, node_stats)

    # Create output directory
    fold_output_dir = Path(output_dir) / fold_name
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    # Save to Excel
    excel_path = fold_output_dir / f"interpretability_analysis.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        top_nodes_df.to_excel(writer, sheet_name='Top_ROIs', index=False)
        top_edges_df.to_excel(writer, sheet_name='Top_Edges', index=False)
        connectivity_df.to_excel(writer, sheet_name='Connectivity_Analysis', index=False)

        # Summary stats
        summary_data = {
            'Metric': [
                'Total ROIs',
                'Mean Node Importance',
                'Std Node Importance',
                'Total Edges Analyzed',
                'Mean Edge Attention',
                'Std Edge Attention',
            ],
            'Value': [
                len(node_stats),
                np.mean([s['mean_importance'] for s in node_stats.values()]),
                np.std([s['mean_importance'] for s in node_stats.values()]),
                len(edge_stats),
                np.mean([s['mean_attention'] for s in edge_stats.values()]),
                np.std([s['mean_attention'] for s in edge_stats.values()]),
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"Saved Excel report: {excel_path}")

    # Create visualizations
    plot_node_importance(
        connectivity_df[['ROI_Index', 'Node_Importance']].rename(
            columns={'Node_Importance': 'Mean_Importance'}
        ),
        fold_output_dir / "node_importance.png"
    )

    plot_edge_attention(
        top_edges_df,
        fold_output_dir / "edge_attention.png"
    )

    plot_connectivity_vs_importance(
        connectivity_df,
        fold_output_dir / "connectivity_vs_importance.png"
    )

    # Print summary
    print(f"\nTop 5 Most Important ROIs:")
    print(top_nodes_df.head(5).to_string(index=False))

    print(f"\nTop 5 Most Important Edges:")
    print(top_edges_df.head(5).to_string(index=False))

    return {
        'fold': fold_name,
        'n_rois': len(node_stats),
        'n_edges': len(edge_stats),
        'top_roi': int(top_nodes_df.iloc[0]['ROI_Index']),
        'top_roi_importance': float(top_nodes_df.iloc[0]['Mean_Importance']),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze GATv2 model interpretability")

    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='../../results/interpretability',
                        help='Output directory for analysis results')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top nodes to report')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'])
    parser.add_argument('--fold', type=str, default=None,
                        help='Analyze specific fold only')

    args = parser.parse_args()

    # Find folds to analyze
    model_dir = Path(args.model_dir)

    if args.fold:
        fold_names = [args.fold]
    else:
        fold_names = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])

    if not fold_names:
        print(f"No folds found in {model_dir}")
        return

    print(f"\nFound {len(fold_names)} folds to analyze")

    # Analyze all folds
    all_results = []

    for fold_name in fold_names:
        try:
            result = analyze_fold(
                args.model_dir,
                fold_name,
                args.output_dir,
                top_k=args.top_k,
                device=args.device
            )
            all_results.append(result)
        except Exception as e:
            print(f"\nError analyzing {fold_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save aggregate summary
    if all_results:
        aggregate_path = Path(args.output_dir) / "aggregate_interpretability.json"
        with open(aggregate_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Analysis complete!")
        print(f"Results saved to: {args.output_dir}/")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
