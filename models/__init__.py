"""
Advanced GNN Models for Brain Connectivity Analysis
====================================================
State-of-the-art architectures for fMRI-based cognitive prediction.

Models:
- FBNetGen: Task-aware GNN with learned graph generation
- BrainGT: Graph Transformer with cross-attention
- BrainGNN: ROI-aware graph convolution with interpretability
- Ensemble: Combined predictions from multiple models
"""

from .fbnetgen import FBNetGen, TimeSeriesEncoder, GraphGenerator
from .brain_gt import BrainGT, BrainGraphTransformer
from .brain_gnn import BrainGNN, ROIAwareConv, ROIPool
from .ensemble import EnsembleModel, WeightedEnsemble

__all__ = [
    'FBNetGen', 'TimeSeriesEncoder', 'GraphGenerator',
    'BrainGT', 'BrainGraphTransformer',
    'BrainGNN', 'ROIAwareConv', 'ROIPool',
    'EnsembleModel', 'WeightedEnsemble',
]
