"""
Enhanced GNN Models with Advanced Techniques
===========================================

This module contains enhanced versions of the base models with additional
improvements for better accuracy:

- Virtual Nodes: Auxiliary nodes for long-range information aggregation
- DropEdge/Biased DropEdge: Regularization through edge dropout
- GraphNorm/PairNorm: Graph-specific normalization to prevent over-smoothing
- Enhanced Residual Connections: Better gradient flow
- Data Augmentation: Bootstrap-based augmentation for fMRI
- Additional Regularization: TopK pooling loss, consistency losses

All enhancements are based on 2024-2025 state-of-the-art research.
"""

from .brain_gt_enhanced import BrainGTEnhanced
from .brain_gnn_enhanced import BrainGNNEnhanced
from .fbnetgen_enhanced import FBNetGenEnhanced

__all__ = [
    'BrainGTEnhanced',
    'BrainGNNEnhanced',
    'FBNetGenEnhanced',
]
