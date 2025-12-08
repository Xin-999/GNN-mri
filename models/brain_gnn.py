"""
BrainGNN: Interpretable Brain Graph Neural Network for fMRI Analysis
====================================================================
Based on: Li et al., Medical Image Analysis 2021
Paper: https://doi.org/10.1016/j.media.2021.102233

Key innovations:
1. ROI-aware Graph Convolution (Ra-GConv): Region-specific kernels
2. ROI-selection Pooling (R-pool): Interpretable node selection
3. Special regularization: Unit loss, GLC loss, TPK loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add
import math
from typing import Optional, Tuple


class ROIAwareConv(MessagePassing):
    """
    ROI-Aware Graph Convolution (Ra-GConv).

    Key idea: Different brain regions (ROIs) should have different
    transformation weights, reflecting their unique functional roles.

    Each ROI's kernel is a linear combination of K basis kernels,
    where coefficients depend on the ROI's community/module membership.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_rois: int = 268,
        n_communities: int = 7,  # Number of brain modules/communities
        use_edge_weight: bool = True,
        bias: bool = True,
    ):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_rois = n_rois
        self.n_communities = n_communities
        self.use_edge_weight = use_edge_weight

        # Basis kernels (one per community)
        self.basis_kernels = nn.Parameter(
            torch.Tensor(n_communities, in_channels, out_channels)
        )

        # ROI-to-community assignment (learnable soft assignment)
        self.roi_community = nn.Parameter(torch.Tensor(n_rois, n_communities))

        # Edge weight transformation
        if use_edge_weight:
            self.edge_weight_transform = nn.Sequential(
                nn.Linear(1, out_channels),
                nn.Sigmoid(),
            )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.basis_kernels)
        nn.init.xavier_uniform_(self.roi_community)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr=None, node_ids=None):
        """
        Args:
            x: (num_nodes, in_channels)
            edge_index: (2, num_edges)
            edge_attr: (num_edges, 1) - edge weights
            node_ids: (num_nodes,) - ROI indices (0 to n_rois-1)
        """
        # Add self-loops
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr,
            fill_value=1.0,
            num_nodes=x.size(0)
        )

        # Compute ROI-specific kernels
        # Soft community assignment
        community_weights = F.softmax(self.roi_community, dim=-1)  # (n_rois, n_communities)

        # Combine basis kernels based on community assignment
        # roi_kernels: (n_rois, in_channels, out_channels)
        roi_kernels = torch.einsum('rc,cio->rio', community_weights, self.basis_kernels)

        # Get kernel for each node based on its ROI
        if node_ids is None:
            # Assume nodes are ordered by ROI within batch
            num_nodes = x.size(0)
            node_ids = torch.arange(num_nodes, device=x.device) % self.n_rois

        node_kernels = roi_kernels[node_ids]  # (num_nodes, in_channels, out_channels)

        # Transform node features using ROI-specific kernels
        x_transformed = torch.bmm(x.unsqueeze(1), node_kernels).squeeze(1)

        # Message passing
        out = self.propagate(
            edge_index, x=x_transformed,
            edge_attr=edge_attr, size=None
        )

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, edge_attr):
        """Compute messages with edge weight modulation."""
        if self.use_edge_weight and edge_attr is not None:
            edge_weight = self.edge_weight_transform(edge_attr.view(-1, 1))
            return x_j * edge_weight
        return x_j


class ROIPool(nn.Module):
    """
    ROI-selection Pooling (R-pool).

    Learns to select the most important brain regions (ROIs)
    for the prediction task, providing interpretability.
    """

    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,  # Fraction of nodes to keep
        min_nodes: int = 10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.min_nodes = min_nodes

        # Attention for node importance
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Tanh(),
            nn.Linear(in_channels, 1),
        )

    def forward(self, x, batch):
        """
        Args:
            x: (num_nodes, in_channels)
            batch: (num_nodes,) - batch assignment
        Returns:
            x_pooled: (num_selected, in_channels)
            batch_pooled: (num_selected,)
            scores: (num_nodes,) - importance scores
            perm: indices of selected nodes
        """
        # Compute importance scores
        scores = self.attention(x).squeeze(-1)

        # Get number of nodes per graph
        batch_size = batch.max().item() + 1
        num_nodes_per_graph = scatter_add(
            torch.ones_like(batch, dtype=torch.float),
            batch, dim=0, dim_size=batch_size
        )

        # Select top-k nodes per graph
        k_per_graph = torch.clamp(
            (num_nodes_per_graph * self.ratio).long(),
            min=self.min_nodes
        )

        # Sort scores within each graph
        perm_list = []
        for i in range(batch_size):
            mask = batch == i
            graph_scores = scores[mask]
            k = min(k_per_graph[i].item(), graph_scores.size(0))

            # Get top-k indices
            _, top_indices = graph_scores.topk(k, largest=True)
            node_indices = torch.where(mask)[0]
            perm_list.append(node_indices[top_indices])

        perm = torch.cat(perm_list)

        # Select nodes
        x_pooled = x[perm] * torch.sigmoid(scores[perm]).unsqueeze(-1)
        batch_pooled = batch[perm]

        return x_pooled, batch_pooled, scores, perm


class BrainGNNBlock(nn.Module):
    """
    Single BrainGNN block: Ra-GConv + R-pool + normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_rois: int = 268,
        n_communities: int = 7,
        pool_ratio: float = 0.8,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.conv = ROIAwareConv(
            in_channels, out_channels,
            n_rois=n_rois,
            n_communities=n_communities,
        )
        self.pool = ROIPool(out_channels, ratio=pool_ratio)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch, node_ids=None):
        # ROI-aware convolution
        x = self.conv(x, edge_index, edge_attr, node_ids)
        x = F.elu(x)
        x = self.norm(x)
        x = self.dropout(x)

        # ROI-selection pooling
        x, batch, scores, perm = self.pool(x, batch)

        # Update edge_index to only include selected nodes
        # (simplified: just use fully connected for pooled nodes)

        return x, batch, scores, perm


class BrainGNN(nn.Module):
    """
    Full BrainGNN Model for cognitive score regression.

    Architecture:
    1. Input projection
    2. Multiple BrainGNN blocks (Ra-GConv + R-pool)
    3. Global pooling with attention
    4. Regression head

    Regularization:
    - Unit loss: Prevents representation collapse
    - GLC loss: Group Lasso for community sparsity
    - TPK loss: Total pooling knowledge
    """

    def __init__(
        self,
        in_dim: int = 268,
        hidden_dim: int = 128,
        n_rois: int = 268,
        n_communities: int = 7,
        n_layers: int = 3,
        pool_ratios: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_rois = n_rois
        self.n_layers = n_layers

        if pool_ratios is None:
            pool_ratios = [0.8, 0.6, 0.4]

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # BrainGNN blocks
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            ratio = pool_ratios[i] if i < len(pool_ratios) else pool_ratios[-1]
            self.blocks.append(
                BrainGNNBlock(
                    hidden_dim, hidden_dim,
                    n_rois=n_rois,
                    n_communities=n_communities,
                    pool_ratio=ratio,
                    dropout=dropout,
                )
            )

        # Global attention pooling
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Store scores for regularization
        self.pool_scores = []

    def forward(self, data, return_scores=False):
        """
        Args:
            data: PyG Data/Batch object
            return_scores: If True, return pooling scores for interpretability
        Returns:
            predictions: (batch_size,)
            scores (optional): List of pooling scores per layer
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch

        if edge_attr is not None and edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)

        # Input projection
        x = self.input_proj(x)

        # Store pooling scores for regularization
        self.pool_scores = []

        # Apply BrainGNN blocks
        for block in self.blocks:
            x, batch, scores, perm = block(x, edge_index, edge_attr, batch)
            self.pool_scores.append(scores)

            # Update edge_index for pooled graph (simplified)
            # In practice, would need to recompute edges
            edge_index = None
            edge_attr = None

        # Global attention pooling
        gate = self.global_attention(x)
        gate = softmax(gate, batch, dim=0)
        x = x * gate
        x = global_add_pool(x, batch)

        # Prediction
        out = self.head(x)

        if return_scores:
            return out.squeeze(-1), self.pool_scores
        return out.squeeze(-1)

    def compute_unit_loss(self):
        """
        Unit loss: Encourages unit-normalized node representations.
        Prevents representation collapse.
        """
        loss = 0.0
        for block in self.blocks:
            # Get community weights
            weights = F.softmax(block.conv.roi_community, dim=-1)
            # Each ROI should have unit norm assignment
            norms = weights.norm(dim=-1)
            loss += ((norms - 1.0) ** 2).mean()
        return loss

    def compute_glc_loss(self):
        """
        Group Lasso Constraint (GLC) loss.
        Encourages sparse community assignments.
        """
        loss = 0.0
        for block in self.blocks:
            weights = F.softmax(block.conv.roi_community, dim=-1)
            # L2,1 norm for group sparsity
            loss += weights.norm(dim=0).sum()
        return loss

    def compute_tpk_loss(self):
        """
        Total Pooling Knowledge (TPK) loss.
        Ensures pooling covers diverse brain regions.
        """
        if not self.pool_scores:
            return torch.tensor(0.0)

        loss = 0.0
        for scores in self.pool_scores:
            # Encourage uniform importance across nodes
            if scores.numel() > 0:
                probs = F.softmax(scores, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                # Maximize entropy (uniform selection)
                loss += -entropy

        return loss / max(len(self.pool_scores), 1)

    def get_roi_importance(self, data):
        """
        Get importance scores for each ROI for interpretability.

        Returns:
            importance: (n_rois,) - average importance per ROI
        """
        self.eval()
        with torch.no_grad():
            _, scores_list = self.forward(data, return_scores=True)

            # Average scores across layers
            all_scores = torch.cat([s for s in scores_list if s.numel() > 0])
            return all_scores


class SimpleBrainGNN(nn.Module):
    """
    Simplified BrainGNN without hierarchical pooling.
    Easier to train, still uses ROI-aware convolution.
    """

    def __init__(
        self,
        in_dim: int = 268,
        hidden_dim: int = 128,
        n_rois: int = 268,
        n_communities: int = 7,
        n_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # ROI-aware convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(n_layers):
            self.convs.append(
                ROIAwareConv(
                    hidden_dim, hidden_dim,
                    n_rois=n_rois,
                    n_communities=n_communities,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Global attention pooling
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch

        if edge_attr is not None and edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)

        # Input projection
        x = self.input_proj(x)

        # Apply ROI-aware convolutions with residual
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)
            x = norm(x)
            x = self.dropout(x)
            x = x + x_res  # Residual

        # Global attention pooling
        gate = self.global_attention(x)
        gate = softmax(gate, batch, dim=0)
        x = x * gate
        x = global_add_pool(x, batch)

        # Prediction
        out = self.head(x)

        return out.squeeze(-1)
