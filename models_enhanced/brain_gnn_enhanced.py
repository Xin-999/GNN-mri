"""
Enhanced BrainGNN - ROI-Aware GNN with Advanced Regularization
==============================================================

Enhancements over base BrainGNN:
1. Biased DropEdge: Removes inter-class edges preferentially to reduce noise
2. Enhanced TopK Pooling Loss: Stronger regularization for ROI selection
3. Group-Level Consistency Loss: Improved stability
4. Data Augmentation Support: Bootstrap-based augmentation for fMRI
5. PairNorm: Prevents over-smoothing in deep networks

Based on 2024-2025 research:
- Biased DropEdge: https://www.sciencedirect.com/science/article/abs/pii/S0950705125006616 (May 2025)
- fMRI Augmentation: https://pmc.ncbi.nlm.nih.gov/articles/PMC7544244/
- PairNorm: Various papers on mitigating over-smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import dropout_edge, softmax
import math


class PairNorm(nn.Module):
    """
    PairNorm layer to prevent over-smoothing.

    Normalizes pairwise distances between node representations
    to prevent them from converging to the same vector.
    """
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        """
        Args:
            x: Node features [N, F]
        """
        # Center the features
        mean = x.mean(dim=0, keepdim=True)
        x_centered = x - mean

        # Compute L2 norm
        l2_norm = torch.norm(x_centered, p=2, dim=1, keepdim=True)
        l2_norm = l2_norm + 1e-6  # Avoid division by zero

        # Normalize
        x_normalized = x_centered / l2_norm * self.scale * math.sqrt(x.size(0))

        return x_normalized


class BiasedDropEdge:
    """
    Biased DropEdge: Preferentially drops inter-class edges to reduce noise.

    This is a 2025 advancement over standard DropEdge that removes edges
    connecting nodes from different classes/groups with higher probability.

    For brain graphs: Drops edges between functionally dissimilar ROIs
    with higher probability.
    """
    def __init__(self, p=0.1, bias_strength=2.0):
        """
        Args:
            p: Base dropout probability
            bias_strength: How much more likely to drop inter-class edges (>1.0)
        """
        self.p = p
        self.bias_strength = bias_strength

    def __call__(self, edge_index, edge_attr=None, node_features=None, training=True):
        """
        Apply biased edge dropout.

        Args:
            edge_index: Edge indices [2, E]
            edge_attr: Edge attributes [E, *]
            node_features: Node features [N, F] (used to compute similarity)
            training: Whether in training mode
        """
        if not training or self.p <= 0:
            return edge_index, edge_attr

        # If no node features, fall back to standard DropEdge
        if node_features is None:
            return dropout_edge(edge_index, p=self.p, force_undirected=True, training=training)

        # Compute edge dropout probabilities based on node similarity
        src, dst = edge_index[0], edge_index[1]
        src_features = node_features[src]
        dst_features = node_features[dst]

        # Cosine similarity between connected nodes
        similarity = F.cosine_similarity(src_features, dst_features, dim=-1)

        # Convert similarity to dropout probability
        # High similarity -> low dropout (keep edge)
        # Low similarity -> high dropout (remove edge)
        dropout_prob = self.p + (1 - similarity) * (self.p * self.bias_strength)
        dropout_prob = torch.clamp(dropout_prob, 0.0, 0.9)

        # Sample edges to keep
        rand = torch.rand(edge_index.size(1), device=edge_index.device)
        keep_mask = rand > dropout_prob

        # Apply mask
        edge_index = edge_index[:, keep_mask]
        if edge_attr is not None:
            edge_attr = edge_attr[keep_mask]

        return edge_index, edge_attr


class EnhancedROIAwareConv(MessagePassing):
    """
    Enhanced ROI-aware convolution with PairNorm.
    """
    def __init__(self, in_channels, out_channels, n_rois=268, n_communities=7):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_communities = n_communities

        # Basis kernels for each community
        self.basis_kernels = nn.Parameter(
            torch.Tensor(n_communities, in_channels, out_channels)
        )

        # ROI to community assignment (learnable)
        self.roi_community = nn.Parameter(torch.Tensor(n_rois, n_communities))

        # PairNorm to prevent over-smoothing
        self.pairnorm = PairNorm()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.basis_kernels)
        nn.init.xavier_uniform_(self.roi_community)

    def forward(self, x, edge_index, edge_attr=None, node_ids=None):
        """
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge indices [2, E]
            edge_attr: Edge weights [E, 1]
            node_ids: ROI identifiers [N] (0 to n_rois-1)
        """
        # Get ROI-specific kernels
        community_weights = F.softmax(self.roi_community, dim=-1)
        roi_kernels = torch.einsum('rc,cio->rio', community_weights, self.basis_kernels)

        # Transform features based on ROI
        if node_ids is not None:
            x_transformed = torch.einsum('ni,rio->no', x, roi_kernels[node_ids])
        else:
            # If no node_ids provided, use average kernel
            avg_kernel = roi_kernels.mean(dim=0)
            x_transformed = torch.matmul(x, avg_kernel)

        # Message passing
        out = self.propagate(edge_index, x=x_transformed, edge_attr=edge_attr)

        # Apply PairNorm to prevent over-smoothing
        out = self.pairnorm(out)

        return out

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            return edge_attr.view(-1, 1) * x_j
        return x_j


class EnhancedROIPool(nn.Module):
    """
    Enhanced ROI pooling with stronger regularization.
    """
    def __init__(self, in_channels, ratio=0.5, topk_loss_weight=0.2):
        super().__init__()
        self.ratio = ratio
        self.topk_loss_weight = topk_loss_weight

        self.score_net = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, 1),
        )

    def forward(self, x, batch, return_loss=True):
        """
        Args:
            x: Node features [N, F]
            batch: Batch assignment [N]
            return_loss: Whether to compute regularization loss

        Returns:
            x_pooled: Pooled features [batch_size, F]
            topk_loss: Regularization loss (if return_loss=True)
            selected_mask: Mask of selected nodes (for interpretability)
        """
        # Compute importance scores
        scores = self.score_net(x).squeeze(-1)

        # Select top-k nodes per graph
        batch_size = batch.max().item() + 1
        selected_mask = torch.zeros_like(scores, dtype=torch.bool)

        for i in range(batch_size):
            mask = (batch == i)
            graph_scores = scores[mask]
            k = max(1, int(self.ratio * mask.sum().item()))

            # Select top-k
            _, top_indices = torch.topk(graph_scores, k)
            graph_indices = torch.where(mask)[0]
            selected_mask[graph_indices[top_indices]] = True

        # Pool selected nodes
        x_selected = x[selected_mask]
        batch_selected = batch[selected_mask]
        x_pooled = global_add_pool(x_selected, batch_selected)

        # Compute TopK loss (stronger regularization)
        topk_loss = None
        if return_loss:
            # Encourage diversity in selection
            topk_loss = 0
            for i in range(batch_size):
                mask = (batch == i)
                graph_scores = scores[mask]

                # Selected scores should be significantly higher than unselected
                selected_scores = graph_scores[selected_mask[mask]]
                unselected_scores = graph_scores[~selected_mask[mask]]

                if unselected_scores.numel() > 0:
                    # Margin loss: selected should be > unselected + margin
                    margin = 0.5
                    loss = F.relu(margin - (selected_scores.mean() - unselected_scores.mean()))
                    topk_loss = topk_loss + loss

            topk_loss = topk_loss / batch_size * self.topk_loss_weight

        return x_pooled, topk_loss, selected_mask


class BrainGNNEnhanced(nn.Module):
    """
    Enhanced BrainGNN with multiple improvements.

    Key Enhancements:
    1. Biased DropEdge: Preferentially drops dissimilar edges
    2. PairNorm: Prevents over-smoothing
    3. Enhanced TopK Pooling Loss: Stronger regularization
    4. Group-Level Consistency Loss: More stable training
    5. Data augmentation support (applied during data loading)

    Architecture:
    - Input projection
    - L x (ROI-aware conv + PairNorm + Biased DropEdge)
    - ROI pooling with enhanced regularization
    - Regression head
    """
    def __init__(
        self,
        in_dim=268,
        hidden_dim=128,
        n_layers=3,
        n_rois=268,
        n_communities=7,
        dropout=0.3,
        pool_ratio=0.5,
        edge_dropout=0.1,
        biased_drop_strength=2.0,
        topk_loss_weight=0.2,
        consistency_loss_weight=0.1,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.consistency_loss_weight = consistency_loss_weight

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ROI-aware convolution layers with PairNorm
        self.convs = nn.ModuleList([
            EnhancedROIAwareConv(hidden_dim, hidden_dim, n_rois=n_rois, n_communities=n_communities)
            for _ in range(n_layers)
        ])

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)
        ])

        # Biased DropEdge
        self.biased_dropedge = BiasedDropEdge(p=edge_dropout, bias_strength=biased_drop_strength)

        # ROI pooling with enhanced regularization
        self.roi_pool = EnhancedROIPool(
            hidden_dim,
            ratio=pool_ratio,
            topk_loss_weight=topk_loss_weight
        )

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data, return_losses=True):
        """
        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch
            return_losses: Whether to return regularization losses

        Returns:
            If return_losses=True: (predictions, losses_dict)
            If return_losses=False: predictions
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Input projection
        x = self.input_proj(x)

        # Store representations for consistency loss
        representations = []

        # ROI-aware convolution layers with Biased DropEdge
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            # Apply Biased DropEdge during training
            edge_index_dropped, edge_attr_dropped = self.biased_dropedge(
                edge_index,
                edge_attr,
                node_features=x,
                training=self.training
            )

            # Convolution (includes PairNorm)
            x = conv(x, edge_index_dropped, edge_attr_dropped)
            x = bn(x)
            x = F.relu(x)

            representations.append(x)

        # ROI pooling with TopK loss
        x_pooled, topk_loss, selected_mask = self.roi_pool(x, batch, return_loss=return_losses)

        # Regression
        out = self.head(x_pooled).squeeze(-1)

        if return_losses:
            # Compute group-level consistency loss
            consistency_loss = self._compute_consistency_loss(representations, batch)

            losses = {
                'topk_loss': topk_loss if topk_loss is not None else torch.tensor(0.0, device=out.device),
                'consistency_loss': consistency_loss,
            }

            return out, losses
        else:
            return out

    def _compute_consistency_loss(self, representations, batch):
        """
        Group-level consistency loss: Encourage nodes in the same graph
        to have similar representations.
        """
        if not representations:
            return torch.tensor(0.0, device=batch.device)

        batch_size = batch.max().item() + 1
        consistency_loss = 0

        for repr_layer in representations:
            for i in range(batch_size):
                mask = (batch == i)
                graph_repr = repr_layer[mask]

                if graph_repr.size(0) > 1:
                    # Compute pairwise similarity within graph
                    mean_repr = graph_repr.mean(dim=0, keepdim=True)
                    similarity = F.cosine_similarity(graph_repr, mean_repr, dim=-1)

                    # Encourage high similarity (consistency)
                    consistency_loss = consistency_loss + (1 - similarity).mean()

        consistency_loss = consistency_loss / (len(representations) * batch_size)
        consistency_loss = consistency_loss * self.consistency_loss_weight

        return consistency_loss

    def get_important_rois(self, data):
        """
        Get importance scores for ROIs (for interpretability).

        Args:
            data: PyG Data object

        Returns:
            importance_scores: [n_rois] importance scores
        """
        self.eval()
        with torch.no_grad():
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

            # Forward pass
            x = self.input_proj(x)
            for conv, bn in zip(self.convs, self.batch_norms):
                x = conv(x, edge_index, edge_attr)
                x = bn(x)
                x = F.relu(x)

            # Get selection scores
            scores = self.roi_pool.score_net(x).squeeze(-1)

        return scores


def create_brain_gnn_enhanced(in_dim=268, **kwargs):
    """
    Factory function to create BrainGNN Enhanced model.

    Args:
        in_dim: Input feature dimension (number of ROIs)
        **kwargs: Additional arguments for BrainGNNEnhanced

    Returns:
        BrainGNNEnhanced model
    """
    return BrainGNNEnhanced(in_dim=in_dim, **kwargs)


if __name__ == "__main__":
    # Test the model
    from torch_geometric.data import Data, Batch

    # Create dummy data
    n_nodes = 268
    x = torch.randn(n_nodes, 268)
    edge_index = torch.randint(0, n_nodes, (2, 1000))
    edge_attr = torch.randn(1000, 1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch_data = Batch.from_data_list([data, data])

    # Create model
    model = BrainGNNEnhanced(
        in_dim=268,
        hidden_dim=128,
        n_layers=3,
        dropout=0.3,
        edge_dropout=0.1,
        biased_drop_strength=2.0,
    )

    # Forward pass
    model.train()
    out, losses = model(batch_data, return_losses=True)

    print(f"Model created successfully!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"TopK loss: {losses['topk_loss'].item():.4f}")
    print(f"Consistency loss: {losses['consistency_loss'].item():.4f}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
