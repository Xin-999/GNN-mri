"""
Enhanced FBNetGen - Task-Aware Graph Generation with Improvements
================================================================

Enhancements over base FBNetGen:
1. PairNorm: Prevents over-smoothing in deep networks
2. Enhanced Residual Connections: Better gradient flow
3. Temperature Scaling: Learnable temperature for graph generation
4. Sparsity Regularization: Encourages sparse, interpretable graphs
5. Graph Structure Consistency Loss: More stable graph learning

Based on 2024-2025 research:
- PairNorm: Various papers on mitigating over-smoothing
- Sparse graph learning: Recent advances in learnable graph structures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.utils import dense_to_sparse
import math


class PairNorm(nn.Module):
    """
    PairNorm layer to prevent over-smoothing.

    Normalizes pairwise distances between node representations.
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


class EnhancedGraphGenerator(nn.Module):
    """
    Enhanced graph generator with temperature scaling and sparsity control.
    """
    def __init__(self, node_feature_dim, hidden_dim=64, sparsity_weight=0.01):
        super().__init__()
        self.sparsity_weight = sparsity_weight

        # Learnable temperature for graph generation
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Node transformation
        self.node_transform = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention mechanism for edge weight computation
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features, return_loss=True):
        """
        Generate adjacency matrix from node features.

        Args:
            node_features: [N, F] node features
            return_loss: Whether to compute sparsity loss

        Returns:
            adjacency: [N, N] adjacency matrix
            sparsity_loss: Regularization loss (if return_loss=True)
        """
        N = node_features.size(0)

        # Transform node features
        h = self.node_transform(node_features)

        # Compute pairwise attention scores
        # Expand to [N, N, hidden_dim * 2]
        h_i = h.unsqueeze(1).expand(N, N, -1)
        h_j = h.unsqueeze(0).expand(N, N, -1)
        h_concat = torch.cat([h_i, h_j], dim=-1)

        # Compute attention scores
        scores = self.attn(h_concat).squeeze(-1)

        # Apply temperature scaling (learnable)
        scores = scores / torch.clamp(self.temperature, min=0.1, max=5.0)

        # Symmetrize
        scores = (scores + scores.t()) / 2

        # Sigmoid to [0, 1]
        adjacency = torch.sigmoid(scores)

        # Sparsity regularization
        sparsity_loss = None
        if return_loss:
            # L1 regularization to encourage sparse graphs
            sparsity_loss = self.sparsity_weight * adjacency.abs().mean()

        return adjacency, sparsity_loss


class EnhancedGNNPredictor(nn.Module):
    """
    Enhanced GNN predictor with PairNorm and residual connections.
    """
    def __init__(self, in_dim, hidden_dim, n_layers=3, n_heads=4, dropout=0.3):
        super().__init__()
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GAT layers with PairNorm
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                hidden_dim,
                hidden_dim // n_heads,
                heads=n_heads,
                dropout=dropout,
                concat=True,
            )
            for _ in range(n_layers)
        ])

        self.pairnorms = nn.ModuleList([
            PairNorm() for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(n_layers)
        ])

        # Residual projections (if needed)
        self.residual_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) if i > 0 else nn.Identity()
            for i in range(n_layers)
        ])

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Node features [N, in_dim]
            edge_index: Edge indices [2, E]
            edge_weight: Edge weights [E]

        Returns:
            Node embeddings [N, hidden_dim]
        """
        # Input projection
        x = self.input_proj(x)

        # GAT layers with PairNorm and residual connections
        for i, (gat, pairnorm, ln, dropout, residual_proj) in enumerate(
            zip(self.gat_layers, self.pairnorms, self.layer_norms, self.dropouts, self.residual_projs)
        ):
            identity = residual_proj(x)

            # GAT convolution
            x = gat(x, edge_index, edge_attr=edge_weight)

            # PairNorm to prevent over-smoothing
            x = pairnorm(x)

            # Dropout
            x = dropout(x)

            # Residual connection
            x = x + identity

            # Layer normalization
            x = ln(x)

            # Activation
            if i < self.n_layers - 1:
                x = F.relu(x)

        return x


class FBNetGenEnhanced(nn.Module):
    """
    Enhanced FBNetGen with multiple improvements.

    Key Enhancements:
    1. PairNorm: Prevents over-smoothing in deep GNN
    2. Enhanced residual connections: Better gradient flow
    3. Learnable temperature scaling: Better graph generation control
    4. Sparsity regularization: Encourages interpretable sparse graphs
    5. Graph consistency loss: More stable graph learning

    Architecture:
    - Node feature encoder
    - Enhanced graph generator (with temperature + sparsity)
    - Enhanced GNN predictor (with PairNorm + residuals)
    - Regression head
    """
    def __init__(
        self,
        in_dim=268,
        hidden_dim=128,
        n_gnn_layers=3,
        n_heads=4,
        dropout=0.3,
        sparsity_weight=0.01,
        consistency_weight=0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.consistency_weight = consistency_weight

        # Node feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Enhanced graph generator
        self.graph_generator = EnhancedGraphGenerator(
            hidden_dim,
            hidden_dim=hidden_dim // 2,
            sparsity_weight=sparsity_weight
        )

        # Enhanced GNN predictor
        self.gnn_predictor = EnhancedGNNPredictor(
            hidden_dim,
            hidden_dim,
            n_layers=n_gnn_layers,
            n_heads=n_heads,
            dropout=dropout
        )

        # Attention pooling
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
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

        # Store previous adjacency for consistency loss
        self.register_buffer('prev_adjacency', None)

    def forward(self, data, return_losses=True, return_graph=False):
        """
        Args:
            data: PyG Data object with x, batch
            return_losses: Whether to return regularization losses
            return_graph: Whether to return generated graph

        Returns:
            If return_losses=True: (predictions, losses_dict)
            If return_losses=False: predictions
            If return_graph=True: Also returns adjacency matrix
        """
        x, batch = data.x, data.batch
        batch_size = batch.max().item() + 1

        # Process each graph in the batch separately
        predictions = []
        all_losses = {'sparsity_loss': [], 'consistency_loss': []}
        all_adjacencies = []

        for i in range(batch_size):
            mask = (batch == i)
            graph_x = x[mask]

            # Encode node features
            node_features = self.encoder(graph_x)

            # Generate graph
            adjacency, sparsity_loss = self.graph_generator(node_features, return_loss=return_losses)

            # Convert dense adjacency to sparse edge_index and edge_weight
            edge_index, edge_weight = dense_to_sparse(adjacency)

            # GNN prediction
            node_embeddings = self.gnn_predictor(node_features, edge_index, edge_weight)

            # Attention pooling
            gate_scores = self.pool_gate(node_embeddings)
            gate_weights = F.softmax(gate_scores, dim=0)
            graph_embedding = (node_embeddings * gate_weights).sum(dim=0)

            # Regression
            pred = self.head(graph_embedding).squeeze(-1)
            predictions.append(pred)

            if return_losses:
                all_losses['sparsity_loss'].append(sparsity_loss)

                # Graph consistency loss
                if self.training and self.prev_adjacency is not None:
                    consistency_loss = F.mse_loss(adjacency, self.prev_adjacency)
                    all_losses['consistency_loss'].append(consistency_loss * self.consistency_weight)
                else:
                    all_losses['consistency_loss'].append(torch.tensor(0.0, device=pred.device))

                # Update previous adjacency
                if self.training:
                    self.prev_adjacency = adjacency.detach()

            if return_graph:
                all_adjacencies.append(adjacency)

        # Stack predictions
        out = torch.stack(predictions)

        if return_losses:
            losses = {
                'sparsity_loss': torch.stack(all_losses['sparsity_loss']).mean(),
                'consistency_loss': torch.stack(all_losses['consistency_loss']).mean(),
            }

            if return_graph:
                return out, losses, all_adjacencies
            else:
                return out, losses
        else:
            if return_graph:
                return out, all_adjacencies
            else:
                return out


class FBNetGenFromGraphEnhanced(nn.Module):
    """
    Enhanced FBNetGen that uses pre-computed graphs (simplified version).

    Uses the same enhancements (PairNorm, residuals) but skips graph generation.
    Useful for comparison with base models.
    """
    def __init__(
        self,
        in_dim=268,
        hidden_dim=128,
        n_gnn_layers=3,
        n_heads=4,
        dropout=0.3,
    ):
        super().__init__()

        # Node feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Enhanced GNN predictor (with PairNorm)
        self.gnn_predictor = EnhancedGNNPredictor(
            hidden_dim,
            hidden_dim,
            n_layers=n_gnn_layers,
            n_heads=n_heads,
            dropout=dropout
        )

        # Attention pooling
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
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

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Encode
        x = self.encoder(x)

        # GNN prediction with PairNorm
        x = self.gnn_predictor(x, edge_index, edge_attr)

        # Attention pooling
        gate_scores = self.pool_gate(x)
        gate_weights = torch.softmax(gate_scores, dim=0)

        # Weighted sum per graph
        x_pooled = torch.zeros(batch.max().item() + 1, self.gnn_predictor.n_layers, device=x.device)
        for i in range(batch.max().item() + 1):
            mask = (batch == i)
            x_pooled[i] = (x[mask] * gate_weights[mask]).sum(dim=0)

        # Regression
        out = self.head(x_pooled).squeeze(-1)

        return out


def create_fbnetgen_enhanced(in_dim=268, from_graph=False, **kwargs):
    """
    Factory function to create FBNetGen Enhanced model.

    Args:
        in_dim: Input feature dimension (number of ROIs)
        from_graph: If True, use pre-computed graphs (FBNetGenFromGraphEnhanced)
        **kwargs: Additional arguments

    Returns:
        FBNetGenEnhanced or FBNetGenFromGraphEnhanced model
    """
    if from_graph:
        return FBNetGenFromGraphEnhanced(in_dim=in_dim, **kwargs)
    else:
        return FBNetGenEnhanced(in_dim=in_dim, **kwargs)


if __name__ == "__main__":
    # Test the model
    from torch_geometric.data import Data, Batch

    # Create dummy data
    n_nodes = 268
    x = torch.randn(n_nodes, 268)
    data = Data(x=x)
    batch_data = Batch.from_data_list([data, data])

    # Create model
    model = FBNetGenEnhanced(
        in_dim=268,
        hidden_dim=128,
        n_gnn_layers=3,
        n_heads=4,
        dropout=0.3,
        sparsity_weight=0.01,
    )

    # Forward pass
    model.train()
    out, losses = model(batch_data, return_losses=True)

    print(f"Model created successfully!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Sparsity loss: {losses['sparsity_loss'].item():.4f}")
    print(f"Consistency loss: {losses['consistency_loss'].item():.4f}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
