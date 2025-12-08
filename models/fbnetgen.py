"""
FBNetGen: Task-aware GNN-based fMRI Analysis via Functional Brain Network Generation
=====================================================================================
Based on: Kan et al., MIDL 2022
Paper: https://proceedings.mlr.press/v172/kan22a.html

Key Innovation:
- Learns optimal graph structure from time-series END-TO-END
- Task-aware: Graph generation is optimized for prediction task
- Three components: Time-series encoder, Graph generator, GNN predictor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for time-series transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class TimeSeriesEncoder(nn.Module):
    """
    Encodes fMRI time-series using Transformer architecture.
    Maps (batch, time, n_rois) -> (batch, n_rois, hidden_dim)
    """

    def __init__(
        self,
        n_rois: int = 268,
        time_points: int = 20,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_rois = n_rois
        self.hidden_dim = hidden_dim

        # Project each ROI's time-series to hidden dimension
        self.input_proj = nn.Linear(time_points, hidden_dim)

        # Transformer encoder for capturing temporal dependencies
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=n_rois, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, time_points, n_rois) - raw time-series
        Returns:
            node_features: (batch, n_rois, hidden_dim) - encoded features
        """
        # Transpose to (batch, n_rois, time_points)
        x = x.transpose(1, 2)

        # Project time dimension to hidden
        x = self.input_proj(x)  # (batch, n_rois, hidden_dim)

        # Apply positional encoding and transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # Output projection
        x = self.output_proj(x)

        return x


class GraphGenerator(nn.Module):
    """
    Generates task-aware brain connectivity graphs from encoded features.
    Key: Learns optimal graph structure for downstream task.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_rois: int = 268,
        sparsity: float = 0.5,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_rois = n_rois
        self.sparsity = sparsity
        self.temperature = temperature

        # Learnable parameters for graph generation
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)

        # Bias term for connectivity prior
        self.connectivity_bias = nn.Parameter(torch.zeros(n_rois, n_rois))

    def forward(self, node_features, return_soft=False):
        """
        Args:
            node_features: (batch, n_rois, hidden_dim)
            return_soft: If True, return soft adjacency for interpretability
        Returns:
            adjacency: (batch, n_rois, n_rois) - learned adjacency matrix
        """
        batch_size = node_features.size(0)

        # Compute query and key projections
        Q = self.W_q(node_features)  # (batch, n_rois, hidden_dim)
        K = self.W_k(node_features)  # (batch, n_rois, hidden_dim)

        # Compute attention scores as adjacency
        attention = torch.bmm(Q, K.transpose(1, 2))  # (batch, n_rois, n_rois)
        attention = attention / math.sqrt(self.hidden_dim)

        # Add learned connectivity bias
        attention = attention + self.connectivity_bias.unsqueeze(0)

        # Apply softmax with temperature for soft adjacency
        soft_adj = F.softmax(attention / self.temperature, dim=-1)

        # Make symmetric
        soft_adj = (soft_adj + soft_adj.transpose(1, 2)) / 2

        # Ensure positive values (ReLU)
        soft_adj = F.relu(soft_adj)

        # Apply sparsity via top-k
        if self.training:
            # During training, use soft adjacency with sparsity regularization
            adjacency = soft_adj
        else:
            # During inference, apply hard threshold
            k = int(self.n_rois * self.n_rois * self.sparsity)
            flat_adj = soft_adj.view(batch_size, -1)
            threshold = flat_adj.topk(k, dim=-1).values[:, -1:]
            adjacency = (soft_adj >= threshold.view(batch_size, 1, 1)).float() * soft_adj

        if return_soft:
            return adjacency, soft_adj
        return adjacency


class GNNPredictor(nn.Module):
    """
    GNN-based predictor for regression from generated graphs.
    Uses Graph Attention Networks for better expressiveness.
    """

    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 1,
        n_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            if i == 0:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
                )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Global pooling attention
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # MLP head for regression
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, node_features, edge_index, edge_weight, batch):
        """
        Args:
            node_features: (total_nodes, in_dim)
            edge_index: (2, num_edges)
            edge_weight: (num_edges,)
            batch: (total_nodes,) - batch assignment
        Returns:
            predictions: (batch_size, out_dim)
        """
        x = self.input_proj(node_features)

        # GAT layers with residual connections
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            x_res = x
            x = gat(x, edge_index, edge_attr=edge_weight)
            x = F.elu(x)
            x = norm(x)
            if i > 0:  # Skip connection after first layer
                x = x + x_res

        # Attention-weighted global pooling
        gate = self.pool_gate(x)
        gate = torch.softmax(gate, dim=0)

        # Pool per graph in batch
        x = x * gate
        x = global_add_pool(x, batch)

        # Final prediction
        out = self.mlp(x)

        return out.squeeze(-1)


class FBNetGen(nn.Module):
    """
    Full FBNetGen Model: End-to-end task-aware brain network analysis.

    Pipeline:
    1. TimeSeriesEncoder: Raw fMRI -> Node embeddings
    2. GraphGenerator: Node embeddings -> Learned adjacency matrix
    3. GNNPredictor: Graph -> Cognitive score prediction
    """

    def __init__(
        self,
        n_rois: int = 268,
        time_points: int = 20,
        encoder_dim: int = 64,
        gnn_hidden: int = 128,
        n_encoder_layers: int = 2,
        n_gnn_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.3,
        sparsity: float = 0.4,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_rois = n_rois

        # Component 1: Time-series encoder
        self.encoder = TimeSeriesEncoder(
            n_rois=n_rois,
            time_points=time_points,
            hidden_dim=encoder_dim,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            dropout=dropout,
        )

        # Component 2: Graph generator
        self.graph_gen = GraphGenerator(
            hidden_dim=encoder_dim,
            n_rois=n_rois,
            sparsity=sparsity,
            temperature=temperature,
        )

        # Component 3: GNN predictor
        self.predictor = GNNPredictor(
            in_dim=encoder_dim,
            hidden_dim=gnn_hidden,
            out_dim=1,
            n_layers=n_gnn_layers,
            heads=n_heads,
            dropout=dropout,
        )

        # Regularization loss weights
        self.reg_weights = {
            'sparsity': 0.01,
            'smoothness': 0.01,
        }

    def _adj_to_pyg(self, node_features, adjacency, device):
        """
        Convert dense adjacency and node features to PyG batch.

        Args:
            node_features: (batch, n_rois, hidden_dim)
            adjacency: (batch, n_rois, n_rois)
        Returns:
            PyG Batch object
        """
        batch_size = node_features.size(0)
        data_list = []

        for i in range(batch_size):
            adj = adjacency[i]
            nf = node_features[i]

            # Convert dense adjacency to sparse edge_index
            edge_index, edge_weight = dense_to_sparse(adj)

            # Create PyG Data object
            data = Data(
                x=nf,
                edge_index=edge_index,
                edge_attr=edge_weight,
            )
            data_list.append(data)

        # Batch all graphs
        batch = Batch.from_data_list(data_list)
        return batch.to(device)

    def compute_reg_loss(self, adjacency):
        """
        Compute regularization losses for graph generation.

        Args:
            adjacency: (batch, n_rois, n_rois)
        Returns:
            reg_loss: Scalar tensor
        """
        # Sparsity loss: encourage sparse graphs
        sparsity_loss = adjacency.mean()

        # Smoothness loss: encourage similar nodes to connect
        # (Implemented as Frobenius norm of Laplacian)
        degree = adjacency.sum(dim=-1, keepdim=True)
        laplacian = torch.diag_embed(degree.squeeze(-1)) - adjacency
        smoothness_loss = (laplacian ** 2).mean()

        reg_loss = (
            self.reg_weights['sparsity'] * sparsity_loss +
            self.reg_weights['smoothness'] * smoothness_loss
        )

        return reg_loss

    def forward(self, timeseries, return_graph=False):
        """
        End-to-end forward pass.

        Args:
            timeseries: (batch, time_points, n_rois) or list of (time, n_rois)
            return_graph: If True, also return generated adjacency
        Returns:
            predictions: (batch,) - predicted cognitive scores
            adjacency (optional): (batch, n_rois, n_rois) - learned graphs
        """
        device = next(self.parameters()).device

        # Handle variable-length input
        if isinstance(timeseries, list):
            # Pad to same length
            max_time = max(ts.size(0) for ts in timeseries)
            padded = []
            for ts in timeseries:
                if ts.size(0) < max_time:
                    pad = torch.zeros(max_time - ts.size(0), ts.size(1), device=device)
                    ts = torch.cat([ts, pad], dim=0)
                padded.append(ts)
            timeseries = torch.stack(padded)

        # Ensure correct device
        timeseries = timeseries.to(device)

        # 1. Encode time-series to node features
        node_features = self.encoder(timeseries)  # (batch, n_rois, encoder_dim)

        # 2. Generate task-aware adjacency matrix
        adjacency = self.graph_gen(node_features)  # (batch, n_rois, n_rois)

        # 3. Convert to PyG format and predict
        pyg_batch = self._adj_to_pyg(node_features, adjacency, device)
        predictions = self.predictor(
            pyg_batch.x,
            pyg_batch.edge_index,
            pyg_batch.edge_attr,
            pyg_batch.batch,
        )

        if return_graph:
            return predictions, adjacency
        return predictions

    def get_learned_connectivity(self, timeseries):
        """
        Extract learned brain connectivity for interpretability.

        Args:
            timeseries: (batch, time_points, n_rois)
        Returns:
            adjacency: (batch, n_rois, n_rois) - learned connectivity
            node_importance: (batch, n_rois) - node importance scores
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            timeseries = timeseries.to(device)

            # Encode
            node_features = self.encoder(timeseries)

            # Generate graph with soft adjacency
            adjacency, soft_adj = self.graph_gen(node_features, return_soft=True)

            # Node importance as degree in learned graph
            node_importance = adjacency.sum(dim=-1)

            return adjacency.cpu(), node_importance.cpu()


class FBNetGenFromGraph(nn.Module):
    """
    Simplified FBNetGen that works with pre-computed graphs (your current pipeline).
    Uses the GNN predictor component with graph refinement.
    """

    def __init__(
        self,
        in_dim: int = 268,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.3,
        refine_graph: bool = True,
    ):
        super().__init__()
        self.refine_graph = refine_graph

        # Optional graph refinement layer
        if refine_graph:
            self.graph_refine = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.W_q = nn.Linear(hidden_dim, hidden_dim)
            self.W_k = nn.Linear(hidden_dim, hidden_dim)

        # Main predictor
        self.predictor = GNNPredictor(
            in_dim=in_dim if not refine_graph else hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            n_layers=n_layers,
            heads=n_heads,
            dropout=dropout,
        )

    def forward(self, data):
        """
        Args:
            data: PyG Data/Batch with x, edge_index, edge_attr, batch
        Returns:
            predictions: (batch_size,)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        if self.refine_graph and edge_attr is not None:
            # Refine node features
            x = self.graph_refine(x)

        # Predict
        predictions = self.predictor(
            x,
            edge_index,
            edge_attr.squeeze(-1) if edge_attr is not None else None,
            batch,
        )

        return predictions
