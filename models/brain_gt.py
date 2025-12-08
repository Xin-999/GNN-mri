"""
BrainGT: Brain Graph Transformer
================================
Graph Transformer architecture for brain connectivity analysis.
Combines local graph convolutions with global self-attention.

Key innovations:
- Multi-head self-attention over brain ROIs
- Positional encoding based on brain region identity
- Cross-attention between different brain modules
- Adaptive fusion of local and global features

Reference: Modular Graph Transformer for Brain Disorder Diagnosis (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch
import math
from typing import Optional, Tuple


class BrainPositionalEncoding(nn.Module):
    """
    Learnable positional encoding for brain ROIs.
    Each brain region gets a unique positional embedding.
    """

    def __init__(self, n_rois: int = 268, d_model: int = 64):
        super().__init__()
        self.pos_embedding = nn.Embedding(n_rois, d_model)
        self.n_rois = n_rois

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Returns positional embeddings for all ROIs."""
        positions = torch.arange(self.n_rois, device=device)
        return self.pos_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)


class GraphAttentionLayer(nn.Module):
    """
    Graph-aware attention layer that respects connectivity structure.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_edge_features: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_edge_features = use_edge_features

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        if use_edge_features:
            self.edge_proj = nn.Linear(1, n_heads)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        adj_mask: Optional[torch.Tensor] = None,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, n_nodes, d_model)
            adj_mask: (batch, n_nodes, n_nodes) - binary adjacency mask
            edge_weights: (batch, n_nodes, n_nodes) - edge weight matrix
        Returns:
            out: (batch, n_nodes, d_model)
        """
        batch_size, n_nodes, _ = x.shape

        # Compute Q, K, V
        Q = self.W_q(x).view(batch_size, n_nodes, self.n_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, n_nodes, self.n_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, n_nodes, self.n_heads, self.head_dim)

        # Transpose for attention: (batch, n_heads, n_nodes, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Add edge weight bias if available
        if self.use_edge_features and edge_weights is not None:
            edge_bias = self.edge_proj(edge_weights.unsqueeze(-1))  # (batch, n, n, n_heads)
            edge_bias = edge_bias.permute(0, 3, 1, 2)  # (batch, n_heads, n, n)
            scores = scores + edge_bias

        # Apply adjacency mask (optional - for local attention)
        if adj_mask is not None:
            mask = adj_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, n_nodes, self.d_model)
        out = self.W_o(out)

        return out


class TransformerBlock(nn.Module):
    """
    Transformer block with graph-aware attention.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        use_edge_features: bool = True,
    ):
        super().__init__()

        self.attention = GraphAttentionLayer(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_edge_features=use_edge_features,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adj_mask: Optional[torch.Tensor] = None,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x), adj_mask, edge_weights)
        x = x + self.dropout(attn_out)

        # FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        return x


class LocalGNNLayer(nn.Module):
    """
    Local GNN layer for capturing neighborhood structure.
    Uses GATv2 for better expressiveness.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat = GATv2Conv(
            in_dim, out_dim // heads, heads=heads, dropout=dropout, concat=True
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        out = self.gat(x, edge_index, edge_attr=edge_attr)
        out = F.elu(out)
        out = self.norm(out)
        return out


class BrainGraphTransformer(nn.Module):
    """
    Core Brain Graph Transformer architecture.
    Combines local GNN layers with global transformer attention.
    """

    def __init__(
        self,
        in_dim: int = 268,
        hidden_dim: int = 128,
        n_rois: int = 268,
        n_transformer_layers: int = 4,
        n_gnn_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.2,
        use_local_attention: bool = False,  # If True, mask attention with adjacency
    ):
        super().__init__()
        self.n_rois = n_rois
        self.hidden_dim = hidden_dim
        self.use_local_attention = use_local_attention

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Positional encoding for brain regions
        self.pos_encoding = BrainPositionalEncoding(n_rois, hidden_dim)

        # Local GNN layers
        self.gnn_layers = nn.ModuleList([
            LocalGNNLayer(hidden_dim, hidden_dim, heads=4, dropout=dropout)
            for _ in range(n_gnn_layers)
        ])

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                d_model=hidden_dim,
                n_heads=n_heads,
                ff_dim=hidden_dim * 4,
                dropout=dropout,
                use_edge_features=True,
            )
            for _ in range(n_transformer_layers)
        ])

        # Fusion layer for local + global features
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (total_nodes, in_dim) - node features
            edge_index: (2, num_edges) - edge indices
            edge_attr: (num_edges,) - edge weights
            batch: (total_nodes,) - batch assignment
        Returns:
            local_out: (total_nodes, hidden_dim) - local GNN output
            global_out: (total_nodes, hidden_dim) - global transformer output
        """
        # Input projection
        x = self.input_proj(x)

        # Get batch information
        batch_size = batch.max().item() + 1
        device = x.device

        # --- Local GNN Path ---
        local_x = x
        for gnn in self.gnn_layers:
            local_x = gnn(local_x, edge_index, edge_attr)
            local_x = local_x + x  # Residual

        # --- Global Transformer Path ---
        # Convert to dense batch format for transformer
        x_dense, mask = to_dense_batch(x, batch, max_num_nodes=self.n_rois)

        # Add positional encoding
        pos_enc = self.pos_encoding(batch_size, device)
        x_dense = x_dense + pos_enc

        # Get dense adjacency for attention mask/edge features
        adj_dense = to_dense_adj(edge_index, batch, edge_attr=edge_attr, max_num_nodes=self.n_rois)

        # Apply transformer layers
        for transformer in self.transformer_layers:
            adj_mask = (adj_dense > 0).float() if self.use_local_attention else None
            x_dense = transformer(x_dense, adj_mask, adj_dense)

        # Convert back to sparse format
        global_x = x_dense[mask]

        return local_x, global_x


class BrainGT(nn.Module):
    """
    Full BrainGT Model for cognitive score prediction.

    Architecture:
    1. Input projection
    2. Local GNN for neighborhood features
    3. Global Transformer for long-range dependencies
    4. Adaptive fusion of local + global
    5. Hierarchical pooling
    6. Regression head
    """

    def __init__(
        self,
        in_dim: int = 268,
        hidden_dim: int = 128,
        n_rois: int = 268,
        n_transformer_layers: int = 4,
        n_gnn_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.2,
        pool_type: str = 'attention',  # 'mean', 'attention', 'hierarchical'
    ):
        super().__init__()
        self.pool_type = pool_type

        # Core transformer architecture
        self.brain_transformer = BrainGraphTransformer(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            n_rois=n_rois,
            n_transformer_layers=n_transformer_layers,
            n_gnn_layers=n_gnn_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Attention pooling
        if pool_type == 'attention':
            self.pool_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        # Hierarchical pooling (for brain modules)
        if pool_type == 'hierarchical':
            self.hier_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data) -> torch.Tensor:
        """
        Args:
            data: PyG Data/Batch object
        Returns:
            predictions: (batch_size,) - predicted scores
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch

        if edge_attr is not None and edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)

        # Get local and global representations
        local_out, global_out = self.brain_transformer(x, edge_index, edge_attr, batch)

        # Fuse local and global
        fused = self.fusion(torch.cat([local_out, global_out], dim=-1))

        # Pooling
        if self.pool_type == 'mean':
            pooled = global_mean_pool(fused, batch)

        elif self.pool_type == 'attention':
            gate = self.pool_gate(fused)
            # Normalize within each graph
            gate = torch.zeros_like(gate).scatter_add_(
                0, batch.unsqueeze(-1), torch.exp(gate)
            )
            gate_sum = torch.zeros(batch.max() + 1, 1, device=gate.device).scatter_add_(
                0, batch.unsqueeze(-1), torch.exp(self.pool_gate(fused))
            )
            gate = torch.exp(self.pool_gate(fused)) / gate_sum[batch]
            pooled = global_add_pool(fused * gate, batch)

        elif self.pool_type == 'hierarchical':
            # First aggregate within modules, then across modules
            importance = torch.sigmoid(self.hier_pool(fused))
            pooled = global_add_pool(fused * importance, batch)

        else:
            pooled = global_mean_pool(fused, batch)

        # Prediction
        out = self.head(pooled)

        return out.squeeze(-1)

    @torch.no_grad()
    def get_attention_weights(self, data) -> torch.Tensor:
        """Get attention weights for interpretability."""
        self.eval()
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch

        if edge_attr is not None and edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)

        local_out, global_out = self.brain_transformer(x, edge_index, edge_attr, batch)
        fused = self.fusion(torch.cat([local_out, global_out], dim=-1))

        if self.pool_type == 'attention':
            gate = torch.softmax(self.pool_gate(fused), dim=0)
            return gate.squeeze(-1)

        return torch.ones(x.size(0))
