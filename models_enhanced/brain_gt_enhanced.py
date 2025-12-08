"""
Enhanced BrainGT - Graph Transformer with Advanced Techniques
============================================================

Enhancements over base BrainGT:
1. Virtual Nodes: Auxiliary nodes for efficient long-range information aggregation
2. GraphNorm: Graph-specific normalization to prevent over-smoothing
3. DropEdge: Regularization through random edge dropout during training
4. Enhanced Residual Connections: Pre-LN architecture for better gradient flow
5. Attention Dropout: Additional regularization in attention mechanism

Based on 2024-2025 research:
- Virtual Nodes: https://arxiv.org/abs/2506.19482 (FastEGNN, June 2025)
- DropEdge: https://openreview.net/forum?id=Hkx1qkrKPr (ICLR 2020, still SOTA)
- GraphNorm: Various 2024-2025 papers on graph normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.utils import dropout_edge
import math


class GraphNorm(nn.Module):
    """
    Graph Normalization layer to prevent over-smoothing.

    Normalizes node features across the graph while maintaining
    graph structure information.
    """
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, batch):
        """
        Args:
            x: Node features [N, F]
            batch: Batch assignment [N]
        """
        # Compute mean and std per graph
        batch_size = batch.max().item() + 1

        # Group by batch
        mean_list = []
        std_list = []
        for i in range(batch_size):
            mask = (batch == i)
            graph_x = x[mask]
            mean_list.append(graph_x.mean(dim=0, keepdim=True))
            std_list.append(graph_x.std(dim=0, keepdim=True) + self.eps)

        # Normalize each graph
        x_normalized = torch.zeros_like(x)
        for i in range(batch_size):
            mask = (batch == i)
            x_normalized[mask] = (x[mask] - mean_list[i]) / std_list[i]

        # Apply learned affine transformation
        return self.weight * x_normalized + self.bias


class VirtualNodeLayer(nn.Module):
    """
    Virtual Node layer for global information aggregation.

    Adds a virtual node that connects to all real nodes, enabling
    efficient long-range information propagation across the graph.

    Based on: FastEGNN (June 2025) and other virtual node research.
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.mlp_vn_encode = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.mlp_vn_decode = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x, virtual_node_emb, batch):
        """
        Args:
            x: Node features [N, hidden_dim]
            virtual_node_emb: Virtual node embeddings [batch_size, hidden_dim]
            batch: Batch assignment [N]

        Returns:
            Updated x and virtual_node_emb
        """
        batch_size = batch.max().item() + 1

        # Aggregate information from real nodes to virtual node
        vn_update = torch.zeros(batch_size, x.size(1), device=x.device)
        for i in range(batch_size):
            mask = (batch == i)
            vn_update[i] = x[mask].mean(dim=0)

        # Update virtual node
        virtual_node_emb = virtual_node_emb + self.mlp_vn_encode(vn_update)

        # Broadcast virtual node information back to real nodes
        vn_broadcast = virtual_node_emb[batch]
        x = x + self.mlp_vn_decode(vn_broadcast)

        return x, virtual_node_emb


class EnhancedGraphAttentionLayer(nn.Module):
    """
    Enhanced Graph Attention with DropEdge and Attention Dropout.
    """
    def __init__(self, in_dim, out_dim, n_heads=8, dropout=0.1,
                 attention_dropout=0.1, edge_dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.edge_dropout = edge_dropout

        self.gat = GATv2Conv(
            in_dim,
            out_dim // n_heads,
            heads=n_heads,
            dropout=attention_dropout,
            edge_dim=1,
            concat=True,
        )

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        # Projection for residual if dimensions don't match
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x, edge_index, edge_attr=None, training=False):
        """
        Forward pass with DropEdge during training.
        """
        identity = x

        # Apply DropEdge during training
        if training and self.edge_dropout > 0:
            edge_index, edge_attr = dropout_edge(
                edge_index,
                edge_attr=edge_attr,
                p=self.edge_dropout,
                force_undirected=True,
                training=True
            )

        # Graph attention
        x = self.gat(x, edge_index, edge_attr=edge_attr)
        x = self.dropout(x)

        # Residual connection
        if self.proj is not None:
            identity = self.proj(identity)
        x = x + identity

        # Layer normalization (Pre-LN style for better gradient flow)
        x = self.norm(x)

        return x


class EnhancedTransformerBlock(nn.Module):
    """
    Transformer block with GraphNorm and enhanced residual connections.
    """
    def __init__(self, hidden_dim, n_heads=8, dropout=0.1, attention_dropout=0.1):
        super().__init__()

        # Pre-LayerNorm architecture
        self.norm1 = GraphNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=attention_dropout,
            batch_first=True
        )

        self.norm2 = GraphNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, batch, mask=None):
        """
        Args:
            x: Node features [N, hidden_dim]
            batch: Batch assignment [N]
            mask: Attention mask (optional)
        """
        # Convert to batch format for multi-head attention
        batch_size = batch.max().item() + 1
        max_nodes = max((batch == i).sum() for i in range(batch_size))

        # Pad to create [batch_size, max_nodes, hidden_dim]
        x_batched = torch.zeros(batch_size, max_nodes, x.size(1), device=x.device)
        padding_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=x.device)

        for i in range(batch_size):
            mask_i = (batch == i)
            n_nodes = mask_i.sum()
            x_batched[i, :n_nodes] = x[mask_i]
            padding_mask[i, :n_nodes] = False

        # Pre-LN: Normalize before attention
        x_norm = torch.zeros_like(x)
        for i in range(batch_size):
            mask_i = (batch == i)
            x_norm[mask_i] = self.norm1(x[mask_i], batch[mask_i])

        # Reshape for attention
        x_norm_batched = torch.zeros_like(x_batched)
        for i in range(batch_size):
            mask_i = (batch == i)
            n_nodes = mask_i.sum()
            x_norm_batched[i, :n_nodes] = x_norm[mask_i]

        # Self-attention with residual
        attn_out, _ = self.attn(
            x_norm_batched,
            x_norm_batched,
            x_norm_batched,
            key_padding_mask=padding_mask
        )

        # Unpack and add residual
        x_out = x.clone()
        for i in range(batch_size):
            mask_i = (batch == i)
            n_nodes = mask_i.sum()
            x_out[mask_i] = x[mask_i] + attn_out[i, :n_nodes]

        x = x_out

        # Pre-LN: Normalize before FFN
        x_norm = torch.zeros_like(x)
        for i in range(batch_size):
            mask_i = (batch == i)
            x_norm[mask_i] = self.norm2(x[mask_i], batch[mask_i])

        # FFN with residual
        x = x + self.ffn(x_norm)

        return x


class BrainGTEnhanced(nn.Module):
    """
    Enhanced Brain Graph Transformer with multiple improvements.

    Key Enhancements:
    1. Virtual nodes for long-range information aggregation
    2. GraphNorm to prevent over-smoothing
    3. DropEdge for regularization during training
    4. Enhanced residual connections (Pre-LN architecture)
    5. Attention dropout

    Architecture:
    - Input projection
    - Virtual node initialization
    - L x (GNN layers with DropEdge + Virtual node update)
    - T x (Transformer blocks with GraphNorm)
    - Virtual node + Attention pooling
    - Regression head
    """
    def __init__(
        self,
        in_dim=268,
        hidden_dim=128,
        n_gnn_layers=2,
        n_transformer_layers=4,
        n_heads=8,
        dropout=0.2,
        attention_dropout=0.1,
        edge_dropout=0.1,  # DropEdge rate
        use_virtual_nodes=True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_virtual_nodes = use_virtual_nodes

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Virtual node embedding (one per graph)
        if use_virtual_nodes:
            self.virtual_node_embedding = nn.Parameter(torch.randn(1, hidden_dim))
            self.virtual_node_layers = nn.ModuleList([
                VirtualNodeLayer(hidden_dim, dropout)
                for _ in range(n_gnn_layers)
            ])

        # GNN layers with DropEdge
        self.gnn_layers = nn.ModuleList([
            EnhancedGraphAttentionLayer(
                hidden_dim,
                hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                edge_dropout=edge_dropout,
            )
            for _ in range(n_gnn_layers)
        ])

        # Transformer blocks with GraphNorm
        self.transformer_blocks = nn.ModuleList([
            EnhancedTransformerBlock(
                hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(n_transformer_layers)
        ])

        # Attention pooling
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for concat with virtual node
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Input projection
        x = self.input_proj(x)

        # Initialize virtual nodes (one per graph)
        if self.use_virtual_nodes:
            batch_size = batch.max().item() + 1
            virtual_node_emb = self.virtual_node_embedding.expand(batch_size, -1)

        # GNN layers with virtual node updates
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index, edge_attr, training=self.training)

            if self.use_virtual_nodes:
                x, virtual_node_emb = self.virtual_node_layers[i](x, virtual_node_emb, batch)

        # Transformer blocks with GraphNorm
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, batch)

        # Attention pooling
        gate_scores = self.pool_gate(x)
        gate_weights = torch.softmax(gate_scores, dim=0)

        # Weighted sum pooling
        x_pooled = torch.zeros(batch.max().item() + 1, self.hidden_dim, device=x.device)
        for i in range(batch.max().item() + 1):
            mask = (batch == i)
            x_pooled[i] = (x[mask] * gate_weights[mask]).sum(dim=0)

        # Concatenate with virtual node if used
        if self.use_virtual_nodes:
            x_pooled = torch.cat([x_pooled, virtual_node_emb], dim=-1)
        else:
            # If not using virtual nodes, use mean pooling instead
            x_mean = global_mean_pool(x, batch)
            x_pooled = torch.cat([x_pooled, x_mean], dim=-1)

        # Regression
        out = self.head(x_pooled)

        return out.squeeze(-1)


def create_brain_gt_enhanced(in_dim=268, **kwargs):
    """
    Factory function to create BrainGT Enhanced model.

    Args:
        in_dim: Input feature dimension (number of ROIs)
        **kwargs: Additional arguments for BrainGTEnhanced

    Returns:
        BrainGTEnhanced model
    """
    return BrainGTEnhanced(in_dim=in_dim, **kwargs)


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
    model = BrainGTEnhanced(
        in_dim=268,
        hidden_dim=128,
        n_gnn_layers=2,
        n_transformer_layers=4,
        n_heads=8,
        dropout=0.2,
        edge_dropout=0.1,
        use_virtual_nodes=True,
    )

    # Forward pass
    model.eval()
    out = model(batch_data)

    print(f"Model created successfully!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
