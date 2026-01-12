"""
Diffusion-style GNN regressors for brain connectivity prediction.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP, SGConv, global_add_pool, global_mean_pool


def _get_readout(name: str) -> Callable:
    if name == "mean":
        return global_mean_pool
    if name == "add":
        return global_add_pool
    raise ValueError(f"Unknown readout: {name}")


class AppnpDiffusionRegressor(nn.Module):
    """
    APPNP-based diffusion model with MLP + propagation + graph pooling.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        mlp_layers: int = 2,
        dropout: float = 0.2,
        diffusion_steps: int = 10,
        alpha: float = 0.1,
        diffusion_dropout: float = 0.0,
        readout: str = "mean",
        use_edge_weight: bool = True,
    ):
        super().__init__()
        if mlp_layers < 1:
            raise ValueError("mlp_layers must be >= 1")

        self.use_edge_weight = use_edge_weight
        self.readout = _get_readout(readout)

        mlp = []
        for i in range(mlp_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            mlp.append(nn.Linear(in_ch, hidden_dim))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*mlp)

        self.propagation = APPNP(K=diffusion_steps, alpha=alpha, dropout=diffusion_dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        batch = data.batch

        edge_weight = None
        if self.use_edge_weight and edge_attr is not None:
            edge_weight = edge_attr.squeeze(-1) if edge_attr.dim() > 1 else edge_attr

        x = self.mlp(x)
        x = self.propagation(x, edge_index, edge_weight=edge_weight)
        x = self.readout(x, batch)
        out = self.head(x).squeeze(-1)
        return out


class SgcRegressor(nn.Module):
    """
    Simplified Graph Convolution (SGC) regressor.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        sgc_k: int = 3,
        dropout: float = 0.2,
        readout: str = "mean",
        use_edge_weight: bool = True,
    ):
        super().__init__()
        self.use_edge_weight = use_edge_weight
        self.readout = _get_readout(readout)

        self.sgc = SGConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            K=sgc_k,
            cached=False,
            add_self_loops=True,
            bias=True,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        batch = data.batch

        edge_weight = None
        if self.use_edge_weight and edge_attr is not None:
            edge_weight = edge_attr.squeeze(-1) if edge_attr.dim() > 1 else edge_attr

        x = self.sgc(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.readout(x, batch)
        out = self.head(x).squeeze(-1)
        return out

