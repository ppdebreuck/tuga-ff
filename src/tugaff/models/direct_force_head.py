"""Equivariant direct-force readout from invariant graph features."""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import MLP


class DirectForceHead(nn.Module):
    """Predict forces as a smooth, edge-wise scalar/vector sum."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int | None = None,
        activation: str = "silu",
        num_mlp_layers: int = 2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        d_hidden = d_hidden or d_model
        self.eps = float(eps)
        self.scalar_mlp = MLP(
            d_in=3 * d_model,
            d_hidden=d_hidden,
            d_out=1,
            num_layers=num_mlp_layers,
            activation=activation,
        )
        # Start with a small direct-force contribution without blocking force
        # gradients through the hidden layers and backbone on the first step.
        nn.init.normal_(self.scalar_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.scalar_mlp[-1].bias)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_support: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-node forces from supported directed edge contributions."""
        num_nodes = node_feats.shape[0]
        src, dst = edge_index

        if src.numel() == 0:
            zero = node_feats.new_zeros(())
            for parameter in self.scalar_mlp.parameters():
                zero = zero + parameter.sum() * 0.0
            return node_feats.new_zeros((num_nodes, 3)) + zero

        scalar_inputs = torch.cat(
            (node_feats[src], node_feats[dst], edge_feats), dim=-1
        )
        scalars = self.scalar_mlp(scalar_inputs).squeeze(-1)
        denom = torch.sqrt(edge_vectors.square().sum(dim=-1, keepdim=True) + self.eps**2)
        directions = edge_vectors.to(dtype=node_feats.dtype) / denom.to(
            dtype=node_feats.dtype
        )
        contributions = (
            edge_support.to(dtype=node_feats.dtype).unsqueeze(-1)
            * scalars.unsqueeze(-1)
            * directions
        )
        forces = node_feats.new_zeros((num_nodes, 3))
        forces.index_add_(0, src, contributions)
        return forces
