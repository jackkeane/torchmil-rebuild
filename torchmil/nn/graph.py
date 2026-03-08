"""Graph convolution building blocks for MIL."""

from __future__ import annotations

import torch
from torch import nn


class GraphConv(nn.Module):
    """Simple mask-aware graph convolution layer for batched bags."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        add_self_loops: bool = True,
        bias: bool = True,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.add_self_loops = add_self_loops
        self.lin_self = nn.Linear(in_dim, out_dim, bias=bias)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = activation

    def forward(
        self,
        instances: torch.Tensor,
        adjacency: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if instances.ndim != 3:
            raise ValueError("instances must have shape [B, N, D]")
        if adjacency.ndim != 3:
            raise ValueError("adjacency must have shape [B, N, N]")

        batch_size, num_instances, _ = instances.shape
        if adjacency.shape != (batch_size, num_instances, num_instances):
            raise ValueError("adjacency must have shape [B, N, N]")

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, num_instances),
                dtype=torch.bool,
                device=instances.device,
            )
        if attention_mask.shape != (batch_size, num_instances):
            raise ValueError("attention_mask must have shape [B, N]")

        mask_f = attention_mask.to(instances.dtype)
        inst = instances * mask_f.unsqueeze(-1)

        valid_edges = mask_f.unsqueeze(1) * mask_f.unsqueeze(2)
        adj = adjacency.to(instances.dtype) * valid_edges

        if self.add_self_loops:
            eye = torch.eye(num_instances, device=instances.device, dtype=instances.dtype)
            eye = eye.unsqueeze(0).expand(batch_size, -1, -1)
            adj = adj + eye * valid_edges

        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        norm_adj = adj / degree
        neigh = torch.matmul(norm_adj, inst)

        out = self.lin_self(inst) + self.lin_neigh(neigh)
        if self.activation is not None:
            out = self.activation(out)

        out = out * mask_f.unsqueeze(-1)
        return out
