"""Attention modules for MIL."""

from __future__ import annotations

import torch
from torch import nn


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax over valid elements only, returning 0 for fully-masked rows."""
    if mask.dtype is not torch.bool:
        raise TypeError("mask must be bool")

    masked_scores = scores.masked_fill(~mask, float("-inf"))
    all_masked = ~mask.any(dim=dim, keepdim=True)
    masked_scores = masked_scores.masked_fill(all_masked, 0.0)

    weights = torch.softmax(masked_scores, dim=dim)
    weights = weights.masked_fill(~mask, 0.0)
    return weights


class GatedAttention(nn.Module):
    """Gated attention pooling (Ilse et al., 2018)."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.v = nn.Linear(in_dim, hidden_dim)
        self.u = nn.Linear(in_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        instances: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if instances.ndim != 3:
            raise ValueError("instances must have shape [B, N, D]")

        batch_size, num_instances, _ = instances.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, num_instances),
                dtype=torch.bool,
                device=instances.device,
            )

        if attention_mask.shape != (batch_size, num_instances):
            raise ValueError("attention_mask must have shape [B, N]")

        gate_v = torch.tanh(self.v(instances))
        gate_u = torch.sigmoid(self.u(instances))
        scores = self.w(gate_v * gate_u).squeeze(-1)

        weights = _masked_softmax(scores, attention_mask, dim=1)
        bag_repr = torch.bmm(weights.unsqueeze(1), instances).squeeze(1)

        return weights, bag_repr
