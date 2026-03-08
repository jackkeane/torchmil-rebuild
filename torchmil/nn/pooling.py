"""Pooling operators for MIL bags."""

from __future__ import annotations

import torch
from torch import nn

from .attention import GatedAttention


class MeanPooling(nn.Module):
    """Mask-aware mean pooling over instances."""

    def forward(
        self,
        instances: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        mask_f = attention_mask.to(instances.dtype).unsqueeze(-1)
        summed = (instances * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return summed / denom


class MaxPooling(nn.Module):
    """Mask-aware max pooling over instances."""

    def forward(
        self,
        instances: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        masked = instances.masked_fill(~attention_mask.unsqueeze(-1), float("-inf"))
        pooled = masked.max(dim=1).values

        all_masked = ~attention_mask.any(dim=1, keepdim=True)
        pooled = pooled.masked_fill(all_masked, 0.0)
        return pooled


class AttentionPooling(nn.Module):
    """Gated-attention pooling wrapper."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.attention = GatedAttention(in_dim=in_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        instances: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        weights, bag_repr = self.attention(instances, attention_mask)
        if return_attention:
            return bag_repr, weights
        return bag_repr
