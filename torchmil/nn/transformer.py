"""Transformer encoder blocks for MIL."""

from __future__ import annotations

import torch
from torch import nn


class MILTransformerEncoder(nn.Module):
    """Mask-aware Transformer encoder over bag instances."""

    def __init__(
        self,
        in_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        ff_dim = feedforward_dim if feedforward_dim is not None else in_dim * 4

        layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

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

        key_padding_mask = ~attention_mask

        # Avoid NaNs for fully-masked bags by unmasking one token for encoder math,
        # then hard-masking outputs back to 0 afterwards.
        if key_padding_mask.all(dim=1).any():
            key_padding_mask = key_padding_mask.clone()
            all_masked_rows = key_padding_mask.all(dim=1)
            key_padding_mask[all_masked_rows, 0] = False

        encoded = self.encoder(instances, src_key_padding_mask=key_padding_mask)
        encoded = encoded.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        return encoded
