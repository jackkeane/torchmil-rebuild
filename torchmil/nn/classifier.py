"""Bag-level classification head for MIL."""

from __future__ import annotations

import torch
from torch import nn


class BagClassifier(nn.Module):
    """MLP classification head for bag-level representations.

    Takes a bag representation ``[B, D]`` and outputs logits ``[B, num_classes]``.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 2,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        if hidden_dims is None:
            hidden_dims = []

        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, bag_repr: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            bag_repr: Bag representation of shape ``[B, D]``.

        Returns:
            Logits of shape ``[B, num_classes]``.
        """
        if bag_repr.ndim != 2:
            raise ValueError(f"bag_repr must have shape [B, D], got {tuple(bag_repr.shape)}")
        return self.mlp(bag_repr)
