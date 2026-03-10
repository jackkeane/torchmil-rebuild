"""TransMIL model (Shao et al., 2021)."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from torchmil.nn import MILTransformerEncoder

from .base import MILModel


class TransMIL(MILModel):
    """Transformer MIL with a learnable CLS token.

    Args:
        in_shape: Input feature shape or feature dimension.
        num_classes: Number of bag-level classes.
        num_heads: Number of transformer attention heads.
        num_layers: Number of transformer encoder layers.
        dropout: Dropout probability in transformer blocks.
        criterion: Optional loss module (defaults to CrossEntropyLoss).
    """

    def __init__(
        self,
        in_shape: int | tuple[int, ...],
        num_classes: int = 2,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        criterion: nn.Module | None = None,
    ) -> None:
        super().__init__(in_shape=in_shape, criterion=criterion)
        self.num_classes = num_classes
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.in_dim))
        self.encoder = MILTransformerEncoder(
            in_dim=self.in_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(self.in_dim)
        self.classifier = nn.Linear(self.in_dim, num_classes)

    def forward(self, bag: Mapping[str, Tensor]) -> Tensor:
        instances, attention_mask = self._get_instances_and_mask(bag)
        batch_size = instances.shape[0]

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, instances], dim=1)
        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=instances.device)
        mask = torch.cat([cls_mask, attention_mask], dim=1)

        encoded = self.encoder(x, mask)
        cls_repr = self.norm(encoded[:, 0])
        return self.classifier(cls_repr)
