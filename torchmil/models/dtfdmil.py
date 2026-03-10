"""DTFDMIL model (Zhang et al., 2022)."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from torchmil.nn import GatedAttention

from .base import MILModel


class DTFDMIL(MILModel):
    """Double-tier feature distillation MIL model.

    Args:
        in_shape: Input feature shape or feature dimension.
        num_classes: Number of bag-level classes.
        hidden_dim: Hidden projection size.
        top_k: Number of top-attended instances for local distillation.
        criterion: Optional loss module (defaults to CrossEntropyLoss).
    """

    def __init__(
        self,
        in_shape: int | tuple[int, ...],
        num_classes: int = 2,
        hidden_dim: int = 256,
        top_k: int = 4,
        criterion: nn.Module | None = None,
    ) -> None:
        super().__init__(in_shape=in_shape, criterion=criterion)
        self.num_classes = num_classes
        self.top_k = top_k

        self.project = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attention = GatedAttention(in_dim=hidden_dim, hidden_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, bag: Mapping[str, Tensor]) -> Tensor:
        instances, attention_mask = self._get_instances_and_mask(bag)
        feat = self.project(instances)

        attention, global_repr = self.attention(feat, attention_mask)

        k = min(self.top_k, feat.shape[1])
        top_idx = torch.topk(attention.masked_fill(~attention_mask, float("-inf")), k=k, dim=1).indices
        top_feat = torch.gather(feat, 1, top_idx.unsqueeze(-1).expand(-1, -1, feat.shape[-1]))
        local_repr = top_feat.mean(dim=1)

        distill_repr = torch.cat([global_repr, local_repr], dim=1)
        return self.classifier(distill_repr)
