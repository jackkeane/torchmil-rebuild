"""DSMIL model (Li et al., 2021)."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from torchmil.nn import MaxPooling

from .base import MILModel


class DSMIL(MILModel):
    """Dual-stream MIL with max and attention-style critical instance pooling."""

    def __init__(
        self,
        in_shape: int | tuple[int, ...],
        num_classes: int = 2,
        hidden_dim: int = 256,
        criterion: nn.Module | None = None,
    ) -> None:
        super().__init__(in_shape=in_shape, criterion=criterion)
        self.num_classes = num_classes

        self.embed = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.instance_classifier = nn.Linear(hidden_dim, num_classes)
        self.max_pool = MaxPooling()
        self.bag_classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, bag: Mapping[str, Tensor]) -> Tensor:
        instances, attention_mask = self._get_instances_and_mask(bag)
        embedded = self.embed(instances)

        inst_logits = self.instance_classifier(embedded)
        inst_scores = inst_logits.max(dim=-1).values.masked_fill(~attention_mask, float("-inf"))

        critical_idx = inst_scores.argmax(dim=1)
        critical = torch.gather(
            embedded,
            1,
            critical_idx[:, None, None].expand(-1, 1, embedded.shape[-1]),
        ).squeeze(1)

        attn_scores = torch.einsum("bnd,bd->bn", embedded, critical)
        attn_scores = attn_scores.masked_fill(~attention_mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1).masked_fill(~attention_mask, 0.0)
        attn_repr = torch.bmm(attn_weights.unsqueeze(1), embedded).squeeze(1)

        max_repr = self.max_pool(embedded, attention_mask)
        bag_repr = torch.cat([max_repr, attn_repr], dim=1)
        return self.bag_classifier(bag_repr)
