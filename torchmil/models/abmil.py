"""ABMIL model (Ilse et al., 2018)."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn

from torchmil.nn import AttentionPooling

from .base import MILModel


class ABMIL(MILModel):
    """Attention-based deep MIL classifier."""

    def __init__(
        self,
        in_shape: int | tuple[int, ...],
        num_classes: int = 2,
        hidden_dim: int = 256,
        criterion: nn.Module | None = None,
    ) -> None:
        super().__init__(in_shape=in_shape, criterion=criterion)
        self.num_classes = num_classes
        self.pool = AttentionPooling(in_dim=self.in_dim, hidden_dim=hidden_dim)
        self.classifier = nn.Linear(self.in_dim, num_classes)

    def forward(self, bag: Mapping[str, Tensor]) -> Tensor:
        instances, attention_mask = self._get_instances_and_mask(bag)
        bag_repr = self.pool(instances, attention_mask)
        return self.classifier(bag_repr)
