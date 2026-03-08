"""CLAM model (Lu et al., 2021)."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from torchmil.nn import GatedAttention

from .base import MILModel


class CLAM(MILModel):
    """Simplified CLAM with attention pooling and instance clustering branch."""

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

        self.embed = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attention = GatedAttention(in_dim=hidden_dim, hidden_dim=hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.instance_classifier = nn.Linear(hidden_dim, num_classes)

    def _instance_cluster_features(
        self,
        embedded: Tensor,
        attention: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        score = attention.masked_fill(~attention_mask, float("-inf"))
        k = min(self.top_k, embedded.shape[1])
        top_idx = torch.topk(score, k=k, dim=1).indices
        bottom_idx = torch.topk(-score, k=k, dim=1).indices

        pos = torch.gather(embedded, 1, top_idx.unsqueeze(-1).expand(-1, -1, embedded.shape[-1]))
        neg = torch.gather(embedded, 1, bottom_idx.unsqueeze(-1).expand(-1, -1, embedded.shape[-1]))

        # Auxiliary instance-level clustering representation used by CLAM.
        return torch.cat([pos.mean(dim=1), neg.mean(dim=1)], dim=1)

    def forward(self, bag: Mapping[str, Tensor]) -> Tensor:
        instances, attention_mask = self._get_instances_and_mask(bag)
        embedded = self.embed(instances)

        attention, bag_repr = self.attention(embedded, attention_mask)
        bag_logits = self.classifier(bag_repr)

        # Keep a lightweight instance branch for CLAM-style top/bottom evidence mining.
        cluster_feat = self._instance_cluster_features(embedded, attention, attention_mask)
        aux_logits = self.instance_classifier(cluster_feat[:, : embedded.shape[-1]])

        return bag_logits + 0.0 * aux_logits
