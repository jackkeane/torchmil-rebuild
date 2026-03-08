"""Base MIL model interface."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn


class MILModel(nn.Module):
    """Base class for MIL bag-level models."""

    def __init__(
        self,
        in_shape: int | tuple[int, ...],
        criterion: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if isinstance(in_shape, int):
            in_dim = in_shape
        else:
            if len(in_shape) == 0:
                raise ValueError("in_shape cannot be empty")
            in_dim = int(in_shape[-1])

        if in_dim <= 0:
            raise ValueError("in_shape must define a positive feature dimension")

        self.in_shape = in_shape
        self.in_dim = in_dim
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    def _get_instances_and_mask(self, bag: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        instances = bag["instances"]
        if instances.ndim != 3:
            raise ValueError("bag['instances'] must have shape [B, N, D]")

        batch_size, num_instances, _ = instances.shape
        attention_mask = bag.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, num_instances),
                dtype=torch.bool,
                device=instances.device,
            )

        if attention_mask.shape != (batch_size, num_instances):
            raise ValueError("bag['attention_mask'] must have shape [B, N]")
        if attention_mask.dtype is not torch.bool:
            attention_mask = attention_mask.to(torch.bool)

        return instances, attention_mask

    def forward(self, bag: Mapping[str, Tensor]) -> Tensor:  # pragma: no cover - abstract contract
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, bag: Mapping[str, Tensor]) -> Tensor:
        logits = self.forward(bag)
        if logits.ndim != 2:
            raise ValueError("model logits must have shape [B, C]")

        if logits.shape[1] == 1:
            return (torch.sigmoid(logits[:, 0]) >= 0.5).to(torch.long)
        return torch.argmax(logits, dim=1)

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), Path(path))

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> "MILModel":
        state = torch.load(Path(path), map_location=map_location)
        self.load_state_dict(state)
        return self
