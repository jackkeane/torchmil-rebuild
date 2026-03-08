"""TensorDict-based bag representation for MIL data."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict


def make_bag(
    instances: torch.Tensor,
    label: torch.Tensor | int | float,
    adjacency: torch.Tensor | None = None,
    instance_labels: torch.Tensor | None = None,
) -> TensorDict:
    """Create a MIL bag as a TensorDict.

    Args:
        instances: Instance features of shape ``[N_i, D]``.
        label: Bag-level label.
        adjacency: Optional adjacency matrix of shape ``[N_i, N_i]``.
        instance_labels: Optional instance labels of shape ``[N_i]``.

    Returns:
        TensorDict representing one bag.
    """
    if not torch.is_tensor(instances):
        raise TypeError("instances must be a torch.Tensor")
    if instances.ndim != 2:
        raise ValueError("instances must have shape [N_i, D]")

    num_instances = instances.shape[0]
    device = instances.device

    if not torch.is_tensor(label):
        label = torch.tensor(label)

    bag_data: dict[str, Any] = {
        "instances": instances,
        "label": label.to(device=device),
        "length": torch.tensor(num_instances, device=device, dtype=torch.long),
    }

    if adjacency is not None:
        if not torch.is_tensor(adjacency):
            raise TypeError("adjacency must be a torch.Tensor")
        if adjacency.shape != (num_instances, num_instances):
            raise ValueError("adjacency must have shape [N_i, N_i]")
        bag_data["adjacency"] = adjacency.to(device=device)

    if instance_labels is not None:
        if not torch.is_tensor(instance_labels):
            raise TypeError("instance_labels must be a torch.Tensor")
        if instance_labels.shape != (num_instances,):
            raise ValueError("instance_labels must have shape [N_i]")
        bag_data["instance_labels"] = instance_labels.to(device=device)

    return TensorDict(bag_data, batch_size=[])


def validate_bag(bag: TensorDict) -> None:
    """Validate a bag TensorDict schema."""
    required_keys = {"instances", "label", "length"}
    missing = required_keys - set(bag.keys())
    if missing:
        raise KeyError(f"bag is missing required keys: {sorted(missing)}")

    instances = bag["instances"]
    if instances.ndim != 2:
        raise ValueError("bag['instances'] must have shape [N_i, D]")

    expected_n = instances.shape[0]
    if bag["length"].item() != expected_n:
        raise ValueError("bag['length'] must equal number of instances")

    if "adjacency" in bag:
        if bag["adjacency"].shape != (expected_n, expected_n):
            raise ValueError("bag['adjacency'] must have shape [N_i, N_i]")

    if "instance_labels" in bag:
        if bag["instance_labels"].shape != (expected_n,):
            raise ValueError("bag['instance_labels'] must have shape [N_i]")
