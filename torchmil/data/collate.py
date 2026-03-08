"""Custom collate functions for variable-size MIL bags."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from .bag import validate_bag


def mil_collate_fn(bags: list[TensorDict]) -> TensorDict:
    """Collate variable-length MIL bags into a padded batch TensorDict."""
    if len(bags) == 0:
        raise ValueError("bags must be a non-empty list")

    for bag in bags:
        validate_bag(bag)

    lengths = torch.tensor([int(bag["length"].item()) for bag in bags], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch_size = len(bags)
    feat_dim = bags[0]["instances"].shape[1]
    device = bags[0]["instances"].device
    dtype = bags[0]["instances"].dtype

    # Validate feature dimension consistency across all bags
    for i, bag in enumerate(bags):
        if bag["instances"].shape[1] != feat_dim:
            raise ValueError(
                f"Feature dimension mismatch: bag 0 has D={feat_dim}, "
                f"but bag {i} has D={bag['instances'].shape[1]}"
            )

    instances = torch.zeros((batch_size, max_len, feat_dim), dtype=dtype, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    labels = torch.stack([bag["label"].to(device=device) for bag in bags], dim=0)

    has_adjacency = ["adjacency" in bag for bag in bags]
    has_instance_labels = ["instance_labels" in bag for bag in bags]
    if any(has_adjacency) and not all(has_adjacency):
        raise ValueError("Either all bags must include adjacency or none of them")
    if any(has_instance_labels) and not all(has_instance_labels):
        raise ValueError("Either all bags must include instance_labels or none of them")

    adjacency = None
    if all(has_adjacency):
        adjacency = torch.zeros((batch_size, max_len, max_len), dtype=bags[0]["adjacency"].dtype, device=device)

    instance_labels = None
    if all(has_instance_labels):
        instance_labels = torch.full(
            (batch_size, max_len),
            fill_value=-1,
            dtype=bags[0]["instance_labels"].dtype,
            device=device,
        )

    for idx, bag in enumerate(bags):
        n_inst = int(bag["length"].item())
        instances[idx, :n_inst] = bag["instances"]
        attention_mask[idx, :n_inst] = True
        if adjacency is not None:
            adjacency[idx, :n_inst, :n_inst] = bag["adjacency"]
        if instance_labels is not None:
            instance_labels[idx, :n_inst] = bag["instance_labels"]

    batch_data = {
        "instances": instances,
        "label": labels,
        "length": lengths.to(device=device),
        "attention_mask": attention_mask,
    }

    if adjacency is not None:
        batch_data["adjacency"] = adjacency
    if instance_labels is not None:
        batch_data["instance_labels"] = instance_labels

    return TensorDict(batch_data, batch_size=[batch_size])
