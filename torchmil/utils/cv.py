"""Cross-validation utilities."""

from __future__ import annotations

import torch


def kfold_split_indices(
    n_samples: int,
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Create K-fold train/validation index splits."""
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_splits > n_samples:
        raise ValueError("n_splits cannot be greater than n_samples")

    indices = list(range(n_samples))
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_samples, generator=g).tolist()
        indices = [indices[i] for i in perm]

    fold_sizes = [n_samples // n_splits] * n_splits
    for i in range(n_samples % n_splits):
        fold_sizes[i] += 1

    splits: list[tuple[list[int], list[int]]] = []
    current = 0
    for fold_size in fold_sizes:
        start, end = current, current + fold_size
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        splits.append((train_idx, val_idx))
        current = end

    return splits
