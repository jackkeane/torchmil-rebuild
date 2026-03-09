"""Dataset abstractions for pre-processed MIL features."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from torchmil.data import make_bag


class ProcessedMILDataset(Dataset):
    """Base dataset for MIL features extracted and stored on disk.

    Expected storage format is a CSV manifest with at least these columns:
    - ``features_path``: relative or absolute file path to a ``[N_i, D]`` tensor
    - ``label``: bag-level label

    Optional columns:
    - ``split``: split name (for example ``train`` or ``test``)
    - ``adjacency_path``: path to a ``[N_i, N_i]`` adjacency tensor
    - ``instance_labels_path``: path to a ``[N_i]`` tensor
    """

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        manifest_file: str = "manifest.csv",
        samples: list[dict[str, Any]] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()

        if samples is None:
            # Accept absolute path OR relative-to-root path
            _mf = Path(manifest_file)
            manifest_path = _mf if _mf.is_absolute() else self.root / manifest_file
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
            loaded_samples = self._load_manifest(manifest_path)
        else:
            loaded_samples = [dict(sample) for sample in samples]

        normalized_samples = [self._normalize_sample(sample) for sample in loaded_samples]
        if split is not None:
            normalized_samples = [
                sample for sample in normalized_samples if sample.get("split") == split
            ]

        if len(normalized_samples) == 0:
            split_msg = f" for split={split!r}" if split is not None else ""
            raise ValueError(f"No samples found in dataset{split_msg}")

        self.split = split
        self._samples = normalized_samples
        self.data_dim = self._infer_data_dim()

    @staticmethod
    def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]

        if len(rows) == 0:
            raise ValueError(f"Manifest is empty: {manifest_path}")
        if "features_path" not in rows[0] or "label" not in rows[0]:
            raise ValueError(
                "Manifest must include at least 'features_path' and 'label' columns"
            )

        return rows

    def _normalize_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(sample)
        normalized["features_path"] = self._resolve_path(sample["features_path"])

        for key in ("adjacency_path", "instance_labels_path"):
            if key in sample and sample[key] not in (None, ""):
                normalized[key] = self._resolve_path(sample[key])
            else:
                normalized.pop(key, None)

        normalized["label"] = self._parse_label(sample["label"])
        return normalized

    def _resolve_path(self, path_like: str | Path) -> Path:
        path = Path(path_like)
        if not path.is_absolute():
            path = self.root / path
        return path

    @staticmethod
    def _parse_label(raw_label: Any) -> Any:
        if isinstance(raw_label, str):
            if raw_label.isdigit() or (raw_label.startswith("-") and raw_label[1:].isdigit()):
                return int(raw_label)
            try:
                return float(raw_label)
            except ValueError:
                return raw_label
        return raw_label

    @staticmethod
    @lru_cache(maxsize=256)
    def _cached_load(path: str) -> torch.Tensor:
        file_path = Path(path)
        suffix = file_path.suffix.lower()

        if suffix in {".pt", ".pth"}:
            loaded = torch.load(file_path, map_location="cpu")
            if not isinstance(loaded, torch.Tensor):
                raise TypeError(f"Expected tensor in {file_path}, got {type(loaded).__name__}")
            return loaded

        if suffix == ".npy":
            import numpy as np

            return torch.from_numpy(np.load(file_path))

        raise ValueError(
            f"Unsupported feature format for {file_path}. Use .pt/.pth/.npy"
        )

    def _load_tensor(self, path: Path) -> torch.Tensor:
        tensor = self._cached_load(str(path))
        if tensor.ndim != 2 and path == self._samples[0]["features_path"]:
            raise ValueError(f"Expected a 2D feature tensor at {path}, got shape {tuple(tensor.shape)}")
        return tensor

    def _infer_data_dim(self) -> int:
        first_path = self._samples[0]["features_path"]
        first_tensor = self._load_tensor(first_path)
        if first_tensor.ndim != 2:
            raise ValueError(
                f"Expected features with shape [N_i, D], got {tuple(first_tensor.shape)}"
            )
        return int(first_tensor.shape[1])

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        sample = self._samples[index]
        instances = self._cached_load(str(sample["features_path"]))
        if instances.ndim != 2:
            raise ValueError(
                f"Expected features with shape [N_i, D], got {tuple(instances.shape)} "
                f"for sample index {index}"
            )

        adjacency = None
        if "adjacency_path" in sample:
            adjacency = self._cached_load(str(sample["adjacency_path"]))

        instance_labels = None
        if "instance_labels_path" in sample:
            instance_labels = self._cached_load(str(sample["instance_labels_path"]))
            if instance_labels.ndim != 1:
                raise ValueError(
                    "Expected instance labels with shape [N_i], "
                    f"got {tuple(instance_labels.shape)} for sample index {index}"
                )

        return make_bag(
            instances=instances,
            label=sample["label"],
            adjacency=adjacency,
            instance_labels=instance_labels,
        )
