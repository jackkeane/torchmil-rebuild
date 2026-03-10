"""CAMELYON16 MIL dataset backed by pre-extracted features."""

from __future__ import annotations

import csv
import tarfile
from pathlib import Path

import numpy as np

from .base import ProcessedMILDataset

# Real HuggingFace repo
_DEFAULT_REPO_ID = "torchmil/Camelyon16_MIL"
_PATCH_SIZE = "patches_512"


def _extract_tar(tar_path: Path, dest: Path) -> None:
    """Extract tar.gz safely with path traversal protection."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as t:
        # Python 3.12+: use filter='data' to prevent path traversal (CVE-2007-4559)
        t.extractall(dest, filter="data")


def _build_manifest(root: Path, features: str, patch_size: str = _PATCH_SIZE) -> Path:
    """Build manifest.csv from splits.csv + labels + feature files.

    Supports both layouts:
    1) dataset/patches_512/labels and dataset/patches_512/features/features_<name>
    2) flat tar extraction where labels are directly under dataset/patches_512
       and features are directly under dataset/patches_512/features
    """
    splits_csv = root / "dataset" / "splits.csv"

    patch_root = root / "dataset" / patch_size
    labels_dir = patch_root / "labels"
    if not labels_dir.exists():
        labels_dir = patch_root

    features_dir = patch_root / "features" / f"features_{features}"
    if not features_dir.exists():
        features_dir = patch_root / "features"

    manifest_path = patch_root / f"manifest_{features}.csv"

    if not splits_csv.exists():
        raise FileNotFoundError(f"splits.csv not found at {splits_csv}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")
    if not features_dir.exists():
        raise FileNotFoundError(f"Features dir not found: {features_dir}")

    with splits_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        splits = {row["bag_name"]: row["split"] for row in reader}

    rows: list[dict] = []
    for bag_name, split in splits.items():
        feat_path = features_dir / f"{bag_name}.npy"
        lbl_path = labels_dir / f"{bag_name}.npy"
        if not feat_path.exists() or not lbl_path.exists():
            continue

        raw_label = np.load(str(lbl_path)).flat[0]
        label = int(raw_label)

        rows.append({
            "features_path": str(feat_path.relative_to(root)),
            "label": str(label),
            "split": split,
            "adjacency_path": "",
            "instance_labels_path": "",
        })

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["features_path", "label", "split", "adjacency_path", "instance_labels_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


class Camelyon16MIL(ProcessedMILDataset):
    """CAMELYON16 MIL dataset.

    After download/extraction, expects the following layout under ``root``:

    .. code-block:: text

        <root>/
          dataset/
            splits.csv
            patches_512/
              labels/         ← {bag_name}.npy
              features/
                features_UNI/ ← {bag_name}.npy
    """

    DEFAULT_REPO_ID = _DEFAULT_REPO_ID

    def __init__(
        self,
        root: str | Path,
        features: str = "UNI",
        split: str | None = "train",
        download: bool = False,
        patch_size: str = _PATCH_SIZE,
        revision: str | None = None,
        repo_id: str = _DEFAULT_REPO_ID,
    ) -> None:
        self.base_root = Path(root).expanduser().resolve()
        self.features = features
        self.patch_size = patch_size
        self.revision = revision
        self.repo_id = repo_id

        manifest_path = self.base_root / "dataset" / patch_size / f"manifest_{features}.csv"

        if not manifest_path.exists() and download:
            self._download_and_extract()

        # Try to build manifest whenever enough files exist (downloaded or pre-populated)
        if not manifest_path.exists():
            try:
                manifest_path = _build_manifest(self.base_root, features, patch_size)
            except FileNotFoundError:
                expected = self.base_root / "dataset" / patch_size / "features"
                raise FileNotFoundError(
                    f"CAMELYON16 data not found for features='{features}'.\n"
                    f"Expected under: {expected}\n"
                    f"Re-run with download=True to fetch from HuggingFace:\n"
                    f"  Camelyon16MIL(root='{root}', features='{features}', download=True)"
                )

        # Use ProcessedMILDataset with inline samples so we don't need root-relative paths.
        super().__init__(
            root=self.base_root,
            split=split,
            manifest_file=str(manifest_path),
        )

    def _download_and_extract(self) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as err:
            raise ImportError(
                "huggingface_hub is required for download=True. "
                "Install with: pip install huggingface_hub"
            ) from err

        dest = self.base_root

        def _dl(hf_path: str) -> Path:
            local = hf_hub_download(
                repo_id=self.repo_id,
                filename=hf_path,
                repo_type="dataset",
                revision=self.revision,
                cache_dir=str(dest / ".hf_cache"),
            )
            return Path(local)

        # splits.csv (tiny)
        splits_csv_src = _dl("dataset/splits.csv")
        splits_dst = dest / "dataset" / "splits.csv"
        splits_dst.parent.mkdir(parents=True, exist_ok=True)
        splits_dst.write_bytes(splits_csv_src.read_bytes())

        # labels.tar.gz → extract to dest/dataset/patches_512/labels/
        labels_tar = _dl(f"dataset/{self.patch_size}/labels.tar.gz")
        _extract_tar(labels_tar, dest / "dataset" / self.patch_size)

        # features_{features}.tar.gz → extract to dest/dataset/patches_512/features/
        features_tar = _dl(f"dataset/{self.patch_size}/features/features_{self.features}.tar.gz")
        _extract_tar(features_tar, dest / "dataset" / self.patch_size / "features")

    def download(self) -> Path:
        """Download and extract dataset from HuggingFace."""
        self._download_and_extract()
        return _build_manifest(self.base_root, self.features, self.patch_size)
