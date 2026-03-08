"""CAMELYON16 MIL dataset backed by pre-extracted features."""

from __future__ import annotations

from pathlib import Path

from .base import ProcessedMILDataset


class Camelyon16MIL(ProcessedMILDataset):
    """CAMELYON16 MIL dataset.

    Local directory layout:

    - ``<root>/camelyon16/<features>/manifest.csv``
    - ``<root>/camelyon16/<features>/...`` feature files referenced in the manifest
    """

    DEFAULT_REPO_ID = "torchmil/camelyon16"

    def __init__(
        self,
        root: str | Path,
        features: str = "UNI",
        split: str | None = "train",
        download: bool = False,
        revision: str | None = None,
        repo_id: str = DEFAULT_REPO_ID,
    ) -> None:
        self.base_root = Path(root).expanduser().resolve()
        self.features = features
        self.revision = revision
        self.repo_id = repo_id

        dataset_root = self.base_root / "camelyon16" / features
        manifest_path = dataset_root / "manifest.csv"

        if not manifest_path.exists() and download:
            self.download()

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"CAMELYON16 manifest not found at {manifest_path}. "
                "Pass download=True to fetch it from HuggingFace."
            )

        super().__init__(root=dataset_root, split=split, manifest_file="manifest.csv")

    def download(self) -> Path:
        """Download CAMELYON16 features from HuggingFace Hub (optional)."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise ImportError(
                "huggingface_hub is required for download=True. "
                "Install it with: pip install huggingface_hub"
            ) from error

        local_root = self.base_root / "camelyon16"
        local_root.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=local_root,
            allow_patterns=[f"{self.features}/**"],
            local_dir_use_symlinks=False,
        )
        return local_root
