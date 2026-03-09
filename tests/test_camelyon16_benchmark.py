from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_camelyon16_benchmark.py"


def _write_manifest(path: Path, rows: list[dict[str, str | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["features_path", "label", "split", "adjacency_path", "instance_labels_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _create_tiny_camelyon(root: Path, features: str = "UNI") -> Path:
    base = root / "dataset" / "patches_512"
    features_dir = base / "features" / f"features_{features}"
    labels_dir = base / "labels"
    features_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    for i in range(4):
        bag_name = f"train_{i}"
        label = i % 2
        n_instances = 3 + (i % 2)
        center = -1.0 if label == 0 else 1.0
        instances = (center + 0.1 * np.random.randn(n_instances, 8)).astype("float32")

        np.save(features_dir / f"{bag_name}.npy", instances)
        np.save(labels_dir / f"{bag_name}.npy", np.array([label], dtype="int64"))
        rows.append(
            {
                "features_path": str((features_dir / f"{bag_name}.npy").relative_to(root)),
                "label": label,
                "split": "train",
                "adjacency_path": "",
                "instance_labels_path": "",
            }
        )

    for i in range(2):
        bag_name = f"test_{i}"
        label = i % 2
        instances = ((-1.0 if label == 0 else 1.0) + 0.1 * np.random.randn(3, 8)).astype("float32")

        np.save(features_dir / f"{bag_name}.npy", instances)
        np.save(labels_dir / f"{bag_name}.npy", np.array([label], dtype="int64"))
        rows.append(
            {
                "features_path": str((features_dir / f"{bag_name}.npy").relative_to(root)),
                "label": label,
                "split": "test",
                "adjacency_path": "",
                "instance_labels_path": "",
            }
        )

    _write_manifest(base / f"manifest_{features}.csv", rows)
    return base


def test_camelyon16_benchmark_smoke(tmp_path: Path):
    _create_tiny_camelyon(tmp_path)
    results_dir = tmp_path / "results"

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--data-root",
        str(tmp_path),
        "--features",
        "UNI",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--models",
        "abmil",
        "clam",
        "transmil",
        "--results-dir",
        str(results_dir),
        "--device",
        "cpu",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert proc.returncode == 0, proc.stderr

    expected_files = [
        "camelyon16_abmil.json",
        "camelyon16_abmil.csv",
        "camelyon16_clam.json",
        "camelyon16_clam.csv",
        "camelyon16_transmil.json",
        "camelyon16_transmil.csv",
        "camelyon16_summary.json",
        "camelyon16_summary.csv",
    ]
    for name in expected_files:
        assert (results_dir / name).exists(), f"missing {name}"


def test_camelyon16_benchmark_missing_data_has_clear_error(tmp_path: Path):
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--data-root",
        str(tmp_path),
        "--features",
        "UNI",
        "--epochs",
        "1",
        "--models",
        "abmil",
        "--device",
        "cpu",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert proc.returncode == 2
    assert "CAMELYON16 benchmark data is unavailable or incomplete." in proc.stderr
    assert "Expected manifest:" in proc.stderr
    assert "--download" in proc.stderr
