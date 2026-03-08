import csv
import sys
import types

import pytest
import torch

from torchmil.data import validate_bag
from torchmil.datasets import Camelyon16MIL, ProcessedMILDataset


def _write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["features_path", "label", "split", "adjacency_path", "instance_labels_path"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_processed_dataset_len_getitem_data_dim(tmp_path):
    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True)

    feats_a = torch.randn(3, 8)
    feats_b = torch.randn(5, 8)
    adj_a = torch.eye(3)
    inst_lbl_a = torch.tensor([0, 1, 1])

    torch.save(feats_a, features_dir / "a.pt")
    torch.save(feats_b, features_dir / "b.pt")
    torch.save(adj_a, features_dir / "a_adj.pt")
    torch.save(inst_lbl_a, features_dir / "a_inst.pt")

    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "features_path": "features/a.pt",
                "label": "1",
                "split": "train",
                "adjacency_path": "features/a_adj.pt",
                "instance_labels_path": "features/a_inst.pt",
            },
            {
                "features_path": "features/b.pt",
                "label": "0",
                "split": "test",
                "adjacency_path": "",
                "instance_labels_path": "",
            },
        ],
    )

    dataset = ProcessedMILDataset(root=tmp_path, split="train")

    assert len(dataset) == 1
    assert dataset.data_dim == 8

    bag = dataset[0]
    validate_bag(bag)

    assert bag["instances"].shape == (3, 8)
    assert int(bag["label"].item()) == 1
    assert bag["adjacency"].shape == (3, 3)
    assert torch.equal(bag["instance_labels"], inst_lbl_a)


def test_processed_dataset_returns_valid_bags_for_all_items(tmp_path):
    feats_dir = tmp_path / "f"
    feats_dir.mkdir(parents=True)

    torch.save(torch.randn(2, 4), feats_dir / "x.pt")
    torch.save(torch.randn(6, 4), feats_dir / "y.pt")

    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "features_path": "f/x.pt",
                "label": "0",
                "split": "train",
                "adjacency_path": "",
                "instance_labels_path": "",
            },
            {
                "features_path": "f/y.pt",
                "label": "1",
                "split": "train",
                "adjacency_path": "",
                "instance_labels_path": "",
            },
        ],
    )

    dataset = ProcessedMILDataset(root=tmp_path, split="train")
    assert len(dataset) == 2

    bag0 = dataset[0]
    bag1 = dataset[1]
    validate_bag(bag0)
    validate_bag(bag1)

    assert bag0["instances"].shape[1] == 4
    assert bag1["instances"].shape[1] == 4


def test_camelyon16_dataset_structure_and_split(tmp_path):
    base = tmp_path / "camelyon16" / "UNI"
    feats = base / "features"
    feats.mkdir(parents=True)

    torch.save(torch.randn(4, 16), feats / "slide_1.pt")
    torch.save(torch.randn(2, 16), feats / "slide_2.pt")

    manifest = base / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "features_path": "features/slide_1.pt",
                "label": "1",
                "split": "train",
                "adjacency_path": "",
                "instance_labels_path": "",
            },
            {
                "features_path": "features/slide_2.pt",
                "label": "0",
                "split": "test",
                "adjacency_path": "",
                "instance_labels_path": "",
            },
        ],
    )

    train_ds = Camelyon16MIL(root=tmp_path, features="UNI", split="train", download=False)
    test_ds = Camelyon16MIL(root=tmp_path, features="UNI", split="test", download=False)

    assert len(train_ds) == 1
    assert len(test_ds) == 1
    assert train_ds.data_dim == 16
    assert test_ds.data_dim == 16

    bag = train_ds[0]
    validate_bag(bag)
    assert bag["instances"].shape == (4, 16)


def test_processed_dataset_missing_columns_raises(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("foo,bar\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="features_path"):
        ProcessedMILDataset(root=tmp_path)


def test_processed_dataset_empty_split_raises(tmp_path):
    feats_dir = tmp_path / "f"
    feats_dir.mkdir(parents=True)
    torch.save(torch.randn(2, 4), feats_dir / "x.pt")

    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [{
            "features_path": "f/x.pt",
            "label": "0",
            "split": "train",
            "adjacency_path": "",
            "instance_labels_path": "",
        }],
    )

    with pytest.raises(ValueError, match="No samples found"):
        ProcessedMILDataset(root=tmp_path, split="test")


def test_processed_dataset_unsupported_extension_raises(tmp_path):
    fdir = tmp_path / "f"
    fdir.mkdir(parents=True)
    (fdir / "x.txt").write_text("not a tensor", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [{
            "features_path": "f/x.txt",
            "label": "0",
            "split": "train",
            "adjacency_path": "",
            "instance_labels_path": "",
        }],
    )

    with pytest.raises(ValueError, match="Unsupported feature format"):
        ProcessedMILDataset(root=tmp_path, split="train")


def test_processed_dataset_npy_loading(tmp_path):
    np = pytest.importorskip("numpy")
    fdir = tmp_path / "f"
    fdir.mkdir(parents=True)
    np.save(fdir / "x.npy", np.random.randn(3, 5).astype("float32"))
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [{
            "features_path": "f/x.npy",
            "label": "1",
            "split": "train",
            "adjacency_path": "",
            "instance_labels_path": "",
        }],
    )

    ds = ProcessedMILDataset(root=tmp_path, split="train")
    bag = ds[0]
    assert bag["instances"].shape == (3, 5)


def test_camelyon16_download_uses_base_root_not_dataset_root(tmp_path, monkeypatch):
    base = tmp_path / "camelyon16" / "UNI"
    feats = base / "features"
    feats.mkdir(parents=True)
    torch.save(torch.randn(1, 2), feats / "s.pt")
    _write_manifest(
        base / "manifest.csv",
        [{
            "features_path": "features/s.pt",
            "label": "1",
            "split": "train",
            "adjacency_path": "",
            "instance_labels_path": "",
        }],
    )

    ds = Camelyon16MIL(root=tmp_path, features="UNI", split="train", download=False)

    called = {}

    def fake_snapshot_download(**kwargs):
        called.update(kwargs)
        return str(kwargs["local_dir"])

    fake_hf = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    out = ds.download()
    assert out == tmp_path / "camelyon16"
    assert called["local_dir"] == tmp_path / "camelyon16"
    assert called["allow_patterns"] == ["UNI/**"]
