#!/usr/bin/env python3
"""Run CAMELYON16 benchmark reproduction for ABMIL/CLAM/TransMIL."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Allow running as a plain script from source checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchmil.data import mil_collate_fn
from torchmil.datasets import Camelyon16MIL
from torchmil.models import ABMIL, CLAM, TransMIL
from torchmil.utils import Trainer, accuracy, auroc, f1, performance

DEFAULT_MODELS = ("abmil", "clam", "transmil")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            logits = model(batch)
            labels = batch["label"].long()
            all_logits.append(logits.detach().cpu())
            all_targets.append(labels.detach().cpu())

    if not all_logits:
        raise ValueError("Test dataloader yielded no batches")

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    return {
        "accuracy": accuracy(logits_cat, targets_cat),
        "auroc": auroc(logits_cat, targets_cat),
        "f1": f1(logits_cat, targets_cat),
        "performance": performance(logits_cat, targets_cat),
    }


def build_model(name: str, in_dim: int, num_classes: int) -> torch.nn.Module:
    model_name = name.lower()
    if model_name == "abmil":
        return ABMIL(in_shape=in_dim, num_classes=num_classes)
    if model_name == "clam":
        return CLAM(in_shape=in_dim, num_classes=num_classes)
    if model_name == "transmil":
        return TransMIL(in_shape=in_dim, num_classes=num_classes)
    raise ValueError(f"Unsupported model: {name}")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_single_metric_csv(path: Path, metrics: dict[str, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def write_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fields = ["model", "accuracy", "auroc", "f1", "performance"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--features", type=str, default="UNI")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--repo-id", type=str, default="torchmil/Camelyon16_MIL")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=list(DEFAULT_MODELS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args(argv)


def load_dataset(args: argparse.Namespace) -> tuple[Camelyon16MIL, Camelyon16MIL]:
    try:
        train_ds = Camelyon16MIL(
            root=args.data_root,
            features=args.features,
            split=args.train_split,
            download=args.download,
            revision=args.revision,
            repo_id=args.repo_id,
        )
        test_ds = Camelyon16MIL(
            root=args.data_root,
            features=args.features,
            split=args.test_split,
            download=False,
            revision=args.revision,
            repo_id=args.repo_id,
        )
    except (FileNotFoundError, ValueError, ImportError) as error:
        expected_manifest = (
            args.data_root.expanduser().resolve()
            / "dataset"
            / "patches_512"
            / f"manifest_{args.features}.csv"
        )
        raise RuntimeError(
            "CAMELYON16 benchmark data is unavailable or incomplete.\n"
            f"Expected manifest: {expected_manifest}\n"
            "How to proceed:\n"
            "1) Download/setup data under <data-root>/dataset/patches_512/ with manifest_<features>.csv and feature files.\n"
            "2) Or re-run with --download (requires `pip install huggingface_hub`).\n"
            f"3) Current flags: --data-root {args.data_root} --features {args.features} --train-split {args.train_split} --test-split {args.test_split}\n"
            f"Original error: {error}"
        ) from error

    return train_ds, test_ds


def run_benchmark(args: argparse.Namespace) -> int:
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    set_seed(args.seed)
    device = torch.device(args.device)

    train_ds, test_ds = load_dataset(args)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mil_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mil_collate_fn,
    )

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | str]] = []
    settings = {
        "batch_size": args.batch_size,
        "optimizer": "Adam",
        "lr": args.lr,
        "epochs": args.epochs,
        "features": args.features,
        "train_split": args.train_split,
        "test_split": args.test_split,
    }

    num_classes = 2
    for model_name in args.models:
        model = build_model(model_name, in_dim=train_ds.data_dim, num_classes=num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        trainer = Trainer(model=model, optimizer=optimizer, device=device)

        history = trainer.train(train_loader, epochs=args.epochs)
        test_metrics = evaluate_model(trainer.model, test_loader, device=device)

        row: dict[str, float | str] = {"model": model_name.upper(), **test_metrics}
        summary_rows.append(row)

        model_payload = {
            "model": model_name.upper(),
            "settings": settings,
            "train_final": history[-1],
            "test_metrics": test_metrics,
        }

        write_json(results_dir / f"camelyon16_{model_name}.json", model_payload)
        write_single_metric_csv(results_dir / f"camelyon16_{model_name}.csv", row)

    write_summary_csv(results_dir / "camelyon16_summary.csv", summary_rows)
    write_json(results_dir / "camelyon16_summary.json", {"settings": settings, "results": summary_rows})

    print(f"Saved benchmark results to {results_dir.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_benchmark(args)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
