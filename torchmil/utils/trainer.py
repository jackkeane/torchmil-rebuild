"""Simple training utility for MIL models."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .metrics import accuracy, auroc, f1, performance


class Trainer:
    """Utility class for model training and validation."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.history: list[dict[str, float | int | bool]] = []

    def _move_batch_to_device(self, batch):
        if hasattr(batch, "to"):
            return batch.to(self.device)
        if isinstance(batch, Mapping):
            return {k: self._move_batch_to_device(v) for k, v in batch.items()}
        if isinstance(batch, Tensor):
            return batch.to(self.device)
        return batch

    def _run_epoch(self, dataloader, training: bool) -> dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_batches = 0
        all_logits = []
        all_targets = []

        for batch in dataloader:
            batch = self._move_batch_to_device(batch)
            labels = batch["label"].long()

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                logits = self.model(batch)
                loss = self.model.criterion(logits, labels)
                if training:
                    loss.backward()
                    self.optimizer.step()

            total_loss += float(loss.detach().item())
            total_batches += 1
            all_logits.append(logits.detach())
            all_targets.append(labels.detach())

        if total_batches == 0:
            raise ValueError("dataloader yielded no batches")

        logits_cat = torch.cat(all_logits, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        mean_loss = total_loss / total_batches

        return {
            "loss": float(mean_loss),
            "accuracy": accuracy(logits_cat, targets_cat),
            "f1": f1(logits_cat, targets_cat),
            "auroc": auroc(logits_cat, targets_cat),
            "performance": performance(logits_cat, targets_cat),
        }

    def train(
        self,
        dataloader,
        epochs: int,
        val_dataloader=None,
        patience: int | None = None,
        verbose: bool = False,
    ) -> list[dict[str, float | int | bool]]:
        if epochs <= 0:
            raise ValueError("epochs must be > 0")
        if patience is not None and patience < 0:
            raise ValueError("patience must be >= 0")

        self.history = []
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch_idx in range(epochs):
            train_stats = self._run_epoch(dataloader, training=True)
            log: dict[str, float | int | bool] = {
                "epoch": epoch_idx + 1,
                "train_loss": train_stats["loss"],
                "train_accuracy": train_stats["accuracy"],
                "train_f1": train_stats["f1"],
                "train_auroc": train_stats["auroc"],
                "train_performance": train_stats["performance"],
            }

            if val_dataloader is not None:
                val_stats = self._run_epoch(val_dataloader, training=False)
                log.update(
                    {
                        "val_loss": val_stats["loss"],
                        "val_accuracy": val_stats["accuracy"],
                        "val_f1": val_stats["f1"],
                        "val_auroc": val_stats["auroc"],
                        "val_performance": val_stats["performance"],
                    }
                )

                if patience is not None:
                    if val_stats["loss"] < best_val_loss:
                        best_val_loss = val_stats["loss"]
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        if epochs_without_improvement > patience:
                            log["early_stopped"] = True
                            self.history.append(log)
                            break

            self.history.append(log)

            if verbose:
                msg = (
                    f"[epoch {log['epoch']}] "
                    f"train_loss={log['train_loss']:.4f} "
                    f"train_perf={log['train_performance']:.4f}"
                )
                if val_dataloader is not None:
                    msg += (
                        f" val_loss={log['val_loss']:.4f} "
                        f"val_perf={log['val_performance']:.4f}"
                    )
                if log.get("early_stopped", False):
                    msg += " [early_stopped]"
                print(msg)

        return self.history
