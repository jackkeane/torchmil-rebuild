"""Training/evaluation metrics for MIL classification."""

from __future__ import annotations

import torch
from torch import Tensor


def _labels_1d(targets: Tensor) -> Tensor:
    if targets.ndim > 1:
        targets = targets.view(-1)
    return targets.to(torch.long)


def _pred_labels(preds: Tensor) -> Tensor:
    if preds.ndim == 1:
        if preds.dtype.is_floating_point:
            return (preds >= 0.5).to(torch.long)
        return preds.to(torch.long)
    if preds.ndim == 2 and preds.shape[1] == 1:
        return (torch.sigmoid(preds[:, 0]) >= 0.5).to(torch.long)
    if preds.ndim == 2:
        return torch.argmax(preds, dim=1).to(torch.long)
    raise ValueError("preds must have shape [N], [N, 1], or [N, C]")


def accuracy(preds: Tensor, targets: Tensor) -> float:
    """Compute classification accuracy."""
    y_true = _labels_1d(targets)
    y_pred = _pred_labels(preds)
    return float((y_pred == y_true).float().mean().item())


def _binary_f1(y_pred: Tensor, y_true: Tensor, positive_class: int = 1) -> float:
    tp = ((y_pred == positive_class) & (y_true == positive_class)).sum().item()
    fp = ((y_pred == positive_class) & (y_true != positive_class)).sum().item()
    fn = ((y_pred != positive_class) & (y_true == positive_class)).sum().item()
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return 0.0
    return float((2 * tp) / denom)


def f1(preds: Tensor, targets: Tensor) -> float:
    """Compute F1 score (binary or macro-average for multiclass)."""
    y_true = _labels_1d(targets)
    y_pred = _pred_labels(preds)

    classes = torch.unique(y_true)
    if classes.numel() <= 2:
        return _binary_f1(y_pred, y_true, positive_class=1)

    scores = []
    for cls in classes.tolist():
        scores.append(_binary_f1(y_pred, y_true, positive_class=int(cls)))
    return float(sum(scores) / len(scores))


def _binary_auc(scores: Tensor, y_true: Tensor) -> float:
    y_true = y_true.to(torch.long)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    sorted_true = y_true[order]
    n = scores.numel()

    ranks = torch.arange(1, n + 1, dtype=torch.float32, device=scores.device)
    tie_start = 0
    while tie_start < n:
        tie_end = tie_start + 1
        while tie_end < n and sorted_scores[tie_end] == sorted_scores[tie_start]:
            tie_end += 1
        if tie_end - tie_start > 1:
            avg_rank = ranks[tie_start:tie_end].mean()
            ranks[tie_start:tie_end] = avg_rank
        tie_start = tie_end

    sum_pos_ranks = ranks[sorted_true == 1].sum().item()
    auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def auroc(preds: Tensor, targets: Tensor) -> float:
    """Compute AUROC (binary or macro one-vs-rest for multiclass)."""
    y_true = _labels_1d(targets)

    if preds.ndim == 1:
        scores = preds
        if not scores.dtype.is_floating_point:
            scores = scores.to(torch.float32)
        if scores.min() < 0.0 or scores.max() > 1.0:
            scores = torch.sigmoid(scores)
        return _binary_auc(scores, y_true)

    if preds.ndim == 2 and preds.shape[1] == 1:
        scores = torch.sigmoid(preds[:, 0])
        return _binary_auc(scores, y_true)

    if preds.ndim == 2:
        probs = torch.softmax(preds, dim=1)
        num_classes = probs.shape[1]
        if num_classes == 2:
            return _binary_auc(probs[:, 1], y_true)

        scores = []
        for cls in range(num_classes):
            cls_true = (y_true == cls).to(torch.long)
            scores.append(_binary_auc(probs[:, cls], cls_true))
        return float(sum(scores) / len(scores))

    raise ValueError("preds must have shape [N], [N, 1], or [N, C]")


def performance(preds: Tensor, targets: Tensor) -> float:
    """Composite metric: (ACC + F1 + AUROC) / 3."""
    acc = accuracy(preds, targets)
    f1_score = f1(preds, targets)
    auc = auroc(preds, targets)
    return float((acc + f1_score + auc) / 3.0)
