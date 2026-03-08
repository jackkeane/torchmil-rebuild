import torch
from torch import nn
from torch.utils.data import DataLoader

from torchmil.data import make_bag, mil_collate_fn
from torchmil.utils import Trainer, accuracy, auroc, f1, kfold_split_indices, performance


class TinyMILModel(nn.Module):
    def __init__(self, in_dim: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, bag):
        instances = bag["instances"]  # [B, N, D]
        mask = bag["attention_mask"].unsqueeze(-1).float()  # [B, N, 1]
        pooled = (instances * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


def _make_synthetic_bags(num_bags: int = 8, in_dim: int = 4):
    torch.manual_seed(0)
    bags = []
    for i in range(num_bags):
        label = i % 2
        n_instances = 2 + (i % 3)
        center = -1.0 if label == 0 else 1.0
        instances = center + 0.2 * torch.randn(n_instances, in_dim)
        bags.append(make_bag(instances=instances, label=label))
    return bags


def test_metrics_perfect_classification():
    logits = torch.tensor(
        [
            [3.0, 1.0],
            [1.0, 3.0],
            [0.5, 2.0],
            [2.0, 0.1],
        ]
    )
    targets = torch.tensor([0, 1, 1, 0])

    assert accuracy(logits, targets) == 1.0
    assert f1(logits, targets) == 1.0
    assert auroc(logits, targets) == 1.0
    assert performance(logits, targets) == 1.0


def test_metrics_imperfect_classification():
    scores = torch.tensor([0.9, 0.8, 0.4, 0.1])  # binary class-1 scores
    targets = torch.tensor([1, 0, 1, 0])

    assert accuracy(scores, targets) == 0.5
    assert f1(scores, targets) == 0.5
    assert auroc(scores, targets) == 0.75
    assert abs(performance(scores, targets) - ((0.5 + 0.5 + 0.75) / 3.0)) < 1e-8


def test_trainer_train_logs_per_epoch():
    train_bags = _make_synthetic_bags(num_bags=8, in_dim=4)
    train_loader = DataLoader(train_bags, batch_size=2, shuffle=False, collate_fn=mil_collate_fn)

    model = TinyMILModel(in_dim=4, num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model=model, optimizer=optimizer, device="cpu")

    logs = trainer.train(train_loader, epochs=3)

    assert len(logs) == 3
    for idx, log in enumerate(logs, start=1):
        assert log["epoch"] == idx
        for key in [
            "train_loss",
            "train_accuracy",
            "train_f1",
            "train_auroc",
            "train_performance",
        ]:
            assert key in log


def test_trainer_validation_and_early_stopping():
    train_bags = _make_synthetic_bags(num_bags=8, in_dim=4)
    val_bags = _make_synthetic_bags(num_bags=6, in_dim=4)

    train_loader = DataLoader(train_bags, batch_size=2, shuffle=False, collate_fn=mil_collate_fn)
    val_loader = DataLoader(val_bags, batch_size=2, shuffle=False, collate_fn=mil_collate_fn)

    model = TinyMILModel(in_dim=4, num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # fixed parameters -> no val improvement
    trainer = Trainer(model=model, optimizer=optimizer, device="cpu")

    logs = trainer.train(train_loader, epochs=10, val_dataloader=val_loader, patience=1)

    assert len(logs) == 3
    assert logs[-1].get("early_stopped", False) is True
    for key in [
        "val_loss",
        "val_accuracy",
        "val_f1",
        "val_auroc",
        "val_performance",
    ]:
        assert key in logs[0]


def test_kfold_split_indices_basic_coverage():
    splits = kfold_split_indices(n_samples=10, n_splits=5, shuffle=False)
    assert len(splits) == 5

    # Validation indices should partition dataset exactly once.
    val_flat = []
    for train_idx, val_idx in splits:
        assert len(train_idx) + len(val_idx) == 10
        assert set(train_idx).isdisjoint(set(val_idx))
        val_flat.extend(val_idx)

    assert sorted(val_flat) == list(range(10))
