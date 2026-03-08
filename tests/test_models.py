import pytest
import torch

from torchmil.data import make_bag, mil_collate_fn
from torchmil.models import ABMIL, CLAM, DSMIL, DTFDMIL, TransMIL


MODEL_CONFIGS = [
    (ABMIL, {"hidden_dim": 16}),
    (CLAM, {"hidden_dim": 16, "top_k": 2}),
    (TransMIL, {"num_heads": 2, "num_layers": 2, "dropout": 0.0}),
    (DSMIL, {"hidden_dim": 16}),
    (DTFDMIL, {"hidden_dim": 16, "top_k": 2}),
]


def _dummy_batch(batch_size: int = 3, feat_dim: int = 8):
    bags = []
    for i, n_inst in enumerate([2, 4, 3][:batch_size]):
        bag = make_bag(
            instances=torch.randn(n_inst, feat_dim),
            label=i % 2,
            adjacency=torch.eye(n_inst),
        )
        bags.append(bag)
    return mil_collate_fn(bags)


@pytest.mark.parametrize("model_cls,extra_kwargs", MODEL_CONFIGS)
def test_model_forward_shape(model_cls, extra_kwargs):
    torch.manual_seed(0)
    batch = _dummy_batch()
    model = model_cls(in_shape=8, num_classes=2, **extra_kwargs)

    logits = model(batch)

    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("model_cls,extra_kwargs", MODEL_CONFIGS)
def test_model_loss_computation(model_cls, extra_kwargs):
    torch.manual_seed(0)
    batch = _dummy_batch()
    model = model_cls(in_shape=8, num_classes=2, **extra_kwargs)

    logits = model(batch)
    loss = model.criterion(logits, batch["label"].long())

    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize("model_cls,extra_kwargs", MODEL_CONFIGS)
def test_model_state_dict_round_trip(model_cls, extra_kwargs):
    torch.manual_seed(0)
    batch = _dummy_batch()

    model_a = model_cls(in_shape=8, num_classes=2, **extra_kwargs)
    model_b = model_cls(in_shape=8, num_classes=2, **extra_kwargs)

    model_a.eval()
    model_b.eval()

    with torch.no_grad():
        out_a = model_a(batch)

    model_b.load_state_dict(model_a.state_dict())

    with torch.no_grad():
        out_b = model_b(batch)

    assert torch.allclose(out_a, out_b, atol=1e-6)
