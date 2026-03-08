import pytest
import torch

from torchmil.data import make_bag, mil_collate_fn, validate_bag


def test_bag_creation_with_different_sizes():
    bag_small = make_bag(
        instances=torch.randn(2, 8),
        label=0,
    )
    bag_large = make_bag(
        instances=torch.randn(5, 8),
        label=1,
    )

    assert bag_small["instances"].shape == (2, 8)
    assert bag_large["instances"].shape == (5, 8)
    assert int(bag_small["length"].item()) == 2
    assert int(bag_large["length"].item()) == 5


def test_collation_of_variable_instance_counts():
    bag_a = make_bag(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), label=0)
    bag_b = make_bag(torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]), label=1)

    batch = mil_collate_fn([bag_a, bag_b])

    assert batch["instances"].shape == (2, 3, 2)
    assert torch.allclose(batch["instances"][0, :2], bag_a["instances"])
    assert torch.equal(batch["instances"][0, 2], torch.zeros(2))
    assert torch.allclose(batch["instances"][1], bag_b["instances"])
    assert torch.equal(batch["length"], torch.tensor([2, 3]))


def test_attention_mask_generation():
    bag_a = make_bag(torch.randn(1, 4), label=0)
    bag_b = make_bag(torch.randn(3, 4), label=1)

    batch = mil_collate_fn([bag_a, bag_b])
    expected = torch.tensor(
        [
            [True, False, False],
            [True, True, True],
        ]
    )
    assert torch.equal(batch["attention_mask"], expected)


def test_adjacency_matrix_handling():
    inst_a = torch.randn(2, 3)
    adj_a = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    inst_b = torch.randn(3, 3)
    adj_b = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    bag_a = make_bag(instances=inst_a, label=0, adjacency=adj_a)
    bag_b = make_bag(instances=inst_b, label=1, adjacency=adj_b)

    batch = mil_collate_fn([bag_a, bag_b])

    assert batch["adjacency"].shape == (2, 3, 3)
    assert torch.allclose(batch["adjacency"][0, :2, :2], adj_a)
    assert torch.equal(batch["adjacency"][0, 2], torch.zeros(3))
    assert torch.equal(batch["adjacency"][0, :, 2], torch.zeros(3))
    assert torch.allclose(batch["adjacency"][1], adj_b)


# --- Reviewer-requested tests ---


def test_single_bag_collation():
    """Single bag should collate without padding."""
    bag = make_bag(torch.randn(4, 6), label=1)
    batch = mil_collate_fn([bag])

    assert batch["instances"].shape == (1, 4, 6)
    assert torch.equal(batch["attention_mask"], torch.ones(1, 4, dtype=torch.bool))
    assert batch["label"].shape == (1,)
    assert batch["length"].item() == 4


def test_instance_labels_collation():
    """Instance labels should be padded with -1."""
    bag_a = make_bag(
        torch.randn(2, 4), label=0,
        instance_labels=torch.tensor([0, 1]),
    )
    bag_b = make_bag(
        torch.randn(3, 4), label=1,
        instance_labels=torch.tensor([1, 0, 1]),
    )
    batch = mil_collate_fn([bag_a, bag_b])

    assert batch["instance_labels"].shape == (2, 3)
    assert torch.equal(batch["instance_labels"][0], torch.tensor([0, 1, -1]))
    assert torch.equal(batch["instance_labels"][1], torch.tensor([1, 0, 1]))


def test_feature_dim_mismatch_raises():
    """Bags with different feature dimensions should raise ValueError."""
    bag_a = make_bag(torch.randn(2, 4), label=0)
    bag_b = make_bag(torch.randn(3, 8), label=1)
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        mil_collate_fn([bag_a, bag_b])


def test_mixed_adjacency_raises():
    """Mixing bags with/without adjacency should raise ValueError."""
    bag_a = make_bag(torch.randn(2, 4), label=0,
                     adjacency=torch.eye(2))
    bag_b = make_bag(torch.randn(3, 4), label=1)
    with pytest.raises(ValueError, match="adjacency"):
        mil_collate_fn([bag_a, bag_b])


def test_mixed_instance_labels_raises():
    """Mixing bags with/without instance_labels should raise ValueError."""
    bag_a = make_bag(torch.randn(2, 4), label=0,
                     instance_labels=torch.tensor([0, 1]))
    bag_b = make_bag(torch.randn(3, 4), label=1)
    with pytest.raises(ValueError, match="instance_labels"):
        mil_collate_fn([bag_a, bag_b])


def test_empty_bag_list_raises():
    """Empty bag list should raise ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        mil_collate_fn([])


def test_invalid_instances_shape_raises():
    """Non-2D instances should raise ValueError."""
    with pytest.raises(ValueError, match="shape"):
        make_bag(torch.randn(5), label=0)


def test_adjacency_shape_mismatch_raises():
    """Adjacency with wrong shape should raise ValueError."""
    with pytest.raises(ValueError, match="shape"):
        make_bag(torch.randn(3, 4), label=0,
                 adjacency=torch.eye(2))


def test_validate_bag_missing_keys():
    """validate_bag should catch missing required keys."""
    from tensordict import TensorDict
    bad_bag = TensorDict({"instances": torch.randn(2, 4)}, batch_size=[])
    with pytest.raises(KeyError, match="missing required keys"):
        validate_bag(bad_bag)
