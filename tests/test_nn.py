import torch

from torchmil.nn import (
    AttentionPooling,
    BagClassifier,
    GatedAttention,
    GraphConv,
    MILTransformerEncoder,
    MaxPooling,
    MeanPooling,
)


def test_gated_attention_shape_and_masking_excludes_padded_instances():
    torch.manual_seed(0)
    module = GatedAttention(in_dim=4, hidden_dim=8)

    x = torch.randn(2, 5, 4)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    )

    weights_a, bag_a = module(x, mask)

    x_perturbed = x.clone()
    x_perturbed[0, 3:] = 1000.0
    x_perturbed[1, 4] = -1000.0

    weights_b, bag_b = module(x_perturbed, mask)

    assert weights_a.shape == (2, 5)
    assert bag_a.shape == (2, 4)
    assert torch.allclose(weights_a[~mask], torch.zeros_like(weights_a[~mask]))
    assert torch.allclose(weights_b[~mask], torch.zeros_like(weights_b[~mask]))
    assert torch.allclose(bag_a, bag_b, atol=1e-6)


def test_transformer_encoder_shape_and_masking():
    torch.manual_seed(0)
    module = MILTransformerEncoder(
        in_dim=8,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
    )
    module.eval()

    x = torch.randn(2, 6, 8)
    mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
        ]
    )

    out_a = module(x, mask)

    x_perturbed = x.clone()
    x_perturbed[0, 3:] = 500.0
    x_perturbed[1, 4:] = -500.0
    out_b = module(x_perturbed, mask)

    assert out_a.shape == (2, 6, 8)
    assert torch.allclose(out_a[~mask], torch.zeros_like(out_a[~mask]))
    assert torch.allclose(out_b[~mask], torch.zeros_like(out_b[~mask]))
    assert torch.allclose(out_a[mask], out_b[mask], atol=1e-5)


def test_graph_conv_shape_and_masking():
    torch.manual_seed(0)
    module = GraphConv(in_dim=3, out_dim=5)

    x = torch.randn(2, 4, 3)
    adj = torch.ones(2, 4, 4)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )

    out_a = module(x, adj, mask)

    x_perturbed = x.clone()
    x_perturbed[0, 2:] = 999.0
    x_perturbed[1, 3] = -999.0
    out_b = module(x_perturbed, adj, mask)

    assert out_a.shape == (2, 4, 5)
    assert torch.allclose(out_a[~mask], torch.zeros_like(out_a[~mask]))
    assert torch.allclose(out_b[~mask], torch.zeros_like(out_b[~mask]))
    assert torch.allclose(out_a[mask], out_b[mask], atol=1e-6)


def test_graph_conv_respects_adjacency_message_passing():
    module = GraphConv(in_dim=1, out_dim=1, add_self_loops=False, bias=False)

    with torch.no_grad():
        module.lin_self.weight.zero_()
        module.lin_neigh.weight.fill_(1.0)

    x = torch.tensor([[[1.0], [3.0], [5.0]]])
    adj = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        ]
    )
    mask = torch.tensor([[True, True, True]])

    out = module(x, adj, mask)
    # With symmetric normalization D^{-1/2}AD^{-1/2},
    # values become [3/sqrt(2), (1+5)/sqrt(2), 3/sqrt(2)].
    s2 = 2.0 ** 0.5
    expected = torch.tensor([[[3.0 / s2], [6.0 / s2], [3.0 / s2]]])

    assert torch.allclose(out, expected, atol=1e-6)


def test_mean_and_max_pooling_are_mask_aware():
    x = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]],
            [[-1.0, 5.0], [2.0, 1.0], [10.0, -10.0]],
        ]
    )
    mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
        ]
    )

    mean_pool = MeanPooling()
    max_pool = MaxPooling()

    mean_out = mean_pool(x, mask)
    max_out = max_pool(x, mask)

    expected_mean = torch.tensor([[2.0, 3.0], [11.0 / 3.0, -4.0 / 3.0]])
    expected_max = torch.tensor([[3.0, 4.0], [10.0, 5.0]])

    assert torch.allclose(mean_out, expected_mean, atol=1e-6)
    assert torch.allclose(max_out, expected_max, atol=1e-6)


def test_attention_pooling_shapes_and_masking():
    torch.manual_seed(0)
    module = AttentionPooling(in_dim=4, hidden_dim=6)

    x = torch.randn(1, 4, 4)
    mask = torch.tensor([[True, True, False, False]])

    bag_a, weights_a = module(x, mask, return_attention=True)

    x_perturbed = x.clone()
    x_perturbed[:, 2:] = 100.0
    bag_b, weights_b = module(x_perturbed, mask, return_attention=True)

    assert bag_a.shape == (1, 4)
    assert weights_a.shape == (1, 4)
    assert torch.allclose(weights_a[~mask], torch.zeros_like(weights_a[~mask]))
    assert torch.allclose(weights_b[~mask], torch.zeros_like(weights_b[~mask]))
    assert torch.allclose(bag_a, bag_b, atol=1e-6)


def test_transformer_all_fully_masked_rows_do_not_nan():
    module = MILTransformerEncoder(in_dim=8, num_heads=2, num_layers=1, dropout=0.0)
    module.eval()

    x = torch.randn(2, 3, 8)
    # First row fully masked, second row partially valid
    mask = torch.tensor([[False, False, False], [True, False, False]])

    out = module(x, mask)

    assert out.shape == (2, 3, 8)
    assert not torch.isnan(out).any()
    assert torch.allclose(out[0], torch.zeros_like(out[0]))


def test_bag_classifier_output_shape():
    clf = BagClassifier(in_dim=16, num_classes=2, hidden_dims=[8], dropout=0.0)
    x = torch.randn(4, 16)
    y = clf(x)
    assert y.shape == (4, 2)
