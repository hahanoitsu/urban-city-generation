import torch

from scripts.run_1024_loss_probe import edge_loss, transport_margin_loss


def test_auxiliary_losses_are_zero_for_exact_target():
    classes = torch.tensor(
        [[[0, 0, 3], [0, 2, 3], [6, 6, 3]]],
        dtype=torch.long,
    )
    target = torch.nn.functional.one_hot(classes, num_classes=8)
    target = target.permute(0, 3, 1, 2).float().mul(2.0).sub(1.0)
    supervision = torch.ones_like(target)
    weights = torch.ones(8)

    assert edge_loss(target, target, supervision).item() == 0.0
    assert transport_margin_loss(target, target, supervision, weights).item() == 0.0


def test_transport_margin_penalises_ambiguous_transport_scores():
    classes = torch.tensor([[[3, 6]]], dtype=torch.long)
    target = torch.nn.functional.one_hot(classes, num_classes=8)
    target = target.permute(0, 3, 1, 2).float().mul(2.0).sub(1.0)
    prediction = torch.zeros_like(target)
    supervision = torch.ones_like(target)
    weights = torch.ones(8)

    loss = transport_margin_loss(prediction, target, supervision, weights)
    assert loss.item() > 0.0
