import torch

from test_plan_cell_graph import make_batch
from urban_model.plan_cell_graph_loss import canonical_cell_order, plan_cell_graph_loss


def geometry_output(batch, target_size_m, error_m=4.0):
    xy = torch.zeros(2, 5, 2)
    for i, count in enumerate(batch["node_count"]):
        order = canonical_cell_order(batch["node_xy"][i, :count], 4)
        xy[i, :count] = batch["node_xy"][i, :count][order]
    xy[..., 0] += error_m * 2 / target_size_m
    curve = torch.zeros(2, 5, 5, 3, 2)
    curve[..., 0] = error_m * 2 / target_size_m
    return {
        "node_xy": xy.requires_grad_(),
        "node_local": torch.zeros(2, 5, 2),
        "node_mode": torch.zeros(2, 5, 2),
        "node_vertical": torch.zeros(2, 5, 4),
        "node_boundary": torch.zeros(2, 5),
        "node_degree": torch.zeros(2, 5, 5),
        "edge_exists": torch.zeros(2, 5, 5),
        "edge_class": torch.zeros(2, 5, 5, 8),
        "edge_vertical": torch.zeros(2, 5, 5, 4),
        "edge_width": torch.zeros(2, 5, 5, 1),
        "edge_curve": curve.requires_grad_(),
    }


def test_same_error_in_metres_has_the_same_loss_across_window_sizes():
    batch = make_batch()
    results = []
    for size in (512.0, 1024.0):
        _, metrics, _ = plan_cell_graph_loss(
            geometry_output(batch, size),
            batch,
            target_size_m=size,
            grid_size=4,
            geometry_scale_m=10.0,
        )
        results.append(metrics)
    for name in ("node_local", "edge_curve", "node_position_mae_m", "curve_mae_m"):
        torch.testing.assert_close(torch.tensor(results[0][name]), torch.tensor(results[1][name]))
    assert abs(results[0]["node_local"] - results[0]["edge_curve"]) < 1e-6
    assert abs(results[0]["node_position_mae_m"] - 4.0) < 1e-5


def test_metric_geometry_loss_gives_curves_a_meaningful_gradient():
    batch = make_batch()
    legacy = geometry_output(batch, 512.0)
    metric = geometry_output(batch, 512.0)
    old_loss, _, _ = plan_cell_graph_loss(legacy, batch, target_size_m=512.0, grid_size=4)
    new_loss, _, _ = plan_cell_graph_loss(
        metric, batch, target_size_m=512.0, grid_size=4, geometry_scale_m=10.0
    )
    old_loss.backward()
    new_loss.backward()
    assert metric["edge_curve"].grad.abs().sum() > 100 * legacy["edge_curve"].grad.abs().sum()
    assert torch.isfinite(metric["node_xy"].grad).all()


def test_metric_loss_still_rewards_real_bends():
    batch = make_batch()
    batch["edge_shape"][:, :, 1, 1] = 0.05
    output = geometry_output(batch, 512.0, error_m=0.0)
    optimizer = torch.optim.SGD([output["edge_curve"]], lr=0.005)
    history = []
    for _ in range(50):
        optimizer.zero_grad()
        loss, metrics, _ = plan_cell_graph_loss(
            output, batch, target_size_m=512.0, grid_size=4, geometry_scale_m=10.0
        )
        history.append(metrics["curve_mae_m"])
        loss.backward()
        optimizer.step()
    assert history[-1] < history[0]
    assert output["edge_curve"].detach().abs().max() > 0.01
