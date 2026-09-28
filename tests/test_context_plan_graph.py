import torch

from test_city_planner import make_batch as make_context_batch
from test_plan_cell_graph import make_batch as make_graph_batch
from urban_model.city_plan_loss import city_plan_loss
from urban_model.city_planner import CityPlannerConfig
from urban_model.context_plan_graph import ContextPlanGraph, context_inputs
from urban_model.plan_cell_graph import PlanCellGraphConfig
from urban_model.plan_cell_graph_loss import plan_cell_graph_loss
from urban_model.spatial_world_data import buffered_splits


def model_and_batch():
    torch.manual_seed(3)
    batch = {**make_context_batch(), **make_graph_batch()}
    batch["plan_orientation_mask"] = torch.ones(2, 16, 4, dtype=torch.bool)
    batch["plan_global_raw"] = torch.zeros(2, 8)
    shared = dict(model_dimensions=32, heads=4, feedforward_dimensions=64, dropout=0.0)
    planner = CityPlannerConfig(
        context_dimensions=7,
        style_dimensions=4,
        plan_dimensions=8,
        orientation_dimensions=4,
        global_dimensions=8,
        grid_size=4,
        context_line_points=3,
        planner_layers=1,
        **shared,
    )
    graph = PlanCellGraphConfig(
        plan_dimensions=8,
        orientation_dimensions=4,
        global_dimensions=8,
        plan_grid_size=4,
        max_nodes=64,
        max_edges=100,
        edge_shape_points=3,
        edge_dimensions=16,
        plan_layers=1,
        node_layers=1,
        **shared,
    )
    norm = {"global_mean": [0.0] * 8, "global_std": [1.0] * 8, "presence_pos_weight": [1.0] * 8}
    return ContextPlanGraph(planner.to_dict(), graph.to_dict(), norm), batch


def test_context_and_graph_heads_receive_gradients():
    model, batch = model_and_batch()
    plan, graph = model(context_inputs(batch), batch)
    a, _ = city_plan_loss(
        plan,
        batch,
        presence_pos_weight=model.presence_pos_weight,
        global_mean=model.global_mean,
        global_std=model.global_std,
    )
    b, _, _ = plan_cell_graph_loss(graph, batch, target_size_m=512, grid_size=4)
    (a + b).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert torch.isfinite(a + b)


def test_generation_uses_context_only_and_repeats_a_seed():
    model, batch = model_and_batch()
    model.eval()
    context = context_inputs(batch)
    context["plan_counts"] = torch.full((2, 16, 8), float("nan"))
    context["node_count"] = torch.tensor([999, 999])
    plan_a, output_a = model.generate(
        context, stochastic=True, generator=torch.Generator().manual_seed(7)
    )
    context.pop("plan_counts")
    context.pop("node_count")
    plan_b, output_b = model.generate(
        context, stochastic=True, generator=torch.Generator().manual_seed(7)
    )
    plan_c, _ = model.generate(
        context, stochastic=True, generator=torch.Generator().manual_seed(19)
    )
    torch.testing.assert_close(plan_a["plan_counts"], plan_b["plan_counts"])
    torch.testing.assert_close(output_a["node_xy"], output_b["node_xy"])
    assert not torch.equal(plan_a["plan_counts"], plan_c["plan_counts"])
    assert torch.isfinite(output_a["node_xy"]).all()
    assert "node_xy" not in context
    assert torch.count_nonzero(context["controls"]) == 0


def test_context_reaches_the_graph_decoder():
    model, batch = model_and_batch()
    model.eval()
    with torch.inference_mode():
        _, graph_a = model(context_inputs(batch), batch)
        _, graph_b = model(context_inputs(batch, use_context=False), batch)
    assert not torch.allclose(graph_a["node_xy"], graph_b["node_xy"])


def test_absent_mode_count_predictions_do_not_change_the_node_mix():
    model, batch = model_and_batch()
    plan = {
        "plan_presence": torch.full_like(batch["plan_presence"], -30.0),
        "plan_log_count": torch.zeros_like(batch["plan_counts"]),
        "plan_orientation": batch["plan_orientation"],
        "plan_global": torch.zeros_like(batch["plan_global"]),
    }
    plan["plan_presence"][:, 0, :2] = 30.0
    plan["plan_log_count"][:, 0, :2] = torch.log1p(torch.tensor(4.0))
    plan["plan_log_count"][:, :, 2] = torch.log1p(torch.tensor(1000.0))
    converted = model.predicted_plan(plan)
    assert torch.equal(converted["node_count"], torch.tensor([4, 4]))
    assert torch.equal(
        converted["plan_counts"][:, 0, :3], torch.tensor([[4.0, 4.0, 0.0], [4.0, 4.0, 0.0]])
    )
    assert torch.count_nonzero(converted["plan_counts"][:, 1:]) == 0


def test_deterministic_conversion_uses_the_conditional_count():
    model, batch = model_and_batch()
    plan = {
        "plan_presence": torch.full_like(batch["plan_presence"], -30.0),
        "plan_log_count": torch.zeros_like(batch["plan_counts"]),
        "plan_orientation": batch["plan_orientation"],
        "plan_global": torch.zeros_like(batch["plan_global"]),
    }
    plan["plan_presence"][:, 0, 0] = torch.logit(torch.tensor(0.6))
    plan["plan_log_count"][:, 0, 0] = torch.log1p(torch.tensor(5.0))
    converted = model.predicted_plan(plan)
    assert torch.equal(converted["node_count"], torch.tensor([5, 5]))


def test_buffered_split_keeps_context_windows_apart():
    payloads = [
        (
            {"id": str(i)},
            {"city_id": "a", "bounds": {"target_projected_m": [i * 512, 0, (i + 1) * 512, 512]}},
        )
        for i in range(100)
    ]
    splits = buffered_splits(payloads, context_size_m=2048)
    assert set(splits.values()) == {"train", "validation", "test", "buffer"}
    for name, other in (("train", "validation"), ("validation", "test")):
        left = [int(key) for key, value in splits.items() if value == name]
        right = [int(key) for key, value in splits.items() if value == other]
        assert (min(right) - max(left)) * 512 >= 2048
