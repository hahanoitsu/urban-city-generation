import torch

from urban_model.city_plan_data import (
    CityPlanConfig,
    build_city_plan,
)


def test_city_plan_tracks_spatial_transport_structure():
    sample = {
        "node_count": torch.tensor(4),
        "node_xy": torch.tensor(
            [
                [-0.75, -0.75],
                [-0.25, -0.75],
                [0.25, 0.25],
                [0.75, 0.75],
            ],
            dtype=torch.float32,
        ),
        "node_mode": torch.tensor([0, 0, 1, 1]),
        "node_boundary": torch.tensor([1.0, 0.0, 0.0, 1.0]),
        "edge_count": torch.tensor(3),
        "edge_from": torch.tensor([0, 1, 2]),
        "edge_to": torch.tensor([1, 2, 3]),
        "edge_class": torch.tensor([0, 2, 3]),
    }
    plan, orientation, orientation_mask, global_values = build_city_plan(
        sample,
        CityPlanConfig(
            grid_size=4,
            corridor_samples_per_cell=4,
        ),
    )

    assert plan.shape == (16, 8)
    assert orientation.shape == (16, 4, 2)
    assert orientation_mask.shape == (16, 4)
    assert float(plan[:, 0].sum()) == 4.0
    assert float(plan[:, 1].sum()) == 2.0
    assert float(plan[:, 2].sum()) == 2.0
    assert float(plan[:, 3].sum()) >= 1.0
    assert float(plan[:, 5].sum()) >= 1.0
    assert float(plan[:, 6].sum()) >= 1.0
    assert float(plan[:, 7].sum()) == 2.0
    assert bool(orientation_mask[:, 0].any())
    assert bool(orientation_mask[:, 2].any())
    assert bool(orientation_mask[:, 3].any())
    assert float(global_values[0]) == 4.0
    assert float(global_values[1]) == 3.0
    assert float(global_values[2]) == 1.0
    assert torch.isclose(global_values[3], torch.tensor(1.0 / 3.0))
