import math

import torch

from urban_model.structured_city import StructuredCityConfig, StructuredCityDenoiser
from urban_model.structured_city_data import SceneTensorConfig, from_local_frames, to_local_frames
from urban_model.structured_city_diffusion import corrupt_scene, structured_city_loss


def ring(points, count):
    points = torch.tensor(points, dtype=torch.float32)
    closed = torch.cat([points, points[:1]])
    lengths = (closed[1:] - closed[:-1]).norm(dim=-1)
    total = float(lengths.sum())
    result = []
    for index in range(count):
        distance = total * index / count
        for side in range(len(points)):
            if distance <= lengths[side] or side == len(points) - 1:
                fraction = distance / max(float(lengths[side]), 1e-9)
                result.append(closed[side] + (closed[side + 1] - closed[side]) * fraction)
                break
            distance -= float(lengths[side])
    return torch.stack(result)


def metres(points):
    return points / 512.0 * 2.0 - 1.0


def scene():
    config = SceneTensorConfig(
        node_slots=4,
        edge_slots=3,
        building_slots=3,
        area_slots=2,
        edge_shape_points=5,
        building_points=8,
        area_points=12,
    )
    node_position = torch.zeros(4, 3)
    node_position[0, :2] = metres(torch.tensor([100.0, 100.0]))
    node_position[1, :2] = metres(torch.tensor([300.0, 140.0]))
    edge_shape = torch.zeros(3, 5, 3)
    start = node_position[0, :2]
    end = node_position[1, :2]
    for index in range(5):
        fraction = index / 4
        bend = math.sin(fraction * math.pi) * 0.04
        edge_shape[0, index, :2] = torch.tensor([0.0, bend])
    building_shape = torch.zeros(3, 8, 2)
    building_shape[0] = metres(ring([[200, 200], [212, 200], [212, 230], [200, 230]], 8))
    building_shape[1] = metres(ring([[50, 400], [53, 400], [53, 403], [50, 403]], 8))
    area_shape = torch.zeros(2, 12, 2)
    area_shape[0] = metres(ring([[0, 0], [250, 0], [250, 100], [100, 100], [100, 300], [0, 300]], 12))
    value = {
        "node_position": node_position,
        "node_presence": torch.tensor([1, 1, 0, 0]),
        "edge_presence": torch.tensor([1, 0, 0]),
        "edge_from": torch.tensor([0, 0, 0]),
        "edge_to": torch.tensor([1, 0, 0]),
        "edge_shape": edge_shape,
        "building_presence": torch.tensor([1, 1, 0]),
        "building_shape": building_shape,
        "area_presence": torch.tensor([1, 0]),
        "area_shape": area_shape,
    }
    return config, value, (start, end)


def test_round_trip():
    config, value, _ends = scene()
    local = to_local_frames(value, config)
    back = from_local_frames(local, config)
    assert torch.allclose(back["building_shape"][:2], value["building_shape"][:2], atol=1e-5)
    assert torch.allclose(back["area_shape"][:1], value["area_shape"][:1], atol=1e-5)
    assert torch.allclose(back["edge_shape"][0], value["edge_shape"][0], atol=1e-5)


def test_local_shapes_are_unit_size():
    config, value, _ends = scene()
    local = to_local_frames(value, config)
    for name in ("building_shape", "area_shape"):
        rms = local[name][0].square().sum(-1).mean().sqrt()
        assert abs(float(rms) - 1.0) < 1e-4
    assert torch.all(local["building_frame"][2] == 0)
    radius = math.exp(float(local["building_frame"][1, 2])) * 10.0
    assert 1.5 < radius < 2.5


def test_model_and_loss_with_frames():
    torch.manual_seed(0)
    model_config = StructuredCityConfig(
        context_dimensions=18,
        relation_count=4,
        port_dimensions=21,
        node_slots=16,
        edge_slots=24,
        building_slots=20,
        area_slots=8,
        edge_shape_points=4,
        building_points=6,
        area_points=8,
        model_dimensions=32,
        attention_heads=4,
        context_layers=1,
        transport_layers=1,
        building_layers=1,
        area_layers=1,
        feedforward_dimensions=64,
        local_frames=True,
    )
    model = StructuredCityDenoiser(model_config)
    batch = 2
    target = {
        "node_position": torch.randn(batch, 16, 3),
        "node_presence": torch.ones(batch, 16, dtype=torch.long),
        "node_z_valid": torch.zeros(batch, 16, dtype=torch.bool),
        "edge_width": torch.randn(batch, 24, 1),
        "edge_shape": torch.randn(batch, 24, 4, 3),
        "edge_presence": torch.ones(batch, 24, dtype=torch.long),
        "edge_mode": torch.randint(0, 2, (batch, 24)),
        "edge_class": torch.randint(0, 7, (batch, 24)),
        "edge_vertical": torch.randint(0, 4, (batch, 24)),
        "edge_from": torch.randint(0, 16, (batch, 24)),
        "edge_to": torch.randint(0, 16, (batch, 24)),
        "edge_width_valid": torch.ones(batch, 24, dtype=torch.bool),
        "edge_width_weight": torch.ones(batch, 24),
        "edge_z_valid": torch.zeros(batch, 24, 4, dtype=torch.bool),
        "building_shape": torch.randn(batch, 20, 6, 2),
        "building_frame": torch.randn(batch, 20, 3),
        "building_height": torch.randn(batch, 20, 1),
        "building_height_valid": torch.ones(batch, 20, dtype=torch.bool),
        "building_height_weight": torch.ones(batch, 20),
        "building_base_z": torch.zeros(batch, 20, 1),
        "building_base_z_valid": torch.zeros(batch, 20, dtype=torch.bool),
        "building_presence": torch.ones(batch, 20, dtype=torch.long),
        "building_kind": torch.randint(0, 8, (batch, 20)),
        "area_shape": torch.randn(batch, 8, 8, 2),
        "area_frame": torch.randn(batch, 8, 3),
        "area_presence": torch.ones(batch, 8, dtype=torch.long),
        "area_kind": torch.randint(0, 6, (batch, 8)),
    }
    noisy = corrupt_scene(target, torch.rand(batch))
    assert not torch.equal(noisy["building_frame"], target["building_frame"])
    output = model(
        noisy,
        torch.randn(batch, 12, 18),
        torch.rand(batch, 4, 12, 12),
        torch.zeros(batch, 12, dtype=torch.bool),
        torch.randn(batch, 10, 21),
        torch.zeros(batch, 10, dtype=torch.bool),
        torch.rand(batch),
    )
    assert output["building_frame"].shape == (batch, 20, 3)
    assert output["area_frame"].shape == (batch, 8, 3)
    loss, parts = structured_city_loss(output, target)
    assert torch.isfinite(loss)
    assert "building_frame" in parts and "area_frame" in parts
