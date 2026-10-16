import numpy as np
import torch
from shapely.geometry import Polygon

from urban_model.grid_city import GridCityConfig, GridCityDenoiser
from urban_model.grid_city_data import (
    D4,
    GridConfig,
    candidate_table,
    decode_grid,
    encode_grid,
    json_from_plain,
    merge_chains,
    transform_inputs,
    transform_plain,
)
from urban_model.grid_city_diffusion import corrupt, generate, grid_city_loss


def square(x, y, size):
    return np.array([[x, y], [x + size, y], [x + size, y + size], [x, y + size]], dtype=float)


def plain():
    road = np.stack([np.linspace(20, 320, 16), np.full(16, 100.0)], axis=1)
    road[:, 1] += np.sin(np.linspace(0, np.pi, 16)) * 15
    return {
        "nodes": [road[0], road[-1], np.array([320.0, 260.0])],
        "edges": [
            {"from": 0, "to": 1, "points": road, "mode": 0, "class": 1, "vertical": 0, "width": 8.0},
            {"from": 1, "to": 2, "points": np.stack([np.full(16, 320.0), np.linspace(100, 260, 16)], axis=1), "mode": 1, "class": 3, "vertical": 2, "width": None},
        ],
        "buildings": [
            {"points": square(40, 140, 20), "height": 30.0, "kind": 2},
            {"points": square(200, 300, 8), "height": None, "kind": 0},
        ],
        "areas": [{"points": square(350, 350, 120), "kind": 0}],
    }


def test_round_trip_keeps_geometry():
    config = GridConfig()
    arrays, dropped = encode_grid(plain(), config)
    assert not dropped
    assert int(arrays["node_presence"].sum()) > 3
    back = json_from_plain(merge_chains(decode_grid(arrays, config)))
    assert len(back["edges"]) == 2
    modes = sorted(edge["mode"] for edge in back["edges"])
    assert modes == ["rail", "road"]
    road = [edge for edge in back["edges"] if edge["mode"] == "road"][0]
    points = np.array([point[:2] for point in road["spline_xyz_m"]])
    assert min(np.linalg.norm(points[0] - [20, 100]), np.linalg.norm(points[-1] - [20, 100])) < 1e-3
    assert np.abs(points[:, 1] - 100).max() > 10
    assert road["width_m"] == 8.0
    for original, decoded in zip(sorted(plain()["buildings"], key=lambda item: item["points"][0][1]), sorted(back["buildings"], key=lambda item: item["footprint_xy_m"][0][1])):
        first = Polygon(original["points"])
        second = Polygon(decoded["footprint_xy_m"])
        assert first.intersection(second).area / first.union(second).area > 0.98
    heights = sorted(item["height_m"] for item in back["buildings"])
    assert abs(heights[-1] - 30.0) < 1e-4


def cross():
    centre = np.array([272.0, 272.0])
    ends = [centre + [0.0, 25.0], centre + [25.0, 0.0], centre + [0.0, -25.0], centre + [-25.0, 0.0]]
    nodes = [centre, *ends]
    edges = [
        {"from": 0, "to": index, "points": np.stack([centre, nodes[index]]), "mode": 0, "class": 2, "vertical": 0, "width": None}
        for index in range(1, 5)
    ]
    return {"nodes": nodes, "edges": edges, "buildings": [], "areas": []}


def test_edge_slots_do_not_depend_on_edge_order():
    config = GridConfig()
    scene = cross()
    first, _ = encode_grid(scene, config)
    second, _ = encode_grid({**scene, "edges": scene["edges"][::-1]}, config)
    for name in ("edge_target", "edge_mode", "edge_class", "edge_vertical", "edge_curve"):
        assert np.array_equal(first[name], second[name])
    centre = (8 * 16 + 8) * config.node_slots
    # right neighbour (0, 1) comes before the one above (1, 0)
    assert first["edge_target"][centre].tolist() == [5 * config.node_slots + 1, 7 * config.node_slots + 1, 0, 0]


def test_candidate_table_edges():
    config = GridConfig(node_grid=3, node_slots=2)
    table = candidate_table(config)
    assert table.shape == (18, 18)
    assert (table[0] < 0).sum() == 10
    centre = (1 * 3 + 1) * 2
    assert (table[centre] >= 0).all()


def test_rotation_moves_port_side():
    ports = torch.zeros(1, 23)
    ports[0, 0] = -1.0
    ports[0, 2] = 1.0
    ports[0, -4] = 1.0
    context = torch.zeros(2, 18)
    context[1, 15] = 1.0
    moved, context = transform_inputs(ports, context, D4[1], 15)
    assert torch.allclose(moved[0, :2], torch.tensor([0.0, -1.0]))
    assert torch.allclose(moved[0, 2:4], torch.tensor([0.0, 1.0]))
    assert moved[0, -4:].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert torch.allclose(context[1, 15:17], torch.tensor([0.0, 1.0]))


def test_all_rotations_encode():
    config = GridConfig()
    for matrix in D4:
        encode_grid(transform_plain(plain(), matrix, 512.0), config)


def small(position_bands=0):
    return GridCityConfig(
        position_bands=position_bands,
        context_dimensions=18,
        relation_count=4,
        port_dimensions=23,
        node_grid=4,
        node_slots=3,
        edge_slots=2,
        building_grid=4,
        building_slots=2,
        area_grid=2,
        area_slots=2,
        building_points=6,
        area_points=8,
        model_dimensions=32,
        attention_heads=4,
        context_layers=1,
        layers=1,
        feedforward_dimensions=64,
        key_dimensions=8,
        dropout=0.0,
    )


def inputs(batch):
    return {
        "context": torch.randn(batch, 5, 18),
        "relations": torch.rand(batch, 4, 5, 5),
        "context_padding": torch.zeros(batch, 5, dtype=torch.bool),
        "ports": torch.randn(batch, 6, 23),
        "port_padding": torch.zeros(batch, 6, dtype=torch.bool),
    }


def empty_scene():
    return {
        "node_pos": torch.zeros(2, 48, 2),
        "node_presence": torch.zeros(2, 48, dtype=torch.long),
        "edge_target": torch.zeros(2, 48, 2, dtype=torch.long),
        "edge_mode": torch.zeros(2, 48, 2, dtype=torch.long),
        "edge_class": torch.zeros(2, 48, 2, dtype=torch.long),
        "edge_vertical": torch.zeros(2, 48, 2, dtype=torch.long),
        "edge_width": torch.zeros(2, 48, 2, 1),
        "edge_width_valid": torch.zeros(2, 48, 2, dtype=torch.bool),
        "edge_width_weight": torch.zeros(2, 48, 2),
        "edge_curve": torch.zeros(2, 48, 2, 2, 2),
        "building_frame": torch.zeros(2, 32, 3),
        "building_shape": torch.zeros(2, 32, 6, 2),
        "building_height": torch.zeros(2, 32, 1),
        "building_height_valid": torch.zeros(2, 32, dtype=torch.bool),
        "building_height_weight": torch.zeros(2, 32),
        "building_presence": torch.zeros(2, 32, dtype=torch.long),
        "building_kind": torch.zeros(2, 32, dtype=torch.long),
        "area_frame": torch.zeros(2, 8, 3),
        "area_shape": torch.zeros(2, 8, 8, 2),
        "area_presence": torch.zeros(2, 8, dtype=torch.long),
        "area_kind": torch.zeros(2, 8, dtype=torch.long),
    }


def test_model_loss_and_generate():
    for bands in (0, 3):
        check_model(small(bands))


def check_model(config):
    torch.manual_seed(0)
    model = GridCityDenoiser(config)
    grid = config.grid()
    scene = empty_scene()
    scene["node_presence"][:, [0, 1]] = 1
    # node 0 points at node 1, slot 1 of the same cell
    scene["edge_target"][:, 0, 0] = 4 * 3 + 1 + 1
    scene["building_presence"][:, 3] = 1
    scene["area_presence"][:, 1] = 1
    time = torch.rand(2)
    noisy = corrupt(scene, time, config)
    batch = inputs(2)
    output = model(noisy, batch["context"], batch["relations"], batch["context_padding"], batch["ports"], batch["port_padding"], time)
    assert output["edge_target"].shape == (2, 48, 2, config.candidates + 1)
    loss, parts = grid_city_loss(output, scene)
    loss.backward()
    assert torch.isfinite(loss)
    assert "edge_target" in parts
    unused = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert not unused

    model.eval()
    generated = generate(model, inputs(1), steps=3, generator=torch.Generator().manual_seed(0))
    arrays = {name: value[0].numpy() for name, value in generated.items()}
    assert not (generated["node_presence"] == 2).any()
    payload = json_from_plain(merge_chains(decode_grid(arrays, grid)))
    assert set(payload) == {"nodes", "edges", "buildings", "areas"}
