from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("skimage")
pytest.importorskip("scipy")

from urban_model.data import layers_to_model_space, model_space_to_layers
from urban_model.vectorize import generated_layers_to_city_state


def _synthetic_layers():
    layers = torch.zeros((12, 32, 32), dtype=torch.float32)
    layers[7, 2:8, 2:8] = 1
    layers[8, 10:16, 10:16] = 1
    layers[9, 10:16, 10:16] = 0.5
    layers[1, 20:23, 2:25] = 1
    layers[11, 2:25, 27:29] = 1
    layers[0, 25:32, 25:32] = 1

    road_vertical = torch.zeros((4, 32, 32), dtype=torch.float32)
    rail_vertical = torch.zeros((4, 32, 32), dtype=torch.float32)
    road_vertical[1, 5:8, 2:20] = 1
    rail_vertical[2, 24:27, 4:28] = 1
    return layers, road_vertical, rail_vertical


def test_multilayer_tensor_keeps_vertical_transport() -> None:
    layers, road_vertical, rail_vertical = _synthetic_layers()
    values = layers_to_model_space(layers, road_vertical, rail_vertical)
    decoded = model_space_to_layers(values)

    assert values.shape == (13, 32, 32)
    assert decoded["surface"][21, 10].item() == 3
    assert decoded["road_underground"].sum().item() > 0
    assert decoded["rail_elevated"].sum().item() > 0
    assert decoded["building_height"].max().item() == pytest.approx(0.5)


def test_multilayer_tensor_compiles_to_layered_city_json() -> None:
    layers, road_vertical, rail_vertical = _synthetic_layers()
    values = layers_to_model_space(layers, road_vertical, rail_vertical)
    city = generated_layers_to_city_state(
        values,
        bounds_m=[0.0, 0.0, 1024.0, 1024.0],
        max_height_m=180.0,
        minimum_component_pixels=2,
        seed=7,
    )

    edges = city["transport_graph"]["edges"]
    assert city["format"] == "urban-city-state-tile"
    assert city["building_solids"]
    assert any(edge["vertical_mode"] == "surface" for edge in edges)
    assert any(edge["vertical_mode"] == "underground" for edge in edges)
    assert any(edge["vertical_mode"] == "elevated" for edge in edges)
    assert all(point[2] == -12.0 for edge in edges if edge["vertical_mode"] == "underground" for point in edge["geometry_local_m"])
    assert all(point[2] == 8.0 for edge in edges if edge["vertical_mode"] == "elevated" for point in edge["geometry_local_m"])
