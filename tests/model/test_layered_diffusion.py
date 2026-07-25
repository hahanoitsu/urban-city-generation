from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("skimage")
pytest.importorskip("scipy")

from urban_model.config import LayeredDiffusionConfig
from urban_model.data import layers_to_model_space, model_space_to_layers
from urban_model.vectorize import generated_layers_to_city_state


def _config() -> LayeredDiffusionConfig:
    return LayeredDiffusionConfig(
        train_manifest=Path("train.jsonl"),
        validation_manifest=Path("validation.jsonl"),
        output_dir=Path("runs/test"),
        resolution=(32, 32),
        crop_size_pixels=32,
        crop_stride_pixels=32,
        block_out_channels=(32, 64),
        attention_levels=(False, True),
        norm_num_groups=8,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        precision="fp32",
        device="cpu",
    )


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
    road_vertical[0, 20:23, 2:25] = 1
    road_vertical[1, 5:8, 2:20] = 1
    rail_vertical[0, 2:25, 27:29] = 1
    rail_vertical[2, 24:27, 4:28] = 1

    road_profiles = torch.zeros((3, 32, 32), dtype=torch.float32)
    rail_profiles = torch.zeros((3, 32, 32), dtype=torch.float32)
    road_profiles[1, 5:8, 2:20] = -10.0
    road_profiles[1, 5:8, 2:5] = torch.linspace(0.0, -10.0, 3)[None, :]
    rail_profiles[2, 24:27, 4:28] = 8.0
    rail_profiles[2, 24:27, 4:8] = torch.linspace(0.0, 8.0, 4)[None, :]
    return layers, road_vertical, rail_vertical, road_profiles, rail_profiles


def test_multilayer_tensor_keeps_profiles() -> None:
    config = _config()
    layers, road_vertical, rail_vertical, road_profiles, rail_profiles = _synthetic_layers()
    values = layers_to_model_space(
        layers,
        road_vertical,
        rail_vertical,
        road_profiles,
        rail_profiles,
        config,
    )
    decoded = model_space_to_layers(values)

    assert values.shape == (19, 32, 32)
    assert decoded["surface"][21, 10].item() == 3
    assert decoded["road_underground"].sum().item() > 0
    assert decoded["rail_elevated"].sum().item() > 0
    assert decoded["building_height"].max().item() == pytest.approx(0.5)
    assert decoded["road_underground_depth_m"].max().item() == pytest.approx(10.0)
    assert decoded["rail_elevated_height_m"].max().item() == pytest.approx(8.0)


def test_multilayer_tensor_compiles_variable_z_city_json() -> None:
    config = _config()
    layers, road_vertical, rail_vertical, road_profiles, rail_profiles = _synthetic_layers()
    values = layers_to_model_space(
        layers,
        road_vertical,
        rail_vertical,
        road_profiles,
        rail_profiles,
        config,
    )
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
    underground = [edge for edge in edges if edge["vertical_mode"] == "underground"]
    elevated = [edge for edge in edges if edge["vertical_mode"] == "elevated"]
    assert underground
    assert elevated
    assert any(edge["minimum_z_m"] < edge["maximum_z_m"] for edge in underground)
    assert any(edge["minimum_z_m"] < edge["maximum_z_m"] for edge in elevated)
    assert all(edge["maximum_grade"] <= 0.081 for edge in underground)
    assert all(edge["maximum_grade"] <= 0.036 for edge in elevated)
