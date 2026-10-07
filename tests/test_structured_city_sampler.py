import importlib.util
from pathlib import Path

import torch

from urban_model.structured_city import StructuredCityConfig, StructuredCityDenoiser
from urban_model.structured_city_data import SceneTensorConfig


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sample_structured_city.py"
SPEC = importlib.util.spec_from_file_location("sample_structured_city", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def small_configs():
    scene = SceneTensorConfig(
        node_slots=16,
        edge_slots=24,
        building_slots=20,
        area_slots=8,
        edge_shape_points=4,
        building_points=6,
        area_points=8,
    )
    model = StructuredCityConfig(
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
    )
    return scene, model


def test_hold_absent_matches_training_zeros():
    scene_config, _model_config = small_configs()
    scene = MODULE.initial_scene(scene_config, torch.device("cpu"))
    counts = {"node": 3, "edge": 4, "building": 5, "area": 2}
    before = scene["building_shape"][:, :5].clone()
    MODULE.hold_absent(scene, counts, torch.zeros(1))
    assert torch.equal(scene["building_shape"][:, :5], before)
    assert torch.all(scene["building_shape"][:, 5:] == 0)
    assert torch.all(scene["node_position"][:, 3:] == 0)
    assert torch.all(scene["edge_mode"][:, 4:] == 0)

    MODULE.hold_absent(scene, counts, torch.ones(1))
    assert torch.all(scene["area_kind"][:, 2:] == MODULE.CATEGORY_MASKS["area_kind"])


def test_generate_runs_with_and_without_hold():
    torch.manual_seed(0)
    scene_config, model_config = small_configs()
    model = StructuredCityDenoiser(model_config).eval()
    context = torch.randn(1, 12, 18)
    relations = torch.rand(1, 4, 12, 12)
    context_padding = torch.zeros(1, 12, dtype=torch.bool)
    ports = torch.randn(1, 10, 21)
    port_padding = torch.zeros(1, 10, dtype=torch.bool)
    for hold in (False, True):
        result = MODULE.generate(
            model,
            context,
            relations,
            context_padding,
            ports,
            port_padding,
            scene_config,
            4,
            hold=hold,
        )
        assert result["building_shape"].shape == (20, 6, 2)
        nodes = int(result["node_presence"].sum())
        assert int(result["edge_from"].max()) < max(nodes, 1)
