import torch

from urban_model.structured_city import StructuredCityConfig, StructuredCityDenoiser


def test_structured_city_forward():
    config = StructuredCityConfig(
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
        model_dimensions=64,
        attention_heads=4,
        context_layers=2,
        transport_layers=2,
        building_layers=2,
        area_layers=2,
        feedforward_dimensions=128,
    )
    model = StructuredCityDenoiser(config)
    batch = 2
    scene = {
        "node_position": torch.randn(batch, 16, 3),
        "node_presence": torch.randint(0, 3, (batch, 16)),
        "edge_width": torch.randn(batch, 24, 1),
        "edge_shape": torch.randn(batch, 24, 4, 3),
        "edge_presence": torch.randint(0, 3, (batch, 24)),
        "edge_mode": torch.randint(0, 3, (batch, 24)),
        "edge_class": torch.randint(0, 8, (batch, 24)),
        "edge_vertical": torch.randint(0, 5, (batch, 24)),
        "building_shape": torch.randn(batch, 20, 6, 2),
        "building_height": torch.randn(batch, 20, 1),
        "building_base_z": torch.randn(batch, 20, 1),
        "building_presence": torch.randint(0, 3, (batch, 20)),
        "building_kind": torch.randint(0, 9, (batch, 20)),
        "area_shape": torch.randn(batch, 8, 8, 2),
        "area_presence": torch.randint(0, 3, (batch, 8)),
        "area_kind": torch.randint(0, 4, (batch, 8)),
    }
    context = torch.randn(batch, 12, 18)
    relations = torch.rand(batch, 4, 12, 12)
    context_padding = torch.zeros(batch, 12, dtype=torch.bool)
    ports = torch.randn(batch, 10, 21)
    port_padding = torch.zeros(batch, 10, dtype=torch.bool)
    time = torch.rand(batch)
    output = model(
        scene,
        context,
        relations,
        context_padding,
        ports,
        port_padding,
        time,
    )
    assert output["node_position"].shape == (batch, 16, 3)
    assert output["edge_from"].shape == (batch, 24, 16)
    assert output["edge_shape"].shape == (batch, 24, 4, 3)
    assert output["building_shape"].shape == (batch, 20, 6, 2)
    assert output["area_shape"].shape == (batch, 8, 8, 2)
