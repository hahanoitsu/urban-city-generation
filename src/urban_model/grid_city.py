from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from urban_model.grid_city_data import GridConfig, candidate_table
from urban_model.structured_city import RelationContextEncoder, TimeEmbedding


@dataclass(frozen=True)
class GridCityConfig:
    context_dimensions: int
    relation_count: int
    port_dimensions: int
    node_grid: int = 16
    node_slots: int = 10
    edge_slots: int = 4
    building_grid: int = 16
    building_slots: int = 8
    area_grid: int = 8
    area_slots: int = 8
    curve_points: int = 2
    building_points: int = 24
    area_points: int = 48
    edge_classes: int = 7
    vertical_classes: int = 4
    building_classes: int = 8
    area_classes: int = 6
    model_dimensions: int = 256
    attention_heads: int = 8
    context_layers: int = 4
    layers: int = 8
    feedforward_dimensions: int = 1024
    key_dimensions: int = 64
    dropout: float = 0.0
    position_bands: int = 0

    @property
    def candidates(self) -> int:
        return 9 * self.node_slots

    def grid(self) -> GridConfig:
        return GridConfig(
            node_grid=self.node_grid,
            node_slots=self.node_slots,
            edge_slots=self.edge_slots,
            building_grid=self.building_grid,
            building_slots=self.building_slots,
            area_grid=self.area_grid,
            area_slots=self.area_slots,
            curve_points=self.curve_points,
            building_points=self.building_points,
            area_points=self.area_points,
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def fourier(xy, bands):
    scales = math.pi * 2.0 ** torch.arange(bands, device=xy.device, dtype=xy.dtype)
    angles = xy[..., None] * scales
    return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)


def cell_centres(grid, slots):
    cells = torch.arange(grid * grid).repeat_interleave(slots)
    row = torch.div(cells, grid, rounding_mode="floor")
    column = cells % grid
    return torch.stack([(column + 0.5) / grid * 2.0 - 1.0, (row + 0.5) / grid * 2.0 - 1.0], dim=-1)


def category_sizes(config: GridCityConfig):
    return {
        "node_presence": 2,
        "edge_target": config.candidates + 1,
        "edge_mode": 2,
        "edge_class": config.edge_classes,
        "edge_vertical": config.vertical_classes,
        "building_presence": 2,
        "building_kind": config.building_classes,
        "area_presence": 2,
        "area_kind": config.area_classes,
    }


class GridCityDenoiser(nn.Module):
    def __init__(self, config: GridCityConfig) -> None:
        super().__init__()
        self.config = config
        d = config.model_dimensions
        sizes = category_sizes(config)
        self.sizes = sizes
        edges = config.edge_slots
        curve = config.curve_points

        self.time = TimeEmbedding(d)
        self.context = RelationContextEncoder(config)
        self.port = nn.Sequential(nn.Linear(config.port_dimensions, d), nn.GELU(), nn.Linear(d, d))

        self.node_in = nn.Linear(2 + edges * (1 + curve * 2), d)
        self.node_presence_in = nn.Embedding(3, d)
        self.edge_target_in = nn.Embedding(edges * (sizes["edge_target"] + 1), d)
        self.edge_mode_in = nn.Embedding(edges * 3, d)
        self.edge_class_in = nn.Embedding(edges * (config.edge_classes + 1), d)
        self.edge_vertical_in = nn.Embedding(edges * (config.vertical_classes + 1), d)
        self.building_in = nn.Linear(3 + config.building_points * 2 + 1, d)
        self.building_presence_in = nn.Embedding(3, d)
        self.building_kind_in = nn.Embedding(config.building_classes + 1, d)
        self.area_in = nn.Linear(3 + config.area_points * 2, d)
        self.area_presence_in = nn.Embedding(3, d)
        self.area_kind_in = nn.Embedding(config.area_classes + 1, d)

        self.node_cell = nn.Embedding(config.node_grid**2, d)
        self.node_slot = nn.Embedding(config.node_slots, d)
        self.building_cell = nn.Embedding(config.building_grid**2, d)
        self.building_slot = nn.Embedding(config.building_slots, d)
        self.area_cell = nn.Embedding(config.area_grid**2, d)
        self.area_slot = nn.Embedding(config.area_slots, d)
        self.kind = nn.Embedding(3, d)

        layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dimensions,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerDecoder(layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(d)

        keys = config.key_dimensions
        self.node_presence_out = nn.Linear(d, 2)
        self.node_pos_out = nn.Linear(d, 2)
        self.edge_query = nn.Linear(d, edges * keys)
        self.edge_key = nn.Linear(d, keys)
        self.edge_none = nn.Linear(d, edges)
        self.edge_mode_out = nn.Linear(d, edges * 2)
        self.edge_class_out = nn.Linear(d, edges * config.edge_classes)
        self.edge_vertical_out = nn.Linear(d, edges * config.vertical_classes)
        self.edge_width_out = nn.Linear(d, edges)
        self.edge_curve_out = nn.Linear(d, edges * curve * 2)
        self.building_presence_out = nn.Linear(d, 2)
        self.building_frame_out = nn.Linear(d, 3)
        self.building_shape_out = nn.Linear(d, config.building_points * 2)
        self.building_height_out = nn.Linear(d, 1)
        self.building_kind_out = nn.Linear(d, config.building_classes)
        self.area_presence_out = nn.Linear(d, 2)
        self.area_frame_out = nn.Linear(d, 3)
        self.area_shape_out = nn.Linear(d, config.area_points * 2)
        self.area_kind_out = nn.Linear(d, config.area_classes)

        table = torch.from_numpy(candidate_table(config.grid()))
        self.register_buffer("candidates", table.clamp_min(0), persistent=False)
        self.register_buffer("candidate_valid", table >= 0, persistent=False)
        self.register_buffer("node_cells", torch.arange(config.node_grid**2).repeat_interleave(config.node_slots), persistent=False)
        self.register_buffer("node_slots", torch.arange(config.node_slots).repeat(config.node_grid**2), persistent=False)
        self.register_buffer("building_cells", torch.arange(config.building_grid**2).repeat_interleave(config.building_slots), persistent=False)
        self.register_buffer("building_slots", torch.arange(config.building_slots).repeat(config.building_grid**2), persistent=False)
        self.register_buffer("area_cells", torch.arange(config.area_grid**2).repeat_interleave(config.area_slots), persistent=False)
        self.register_buffer("area_slots", torch.arange(config.area_slots).repeat(config.area_grid**2), persistent=False)

        # same sin cos position code for ports and every cell so they line up in space
        if config.position_bands:
            self.place = nn.Linear(4 * config.position_bands, d)
            self.edge_geometry = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, edges))
            self.register_buffer("node_xy", cell_centres(config.node_grid, config.node_slots), persistent=False)
            self.register_buffer("building_xy", cell_centres(config.building_grid, config.building_slots), persistent=False)
            self.register_buffer("area_xy", cell_centres(config.area_grid, config.area_slots), persistent=False)

    def _offset(self, values, size):
        edges = values.shape[-1]
        return values + torch.arange(edges, device=values.device) * size

    def forward(self, scene, context_values, relations, context_padding, ports, port_padding, time):
        config = self.config
        batch = context_values.shape[0]
        time_hidden = self.time(time)
        context = self.context(context_values, relations, context_padding)
        bands = config.position_bands
        port_memory = self.port(ports)
        if bands:
            port_memory = port_memory + self.place(fourier(ports[..., :2], bands))
        port_memory = port_memory.masked_fill(port_padding[:, :, None], 0.0)
        memory = torch.cat([context, port_memory], dim=1)
        memory_padding = torch.cat([context_padding, port_padding], dim=1)

        nodes = torch.cat(
            [scene["node_pos"], scene["edge_width"].flatten(2), scene["edge_curve"].flatten(2)],
            dim=-1,
        )
        sizes = self.sizes
        node_hidden = (
            self.node_in(nodes)
            + self.node_presence_in(scene["node_presence"])
            + self.edge_target_in(self._offset(scene["edge_target"], sizes["edge_target"] + 1)).sum(2)
            + self.edge_mode_in(self._offset(scene["edge_mode"], 3)).sum(2)
            + self.edge_class_in(self._offset(scene["edge_class"], config.edge_classes + 1)).sum(2)
            + self.edge_vertical_in(self._offset(scene["edge_vertical"], config.vertical_classes + 1)).sum(2)
            + self.node_cell(self.node_cells)
            + self.node_slot(self.node_slots)
            + self.kind.weight[0]
        )
        buildings = torch.cat(
            [scene["building_frame"], scene["building_shape"].flatten(2), scene["building_height"]],
            dim=-1,
        )
        building_hidden = (
            self.building_in(buildings)
            + self.building_presence_in(scene["building_presence"])
            + self.building_kind_in(scene["building_kind"])
            + self.building_cell(self.building_cells)
            + self.building_slot(self.building_slots)
            + self.kind.weight[1]
        )
        areas = torch.cat([scene["area_frame"], scene["area_shape"].flatten(2)], dim=-1)
        area_hidden = (
            self.area_in(areas)
            + self.area_presence_in(scene["area_presence"])
            + self.area_kind_in(scene["area_kind"])
            + self.area_cell(self.area_cells)
            + self.area_slot(self.area_slots)
            + self.kind.weight[2]
        )
        if bands:
            node_hidden = node_hidden + self.place(fourier(self.node_xy, bands))
            building_hidden = building_hidden + self.place(fourier(self.building_xy, bands))
            area_hidden = area_hidden + self.place(fourier(self.area_xy, bands))
        tokens = torch.cat([node_hidden, building_hidden, area_hidden], dim=1) + time_hidden[:, None]
        hidden = self.norm(self.backbone(tokens, memory, memory_key_padding_mask=memory_padding))
        node_count = node_hidden.shape[1]
        building_count = building_hidden.shape[1]
        node_hidden = hidden[:, :node_count]
        building_hidden = hidden[:, node_count : node_count + building_count]
        area_hidden = hidden[:, node_count + building_count :]

        edges = config.edge_slots
        curve = config.curve_points
        keys = config.key_dimensions
        query = self.edge_query(node_hidden).reshape(batch, node_count, edges, keys)
        key = self.edge_key(node_hidden)[:, self.candidates]
        logits = torch.einsum("bnek,bnck->bnec", query, key) / math.sqrt(keys)
        if bands:
            # where each candidate is from this node right now, in cell widths
            here = self.node_xy + scene["node_pos"].float() / config.node_grid
            delta = (here[:, self.candidates] - here[:, :, None]) * (config.node_grid / 2.0)
            when = time[:, None, None, None].float().expand(-1, delta.shape[1], delta.shape[2], 1)
            geometry = self.edge_geometry(torch.cat([delta, delta.norm(dim=-1, keepdim=True), when], dim=-1))
            logits = logits + geometry.permute(0, 1, 3, 2)
        logits = logits.masked_fill(~self.candidate_valid[None, :, None, :], -1e4)
        target = torch.cat([self.edge_none(node_hidden)[..., None], logits], dim=-1)

        return {
            "node_presence": self.node_presence_out(node_hidden),
            "node_pos": self.node_pos_out(node_hidden),
            "edge_target": target,
            "edge_mode": self.edge_mode_out(node_hidden).reshape(batch, node_count, edges, 2),
            "edge_class": self.edge_class_out(node_hidden).reshape(batch, node_count, edges, config.edge_classes),
            "edge_vertical": self.edge_vertical_out(node_hidden).reshape(batch, node_count, edges, config.vertical_classes),
            "edge_width": self.edge_width_out(node_hidden).reshape(batch, node_count, edges, 1),
            "edge_curve": self.edge_curve_out(node_hidden).reshape(batch, node_count, edges, curve, 2),
            "building_presence": self.building_presence_out(building_hidden),
            "building_frame": self.building_frame_out(building_hidden),
            "building_shape": self.building_shape_out(building_hidden).reshape(batch, building_count, config.building_points, 2),
            "building_height": self.building_height_out(building_hidden),
            "building_kind": self.building_kind_out(building_hidden),
            "area_presence": self.area_presence_out(area_hidden),
            "area_frame": self.area_frame_out(area_hidden),
            "area_shape": self.area_shape_out(area_hidden).reshape(batch, area_hidden.shape[1], config.area_points, 2),
            "area_kind": self.area_kind_out(area_hidden),
        }
