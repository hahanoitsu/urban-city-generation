from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from shapely.geometry import LineString, Polygon, shape
from torch import nn
from torch.utils.data import Dataset

from .schema import STYLE_FIELDS, city_style, style_vector

TOKEN_TYPES = ("pad", "road", "rail", "building")
TOKEN_CLASSES = ("major", "secondary", "local", "rail", "subway", "light_rail", "tram")
VERTICAL_MODES = ("surface", "underground", "elevated", "unknown")

TYPE_OFFSET = 0
CLASS_OFFSET = TYPE_OFFSET + len(TOKEN_TYPES)
VERTICAL_OFFSET = CLASS_OFFSET + len(TOKEN_CLASSES)
GEOMETRY_OFFSET = VERTICAL_OFFSET + len(VERTICAL_MODES)

CX = GEOMETRY_OFFSET
CY = CX + 1
CZ = CY + 1
DX = CZ + 1
DY = DX + 1
DZ = DY + 1
LENGTH = DZ + 1
WIDTH = LENGTH + 1
HEIGHT = WIDTH + 1
AREA = HEIGHT + 1
TOKEN_DIM = AREA + 1

XY_SCALE_M = 512.0
Z_SCALE_M = 96.0
LENGTH_SCALE_M = 1024.0
WIDTH_SCALE_M = 32.0
HEIGHT_SCALE_M = 192.0
BUILDING_LENGTH_SCALE_M = 256.0
BUILDING_WIDTH_SCALE_M = 128.0
AREA_SCALE_M2 = 100_000.0


@dataclass(frozen=True)
class Object3DConfig:
    maximum_tokens: int = 512
    model_dimensions: int = 256
    attention_heads: int = 8
    layers: int = 6
    feedforward_dimensions: int = 1024
    dropout: float = 0.05


def _one_hot(index: int, count: int) -> np.ndarray:
    values = np.zeros(count, dtype=np.float32)
    values[index] = 1.0
    return values


def _normalise_xy(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return float(np.clip(((value - lower) / (upper - lower)) * 2.0 - 1.0, -1.0, 1.0))


def _edge_token(edge: dict[str, Any], bounds: list[float]) -> np.ndarray | None:
    coordinates = edge.get("geometry_local_m") or []
    points = [point for point in coordinates if len(point) >= 2]
    if len(points) < 2:
        return None

    start = points[0]
    end = points[-1]
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    z1 = float(start[2]) if len(start) >= 3 and start[2] is not None else 0.0
    z2 = float(end[2]) if len(end) >= 3 and end[2] is not None else z1

    vx = x2 - x1
    vy = y2 - y1
    vz = z2 - z1
    length = math.sqrt(vx * vx + vy * vy + vz * vz)
    if length <= 1e-5:
        return None

    mode = "rail" if str(edge.get("transport_mode")).lower() == "rail" else "road"
    token_type = TOKEN_TYPES.index(mode)
    edge_class = str(edge.get("class") or ("rail" if mode == "rail" else "local")).lower()
    if edge_class not in TOKEN_CLASSES:
        edge_class = "rail" if mode == "rail" else "local"
    vertical = str(edge.get("vertical_mode") or "unknown").lower()
    if vertical not in VERTICAL_MODES:
        vertical = "unknown"

    minx, miny, maxx, maxy = [float(value) for value in bounds]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    cz = (z1 + z2) / 2.0

    token = np.zeros(TOKEN_DIM, dtype=np.float32)
    token[TYPE_OFFSET:CLASS_OFFSET] = _one_hot(token_type, len(TOKEN_TYPES))
    token[CLASS_OFFSET:VERTICAL_OFFSET] = _one_hot(
        TOKEN_CLASSES.index(edge_class), len(TOKEN_CLASSES)
    )
    token[VERTICAL_OFFSET:GEOMETRY_OFFSET] = _one_hot(
        VERTICAL_MODES.index(vertical), len(VERTICAL_MODES)
    )
    token[CX] = _normalise_xy(cx, minx, maxx)
    token[CY] = _normalise_xy(cy, miny, maxy)
    token[CZ] = float(np.clip(cz / Z_SCALE_M, -1.0, 1.0))
    token[DX] = vx / length
    token[DY] = vy / length
    token[DZ] = vz / length
    token[LENGTH] = float(np.clip(length / LENGTH_SCALE_M, 0.0, 1.0))
    token[WIDTH] = float(np.clip(float(edge.get("width_m", 5.0)) / WIDTH_SCALE_M, 0.0, 1.0))
    return token


def _rectangle_geometry(geometry) -> tuple[float, float, float, float, float]:
    rectangle = geometry.minimum_rotated_rectangle
    if not isinstance(rectangle, Polygon):
        return float(geometry.centroid.x), float(geometry.centroid.y), 1.0, 1.0, 0.0

    points = list(rectangle.exterior.coords)[:-1]
    if len(points) != 4:
        return float(geometry.centroid.x), float(geometry.centroid.y), 1.0, 1.0, 0.0

    edges = []
    for left, right in zip(points, points[1:] + points[:1]):
        vx = right[0] - left[0]
        vy = right[1] - left[1]
        edges.append((math.hypot(vx, vy), vx, vy))
    length, vx, vy = max(edges, key=lambda item: item[0])
    width = max(min(edge[0] for edge in edges), 0.5)
    norm = max(math.hypot(vx, vy), 1e-6)
    return (
        float(geometry.centroid.x),
        float(geometry.centroid.y),
        float(length),
        float(width),
        float(math.atan2(vy / norm, vx / norm)),
    )


def _building_token(building: dict[str, Any], bounds: list[float]) -> np.ndarray | None:
    footprint = building.get("footprint_local_m")
    if not footprint:
        return None
    geometry = shape(footprint)
    if geometry.is_empty or geometry.area <= 1e-5:
        return None

    cx, cy, length, width, angle = _rectangle_geometry(geometry)
    height = max(1.0, float(building.get("height_m", 9.3)))
    base_z = float(building.get("base_z_m", 0.0))
    cz = base_z + height / 2.0
    minx, miny, maxx, maxy = [float(value) for value in bounds]

    token = np.zeros(TOKEN_DIM, dtype=np.float32)
    token[TYPE_OFFSET:CLASS_OFFSET] = _one_hot(
        TOKEN_TYPES.index("building"), len(TOKEN_TYPES)
    )
    token[VERTICAL_OFFSET:GEOMETRY_OFFSET] = _one_hot(
        VERTICAL_MODES.index("surface"), len(VERTICAL_MODES)
    )
    token[CX] = _normalise_xy(cx, minx, maxx)
    token[CY] = _normalise_xy(cy, miny, maxy)
    token[CZ] = float(np.clip(cz / Z_SCALE_M, -1.0, 1.0))
    token[DX] = math.cos(angle)
    token[DY] = math.sin(angle)
    token[DZ] = 0.0
    token[LENGTH] = float(np.clip(length / BUILDING_LENGTH_SCALE_M, 0.0, 1.0))
    token[WIDTH] = float(np.clip(width / BUILDING_WIDTH_SCALE_M, 0.0, 1.0))
    token[HEIGHT] = float(np.clip(height / HEIGHT_SCALE_M, 0.0, 1.0))
    token[AREA] = float(np.clip(float(geometry.area) / AREA_SCALE_M2, 0.0, 1.0))
    return token


def state_tokens(
    payload: dict[str, Any],
    maximum_tokens: int,
) -> tuple[np.ndarray, int, int]:
    bounds = [
        float(value)
        for value in payload.get("coordinate_system", {}).get(
            "local_bounds", [0.0, 0.0, 1024.0, 1024.0]
        )
    ]

    tokens: list[np.ndarray] = []
    graph = payload.get("transport_graph", {})
    for edge in graph.get("edges", []):
        token = _edge_token(edge, bounds)
        if token is not None:
            tokens.append(token)

    buildings: list[tuple[float, np.ndarray]] = []
    for building in payload.get("building_solids", []):
        token = _building_token(building, bounds)
        if token is None:
            continue
        buildings.append((float(token[AREA]), token))
    buildings.sort(key=lambda item: item[0], reverse=True)
    tokens.extend(token for _area, token in buildings)
    total_count = len(tokens)

    if len(tokens) > maximum_tokens:
        tokens = tokens[:maximum_tokens]

    output = np.zeros((maximum_tokens, TOKEN_DIM), dtype=np.float32)
    output[:, TYPE_OFFSET] = 1.0
    if tokens:
        output[: len(tokens)] = np.stack(tokens)
    return output, len(tokens), total_count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class CityObject3DDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        config: Object3DConfig,
        *,
        style_mean: np.ndarray | None = None,
        style_std: np.ndarray | None = None,
        augment: bool = False,
    ) -> None:
        self.manifest = Path(manifest).expanduser().resolve()
        self.root = self.manifest.parent
        self.rows = _read_jsonl(self.manifest)
        self.config = config
        self.augment = bool(augment)

        styles = []
        states = []
        accepted = []
        for row in self.rows:
            sample_path = (self.root / str(row["sample_path"])).resolve()
            state_path = sample_path.parent / "city.json"
            if not state_path.exists():
                continue
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            tokens, count, total_count = state_tokens(payload, config.maximum_tokens)
            if count == 0:
                continue
            style = np.asarray(style_vector(city_style(payload)), dtype=np.float32)
            states.append(
                (
                    tokens,
                    count,
                    total_count,
                    str(row.get("tile_id", state_path.parent.name)),
                )
            )
            styles.append(style)
            accepted.append(row)

        if not states:
            raise ValueError(f"No city-state samples found for {self.manifest}")

        self.states = states
        self.rows = accepted
        style_array = np.stack(styles)
        if style_mean is None:
            style_mean = style_array.mean(axis=0)
        if style_std is None:
            style_std = style_array.std(axis=0)
        style_std = np.asarray(style_std, dtype=np.float32)
        style_std[style_std < 1e-6] = 1.0
        self.style_mean = np.asarray(style_mean, dtype=np.float32)
        self.style_std = style_std
        self.styles = (style_array - self.style_mean) / self.style_std

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, Any]:
        tokens, count, total_count, tile_id = self.states[index]
        values = tokens.copy()
        style = self.styles[index].copy()
        if self.augment:
            order = np.arange(values.shape[0])
            real = order[:count].copy()
            np.random.shuffle(real)
            order[:count] = real
            values = values[order]
        return {
            "tokens": torch.from_numpy(values),
            "style": torch.from_numpy(style),
            "count": count,
            "total_count": total_count,
            "truncated": max(0, total_count - count),
            "tile_id": tile_id,
        }


class Object3DDenoiser(nn.Module):
    def __init__(self, config: Object3DConfig, style_dimensions: int = len(STYLE_FIELDS)) -> None:
        super().__init__()
        self.config = config
        d = config.model_dimensions
        self.input = nn.Linear(TOKEN_DIM, d)
        self.time = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.style = nn.Sequential(
            nn.Linear(style_dimensions, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dimensions,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, TOKEN_DIM))

    def forward(
        self,
        tokens: torch.Tensor,
        timestep: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input(tokens)
        hidden = hidden + self.time(timestep[:, None]).unsqueeze(1)
        hidden = hidden + self.style(style).unsqueeze(1)
        hidden = self.transformer(hidden)
        return self.output(hidden)


def diffusion_coefficients(timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    angle = timestep * (math.pi / 2.0)
    return torch.cos(angle), torch.sin(angle)


def noisy_tokens(
    clean: torch.Tensor,
    timestep: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    alpha, sigma = diffusion_coefficients(timestep)
    return alpha[:, None, None] * clean + sigma[:, None, None] * noise


def object_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    token_type = target[:, :, TYPE_OFFSET:CLASS_OFFSET].argmax(dim=-1)
    object_weight = torch.where(token_type.eq(0), 0.12, 1.0).unsqueeze(-1)

    weights = prediction.new_ones(TOKEN_DIM)
    weights[TYPE_OFFSET:CLASS_OFFSET] = 2.0
    weights[CLASS_OFFSET:VERTICAL_OFFSET] = 1.5
    weights[VERTICAL_OFFSET:GEOMETRY_OFFSET] = 1.5
    weights[CX:HEIGHT + 1] = 2.0

    squared = (prediction.float() - target.float()).square()
    weighted = squared * object_weight.float() * weights
    return weighted.sum() / (object_weight.sum() * weights.sum()).clamp_min(1.0)


@torch.inference_mode()
def sample_tokens(
    model: Object3DDenoiser,
    style: torch.Tensor,
    *,
    steps: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    values = torch.randn(
        (style.shape[0], model.config.maximum_tokens, TOKEN_DIM),
        generator=generator,
        dtype=torch.float32,
    ).to(device)

    model.eval()
    times = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for index in range(steps):
        current = times[index].expand(style.shape[0])
        following = times[index + 1].expand(style.shape[0])
        prediction = model(values, current, style)

        alpha, sigma = diffusion_coefficients(current)
        next_alpha, next_sigma = diffusion_coefficients(following)
        alpha = alpha[:, None, None]
        sigma = sigma[:, None, None].clamp_min(1e-5)
        next_alpha = next_alpha[:, None, None]
        next_sigma = next_sigma[:, None, None]

        estimated_noise = (values - alpha * prediction) / sigma
        values = next_alpha * prediction + next_sigma * estimated_noise

    return model(values, torch.zeros(style.shape[0], device=device), style)


def token_summary(tokens: torch.Tensor) -> dict[str, int]:
    types = tokens[:, TYPE_OFFSET:CLASS_OFFSET].argmax(dim=-1)
    return {
        name: int(types.eq(index).sum().item())
        for index, name in enumerate(TOKEN_TYPES)
    }
