from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry.base import BaseGeometry

from .config import BuildConfig
from .extract import CityLayers
from .schema import CHANNEL_NAMES
from .tile import TileSpec


@dataclass
class RasterResult:
    layers: np.ndarray
    height_confidence: np.ndarray
    landuse_known_mask: np.ndarray
    road_centerlines: np.ndarray
    valid_data_mask: np.ndarray
    transform: tuple[float, ...]
    building_heights_m: list[float]
    height_confidence_counts: dict[int, int]

    @property
    def height_known_mask(self) -> np.ndarray:
        """Compatibility view for older consumers."""
        return (self.height_confidence >= 2).astype(np.uint8)

    @property
    def observed_building_heights(self) -> int:
        return self.height_confidence_counts.get(2, 0) + self.height_confidence_counts.get(3, 0)


def _valid_geometries(geometries: Iterable[BaseGeometry]) -> list[BaseGeometry]:
    return [geometry for geometry in geometries if geometry is not None and not geometry.is_empty]


def _burn(
    geometries: Iterable[BaseGeometry],
    *,
    pixels: int,
    transform,
    all_touched: bool,
    dtype: str = "float32",
    values: Sequence[float | int] | None = None,
) -> np.ndarray:
    geometry_list = list(geometries)
    if values is not None and len(values) != len(geometry_list):
        raise ValueError("Raster values must match geometry count")

    pairs: list[tuple[BaseGeometry, float | int]] = []
    for index, geometry in enumerate(geometry_list):
        if geometry is None or geometry.is_empty:
            continue
        value = 1 if values is None else values[index]
        pairs.append((geometry, value))
    if not pairs:
        return np.zeros((pixels, pixels), dtype=dtype)
    return rasterize(
        shapes=pairs,
        out_shape=(pixels, pixels),
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype=dtype,
    )


def _polygon_only(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame
    return frame[frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()


def _select(frame: gpd.GeoDataFrame, column: str, value: str) -> gpd.GeoDataFrame:
    if frame.empty or column not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame[frame[column] == value].copy()


def _disjoint_priority(raw_masks: dict[str, np.ndarray], priority: list[str]) -> dict[str, np.ndarray]:
    occupied = np.zeros_like(next(iter(raw_masks.values())), dtype=bool)
    result: dict[str, np.ndarray] = {}
    for name in priority:
        active = (raw_masks[name] > 0) & ~occupied
        result[name] = active.astype(np.float32)
        occupied |= active
    return result


def rasterize_tile(
    tile: TileSpec,
    city: CityLayers,
    config: BuildConfig,
    *,
    valid_geometry: BaseGeometry | None = None,
) -> RasterResult:
    pixels = config.raster.pixels
    transform = from_bounds(tile.minx, tile.miny, tile.maxx, tile.maxy, pixels, pixels)
    arrays = np.zeros((len(CHANNEL_NAMES), pixels, pixels), dtype=np.float32)

    if valid_geometry is None:
        valid_geometry = tile.geometry
    valid_data_mask = _burn(
        [valid_geometry],
        pixels=pixels,
        transform=transform,
        all_touched=False,
        dtype="uint8",
    ).astype(np.uint8)

    water = _polygon_only(city.water)
    arrays[0] = _burn(
        water.geometry,
        pixels=pixels,
        transform=transform,
        all_touched=config.raster.all_touched,
    )

    road_surfaces: dict[str, np.ndarray] = {}
    road_centers: dict[str, np.ndarray] = {}
    for road_class in ["major", "secondary", "local"]:
        selected = _select(city.roads, "road_class", road_class)
        if "estimated_width_m" in selected.columns:
            widths = [float(value) for value in selected["estimated_width_m"]]
        else:
            widths = [float(config.roads.widths_m[road_class])] * len(selected)
        buffers = [
            geometry.buffer(width / 2.0, cap_style="flat", join_style="round")
            for geometry, width in zip(selected.geometry, widths, strict=True)
        ]
        road_surfaces[road_class] = _burn(
            buffers,
            pixels=pixels,
            transform=transform,
            all_touched=config.roads.surface_all_touched,
        )
        road_centers[road_class] = _burn(
            selected.geometry,
            pixels=pixels,
            transform=transform,
            all_touched=True,
        )

    # A pixel has one road hierarchy target. Higher-order roads win at junctions.
    roads_disjoint = _disjoint_priority(
        road_surfaces, ["major", "secondary", "local"]
    )
    centers_disjoint = _disjoint_priority(
        road_centers, ["major", "secondary", "local"]
    )
    arrays[1] = roads_disjoint["major"]
    arrays[2] = roads_disjoint["secondary"]
    arrays[3] = roads_disjoint["local"]
    road_centerlines = np.stack(
        [centers_disjoint["major"], centers_disjoint["secondary"], centers_disjoint["local"]],
        axis=0,
    ).astype(np.uint8)

    raw_landuse: dict[str, np.ndarray] = {}
    for landuse_class in ["residential", "commercial_mixed", "industrial", "civic", "green"]:
        selected = _polygon_only(_select(city.landuse, "landuse_class", landuse_class))
        raw_landuse[landuse_class] = _burn(
            selected.geometry,
            pixels=pixels,
            transform=transform,
            all_touched=config.raster.all_touched,
        )
    dedicated_green = _polygon_only(city.green)
    raw_landuse["green"] = np.maximum(
        raw_landuse["green"],
        _burn(
            dedicated_green.geometry,
            pixels=pixels,
            transform=transform,
            all_touched=config.raster.all_touched,
        ),
    )

    # Specific overlays take precedence over broad residential/commercial polygons.
    landuse_disjoint = _disjoint_priority(
        raw_landuse, ["green", "civic", "industrial", "commercial_mixed", "residential"]
    )
    arrays[4] = landuse_disjoint["residential"]
    arrays[5] = landuse_disjoint["commercial_mixed"]
    arrays[6] = landuse_disjoint["industrial"]
    arrays[7] = landuse_disjoint["green"]
    arrays[10] = landuse_disjoint["civic"]
    # Water is not simultaneously a land-use target.
    arrays[[4, 5, 6, 7, 10]] *= (arrays[0] <= 0)[None, :, :]

    known_landuse = _polygon_only(city.landuse_known)
    landuse_known_mask = _burn(
        known_landuse.geometry,
        pixels=pixels,
        transform=transform,
        all_touched=config.raster.all_touched,
        dtype="uint8",
    ).astype(np.uint8)

    buildings = _polygon_only(city.buildings)
    arrays[8] = _burn(
        buildings.geometry,
        pixels=pixels,
        transform=transform,
        all_touched=config.raster.all_touched,
    )

    if "estimated_height_m" not in buildings.columns or "height_confidence" not in buildings.columns:
        raise ValueError("Buildings must be enriched before rasterisation")
    absolute_heights = [float(value) for value in buildings["estimated_height_m"]]
    height_values = [
        min(max(value / config.raster.max_height_m, 0.0), 1.0)
        for value in absolute_heights
    ]
    confidence_values = [int(value) for value in buildings["height_confidence"]]
    arrays[9] = _burn(
        buildings.geometry,
        values=height_values,
        pixels=pixels,
        transform=transform,
        all_touched=config.raster.all_touched,
    )
    height_confidence = _burn(
        buildings.geometry,
        values=confidence_values,
        pixels=pixels,
        transform=transform,
        all_touched=config.raster.all_touched,
        dtype="uint8",
    ).astype(np.uint8)

    rail_mask = np.zeros((pixels, pixels), dtype=np.float32)
    if not city.rail.empty:
        rail_buffers = city.rail.geometry.buffer(3.0, cap_style="flat")
        rail_mask = _burn(
            rail_buffers,
            pixels=pixels,
            transform=transform,
            all_touched=False,
        )
    arrays[11] = rail_mask

    arrays *= valid_data_mask[None, :, :]
    height_confidence *= valid_data_mask
    landuse_known_mask *= valid_data_mask
    road_centerlines *= valid_data_mask[None, :, :]

    counts = {level: sum(value == level for value in confidence_values) for level in range(4)}
    return RasterResult(
        layers=np.clip(arrays, 0.0, 1.0),
        height_confidence=height_confidence,
        landuse_known_mask=landuse_known_mask,
        road_centerlines=road_centerlines,
        valid_data_mask=valid_data_mask,
        transform=tuple(transform)[:6],
        building_heights_m=absolute_heights,
        height_confidence_counts=counts,
    )
