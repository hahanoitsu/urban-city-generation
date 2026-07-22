from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import geopandas as gpd
from shapely.geometry import box

from .extract import CityLayers


@dataclass(frozen=True)
class TileSpec:
    city_id: str
    column: int
    row: int
    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def tile_id(self) -> str:
        # Grid indices are derived from a fixed projected origin, so the same area
        # keeps the same identifier when the requested bounding box changes.
        return f"{self.city_id}_x{self.column:+06d}_y{self.row:+06d}"

    @property
    def geometry(self):
        return box(self.minx, self.miny, self.maxx, self.maxy)


def iter_tile_specs(
    city_id: str,
    bounds: tuple[float, float, float, float],
    tile_size_m: float,
    stride_m: float,
    *,
    include_partial_tiles: bool,
) -> Iterator[TileSpec]:
    minx, miny, maxx, maxy = bounds
    if include_partial_tiles:
        first_col = math.floor(minx / stride_m)
        first_row = math.floor(miny / stride_m)
        last_col = math.floor((maxx - 1e-9) / stride_m)
        last_row = math.floor((maxy - 1e-9) / stride_m)
    else:
        # Use globally anchored lower-left corners while keeping every emitted tile
        # fully inside the requested study rectangle.
        first_col = math.ceil(minx / stride_m)
        first_row = math.ceil(miny / stride_m)
        last_col = math.floor((maxx - tile_size_m) / stride_m)
        last_row = math.floor((maxy - tile_size_m) / stride_m)

    if last_col < first_col or last_row < first_row:
        return

    for row in range(first_row, last_row + 1):
        for column in range(first_col, last_col + 1):
            tile_minx = column * stride_m
            tile_miny = row * stride_m
            yield TileSpec(
                city_id=city_id,
                column=column,
                row=row,
                minx=tile_minx,
                miny=tile_miny,
                maxx=tile_minx + tile_size_m,
                maxy=tile_miny + tile_size_m,
            )


def clip_to_tile(frame: gpd.GeoDataFrame, tile: TileSpec) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame.copy()
    indexes = list(frame.sindex.query(tile.geometry, predicate="intersects"))
    if not indexes:
        return frame.iloc[0:0].copy()
    clipped = frame.iloc[indexes].copy()
    clipped["geometry"] = clipped.geometry.intersection(tile.geometry)
    clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()
    return clipped


def clip_layers(layers: CityLayers, tile: TileSpec) -> CityLayers:
    return CityLayers(**{name: clip_to_tile(frame, tile) for name, frame in layers.items()})
