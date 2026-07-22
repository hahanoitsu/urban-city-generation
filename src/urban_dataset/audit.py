from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .schema import CHANNEL_NAMES, HEIGHT_CONFIDENCE_LABELS
from .utils import write_json


def audit_dataset(dataset_root: str | Path, output: str | Path | None = None) -> dict:
    root = Path(dataset_root).expanduser().resolve()
    tile_dirs = sorted(path.parent for path in root.glob("tiles/*/layers.npz"))
    if not tile_dirs:
        raise FileNotFoundError(f"No tile archives found below {root / 'tiles'}")

    pixel_sum = np.zeros(len(CHANNEL_NAMES), dtype=np.float64)
    positive_sum = np.zeros(len(CHANNEL_NAMES), dtype=np.float64)
    valid_total = 0
    invalid_samples: list[dict] = []
    shape_counts: dict[str, int] = defaultdict(int)
    hashes: dict[str, list[str]] = defaultdict(list)
    building_pixels = 0
    road_pixels = 0
    building_road_overlap = 0
    landuse_known_pixels = 0
    building_without_classified_landuse = 0
    building_without_osm_landuse_coverage = 0
    road_centerline_pixels = np.zeros(3, dtype=np.int64)
    height_confidence_pixels = np.zeros(4, dtype=np.int64)
    road_overlap_pixels = 0
    landuse_overlap_pixels = 0

    for tile_dir in tile_dirs:
        archive_path = tile_dir / "layers.npz"
        try:
            with np.load(archive_path, allow_pickle=False) as archive:
                layers = archive["layers"]
                valid = archive["valid_data_mask"].astype(bool)
                confidence = archive["height_confidence"].astype(np.uint8)
                landuse_known = archive["landuse_known_mask"].astype(bool)
                centerlines = archive["road_centerlines"].astype(bool)
                names = tuple(str(value) for value in archive["channel_names"])
        except Exception as exc:
            invalid_samples.append({"tile_id": tile_dir.name, "reason": f"read_error:{exc}"})
            continue

        shape_counts[str(tuple(layers.shape))] += 1
        if names != CHANNEL_NAMES:
            invalid_samples.append({"tile_id": tile_dir.name, "reason": "channel_name_mismatch"})
            continue
        if layers.shape[0] != len(CHANNEL_NAMES) or layers.ndim != 3:
            invalid_samples.append({"tile_id": tile_dir.name, "reason": "shape_mismatch"})
            continue
        if confidence.shape != layers.shape[1:] or landuse_known.shape != layers.shape[1:]:
            invalid_samples.append({"tile_id": tile_dir.name, "reason": "auxiliary_shape_mismatch"})
            continue
        if centerlines.shape != (3, *layers.shape[1:]):
            invalid_samples.append({"tile_id": tile_dir.name, "reason": "centerline_shape_mismatch"})
            continue
        if not np.isfinite(layers).all():
            invalid_samples.append({"tile_id": tile_dir.name, "reason": "non_finite_values"})
            continue
        if layers.min() < 0 or layers.max() > 1:
            invalid_samples.append({"tile_id": tile_dir.name, "reason": "value_out_of_range"})
            continue
        if confidence.max(initial=0) > 3:
            invalid_samples.append({"tile_id": tile_dir.name, "reason": "height_confidence_out_of_range"})
            continue

        valid_count = int(valid.sum())
        if valid_count == 0:
            invalid_samples.append({"tile_id": tile_dir.name, "reason": "empty_valid_mask"})
            continue
        valid_total += valid_count
        for channel in range(len(CHANNEL_NAMES)):
            values = layers[channel][valid]
            pixel_sum[channel] += values.sum(dtype=np.float64)
            positive_sum[channel] += np.count_nonzero(values > 0.05)

        building = (layers[8] > 0.05) & valid
        road_classes = layers[1:4] > 0.05
        road = road_classes.any(axis=0) & valid
        landuse_classes = layers[[4, 5, 6, 7, 10]] > 0.05
        building_count = int(building.sum())
        building_pixels += building_count
        road_pixels += int(road.sum())
        building_road_overlap += int((building & road).sum())
        landuse_known_pixels += int((landuse_known & valid).sum())
        classified_landuse = landuse_classes.any(axis=0) & valid
        building_without_classified_landuse += int((building & ~classified_landuse).sum())
        building_without_osm_landuse_coverage += int((building & ~landuse_known).sum())
        road_centerline_pixels += centerlines[:, valid].sum(axis=1, dtype=np.int64)
        road_overlap_pixels += int(((road_classes.sum(axis=0) > 1) & valid).sum())
        landuse_overlap_pixels += int(((landuse_classes.sum(axis=0) > 1) & valid).sum())
        for level in range(4):
            height_confidence_pixels[level] += int(((confidence == level) & building).sum())

        digest = hashlib.sha256()
        digest.update(layers.tobytes(order="C"))
        digest.update(confidence.tobytes(order="C"))
        digest.update(landuse_known.tobytes(order="C"))
        digest.update(centerlines.tobytes(order="C"))
        hashes[digest.hexdigest()].append(tile_dir.name)

    duplicate_groups = [members for members in hashes.values() if len(members) > 1]
    channel_means = {
        name: round(float(pixel_sum[i] / max(valid_total, 1)), 8)
        for i, name in enumerate(CHANNEL_NAMES)
    }
    channel_coverage = {
        name: round(float(positive_sum[i] / max(valid_total, 1)), 8)
        for i, name in enumerate(CHANNEL_NAMES)
    }
    confidence_distribution = {
        HEIGHT_CONFIDENCE_LABELS[level]: round(
            float(height_confidence_pixels[level] / max(building_pixels, 1)), 8
        )
        for level in range(4)
    }
    try:
        portable_root = root.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        portable_root = root.name
    report = {
        "dataset_root": portable_root,
        "tile_archives_found": len(tile_dirs),
        "valid_tiles": len(tile_dirs) - len(invalid_samples),
        "invalid_tiles": invalid_samples,
        "shape_counts": dict(sorted(shape_counts.items())),
        "valid_pixels": valid_total,
        "building_pixels": building_pixels,
        "channel_mean": channel_means,
        "channel_coverage_above_0_05": channel_coverage,
        "height_confidence_fraction_of_building_pixels": confidence_distribution,
        "landuse_known_fraction_of_valid_pixels": round(
            landuse_known_pixels / max(valid_total, 1), 8
        ),
        "building_without_classified_landuse_fraction": round(
            building_without_classified_landuse / max(building_pixels, 1), 8
        ),
        "building_without_osm_landuse_coverage_fraction": round(
            building_without_osm_landuse_coverage / max(building_pixels, 1), 8
        ),
        "building_road_overlap_fraction": round(
            building_road_overlap / max(building_pixels, 1), 8
        ),
        "road_centerline_fraction_of_valid_pixels": {
            name: round(float(road_centerline_pixels[index] / max(valid_total, 1)), 8)
            for index, name in enumerate(["major", "secondary", "local"])
        },
        "road_hierarchy_overlap_pixels": road_overlap_pixels,
        "landuse_class_overlap_pixels": landuse_overlap_pixels,
        "road_surface_fraction": round(road_pixels / max(valid_total, 1), 8),
        "exact_duplicate_groups": duplicate_groups,
    }
    destination = Path(output).expanduser().resolve() if output else root / "audit.json"
    write_json(destination, report)
    return report


def create_preview_atlas(
    dataset_root: str | Path,
    output: str | Path,
    *,
    columns: int = 6,
    limit: int = 120,
    thumbnail_size: int = 192,
) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    previews = sorted(root.glob("tiles/*/preview.png"))[:limit]
    if not previews:
        raise FileNotFoundError(f"No preview images found below {root / 'tiles'}")
    columns = max(1, columns)
    rows = (len(previews) + columns - 1) // columns
    label_height = 28
    cell_width = thumbnail_size
    cell_height = thumbnail_size + label_height
    atlas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(atlas)

    for index, preview in enumerate(previews):
        image = Image.open(preview).convert("RGB")
        image.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.NEAREST)
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        atlas.paste(image, (x, y))
        label = preview.parent.name
        if len(label) > 30:
            label = label[:27] + "..."
        draw.text((x + 4, y + thumbnail_size + 6), label, fill="black")

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(destination, optimize=True)
    return destination
