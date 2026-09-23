from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label

from urban_dataset.torch_dataset import UrbanTileDataset


def component_stats(mask: np.ndarray) -> tuple[int, float]:
    labels, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0, 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return int(count), float(sizes.max() / max(sizes.sum(), 1))


def transport_masks(layers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    road = layers[1:4].max(axis=0) > 0.05
    rail = layers[11] > 0.05
    return road, rail


def measure(layers: np.ndarray) -> dict[str, float | int]:
    road, rail = transport_masks(layers)
    overlap = road & rail
    collapsed_road = road & ~rail

    road_components, road_largest = component_stats(road)
    collapsed_components, collapsed_largest = component_stats(collapsed_road)

    return {
        "road_fraction": float(road.mean()),
        "rail_fraction": float(rail.mean()),
        "overlap_fraction": float(overlap.mean()),
        "overlap_of_road": float(overlap.sum() / max(road.sum(), 1)),
        "overlap_of_rail": float(overlap.sum() / max(rail.sum(), 1)),
        "road_components_before": road_components,
        "road_components_after": collapsed_components,
        "road_component_delta": collapsed_components - road_components,
        "road_largest_before": road_largest,
        "road_largest_after": collapsed_largest,
    }


def source_image(layers: np.ndarray) -> Image.Image:
    road, rail = transport_masks(layers)
    overlap = road & rail

    image = np.full((*road.shape, 3), 245, dtype=np.uint8)
    image[road] = (246, 180, 90)
    image[rail] = (85, 176, 194)
    image[overlap] = (190, 80, 190)
    return Image.fromarray(image)


def collapsed_image(layers: np.ndarray) -> Image.Image:
    road, rail = transport_masks(layers)
    image = np.full((*road.shape, 3), 245, dtype=np.uint8)
    image[road & ~rail] = (246, 180, 90)
    image[rail] = (85, 176, 194)
    return Image.fromarray(image)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_overview(entries: list[dict], output: Path, count: int) -> None:
    selected = sorted(
        entries,
        key=lambda item: item["metrics"]["overlap_fraction"],
        reverse=True,
    )[:count]

    size = 384
    header = 28
    canvas = Image.new("RGB", (size * 2, len(selected) * (size + header)), "white")
    draw = ImageDraw.Draw(canvas)

    for row, entry in enumerate(selected):
        y = row * (size + header)
        metrics = entry["metrics"]
        draw.text(
            (4, y + 6),
            (
                f"{entry['tile_id']} overlap={metrics['overlap_fraction']:.5f} "
                f"components {metrics['road_components_before']} -> "
                f"{metrics['road_components_after']}"
            ),
            fill="black",
        )
        draw.text((size + 4, y + 6), "collapsed target", fill="black")

        source = source_image(entry["layers"]).resize(
            (size, size),
            Image.Resampling.NEAREST,
        )
        collapsed = collapsed_image(entry["layers"]).resize(
            (size, size),
            Image.Resampling.NEAREST,
        )
        canvas.paste(source, (0, y + header))
        canvas.paste(collapsed, (size, y + header))

    canvas.save(output / "transport-overlap-overview.png", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-count", type=int, default=8)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    entries = []
    for split in ("train", "validation", "test"):
        manifest = source_root / f"data/manifests/corpus-v2-1024/{split}.jsonl"
        dataset = UrbanTileDataset(manifest, include_auxiliary=False)

        for index in range(len(dataset)):
            sample = dataset[index]
            layers = sample["x"].numpy()
            metrics = measure(layers)
            tile_id = str(sample["tile_id"])
            rows.append({"split": split, "tile_id": tile_id, **metrics})
            entries.append(
                {
                    "split": split,
                    "tile_id": tile_id,
                    "layers": layers,
                    "metrics": metrics,
                }
            )

    write_csv(output / "transport-overlap.csv", rows)
    save_overview(entries, output, min(args.preview_count, len(entries)))

    overlap_rows = [row for row in rows if row["overlap_fraction"] > 0]
    cut_rows = [row for row in rows if row["road_component_delta"] > 0]
    summary = {
        "tiles": len(rows),
        "tiles_with_road_rail_overlap": len(overlap_rows),
        "tiles_with_component_increase_after_rail_priority": len(cut_rows),
        "max_overlap_fraction": max((row["overlap_fraction"] for row in rows), default=0.0),
        "max_road_component_delta": max((row["road_component_delta"] for row in rows), default=0),
    }

    (output / "transport-overlap-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
