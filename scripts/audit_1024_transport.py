from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, convolve, label
from skimage.morphology import skeletonize

from urban_model.config import load_layered_diffusion_config
from urban_model.data import LayeredBlockDataset, SURFACE_CLASS_COUNT

PALETTE = np.asarray(
    [
        (226, 221, 209),
        (111, 174, 105),
        (137, 142, 148),
        (215, 58, 48),
        (239, 116, 66),
        (246, 180, 90),
        (85, 176, 194),
        (78, 151, 211),
    ],
    dtype=np.uint8,
)

LEVELS = ("low", "mid", "high")
SEEDS = (101, 202, 303)


def endpoint_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    skeleton = skeletonize(mask)
    neighbours = convolve(
        skeleton.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    )
    endpoints = skeleton & (neighbours == 2)
    return endpoints, skeleton


def component_stats(mask: np.ndarray) -> tuple[int, float]:
    labels, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0, 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return int(count), float(sizes.max() / max(sizes.sum(), 1))


def transport_metrics(classes: np.ndarray) -> dict[str, float | int]:
    road = (classes >= 3) & (classes <= 5)
    rail = classes == 6

    road_endpoints, road_skeleton = endpoint_mask(road)
    rail_endpoints, rail_skeleton = endpoint_mask(rail)

    road_near_rail = binary_dilation(rail, iterations=3)
    rail_near_road = binary_dilation(road, iterations=3)

    road_components, road_largest = component_stats(road_skeleton)
    rail_components, rail_largest = component_stats(rail_skeleton)

    rail_endpoint_count = int(rail_endpoints.sum())
    road_endpoint_count = int(road_endpoints.sum())

    interior = np.ones(classes.shape, dtype=bool)
    interior[:4] = False
    interior[-4:] = False
    interior[:, :4] = False
    interior[:, -4:] = False
    interior_road_endpoints = road_endpoints & interior
    interior_rail_endpoints = rail_endpoints & interior

    return {
        "road_fraction": float(road.mean()),
        "rail_fraction": float(rail.mean()),
        "road_components": road_components,
        "rail_components": rail_components,
        "road_largest_component_fraction": road_largest,
        "rail_largest_component_fraction": rail_largest,
        "road_endpoints": road_endpoint_count,
        "rail_endpoints": rail_endpoint_count,
        "road_endpoint_near_rail_fraction": float(
            (road_endpoints & road_near_rail).sum() / max(road_endpoint_count, 1)
        ),
        "rail_endpoint_near_road_fraction": float(
            (rail_endpoints & rail_near_road).sum() / max(rail_endpoint_count, 1)
        ),
        "road_interior_endpoints": int(interior_road_endpoints.sum()),
        "rail_interior_endpoints": int(interior_rail_endpoints.sum()),
        "road_interior_endpoint_near_rail_fraction": float(
            (interior_road_endpoints & road_near_rail).sum()
            / max(interior_road_endpoints.sum(), 1)
        ),
        "rail_interior_endpoint_near_road_fraction": float(
            (interior_rail_endpoints & rail_near_road).sum()
            / max(interior_rail_endpoints.sum(), 1)
        ),
    }


def surface_image(classes: np.ndarray) -> Image.Image:
    return Image.fromarray(PALETTE[classes.astype(np.int64)])


def transport_image(classes: np.ndarray) -> Image.Image:
    image = np.full((*classes.shape, 3), 245, dtype=np.uint8)
    for index in (3, 4, 5, 6):
        image[classes == index] = PALETTE[index]
    return Image.fromarray(image)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_real_tiles(entries: list[dict], output: Path, count: int) -> None:
    selected = sorted(entries, key=lambda item: item["metrics"]["rail_fraction"], reverse=True)[:count]
    tile_dir = output / "real-tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    overview_size = 384
    header = 24
    columns = 3
    rows = (len(selected) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * overview_size, rows * (overview_size + header)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for index, entry in enumerate(selected):
        tile_id = entry["tile_id"]
        classes = entry["classes"]
        surface = surface_image(classes)
        transport = transport_image(classes)
        surface.save(tile_dir / f"{tile_id}-surface.png", optimize=True)
        transport.save(tile_dir / f"{tile_id}-transport.png", optimize=True)

        row, column = divmod(index, columns)
        x = column * overview_size
        y = row * (overview_size + header)
        draw.text(
            (x + 4, y + 5),
            f"{tile_id} rail={entry['metrics']['rail_fraction']:.4f}",
            fill="black",
        )
        canvas.paste(
            surface.resize((overview_size, overview_size), Image.Resampling.NEAREST),
            (x, y + header),
        )

    canvas.save(output / "real-rail-rich-overview.png", optimize=True)


def decode_palette(image: np.ndarray) -> np.ndarray:
    classes = np.full(image.shape[:2], -1, dtype=np.int8)
    for index, colour in enumerate(PALETTE):
        classes[np.all(image == colour, axis=-1)] = index
    if (classes < 0).any():
        raise ValueError("Found colours outside the expected surface palette")
    return classes


def audit_generated_sweeps(path: Path) -> list[dict]:
    rows: list[dict] = []
    for image_path in sorted(path.glob("*.png")):
        image = np.asarray(Image.open(image_path).convert("RGB"))
        tile_size = image.shape[1] // 3
        row_height = image.shape[0] // 3
        header = row_height - tile_size
        if tile_size <= 0 or header < 0:
            raise ValueError(f"Unexpected sweep layout: {image_path}")

        for row_index, seed in enumerate(SEEDS):
            for column_index, level_name in enumerate(LEVELS):
                y = row_index * row_height + header
                x = column_index * tile_size
                crop = image[y : y + tile_size, x : x + tile_size]
                classes = decode_palette(crop)
                rows.append(
                    {
                        "control": image_path.stem,
                        "seed": seed,
                        "level": level_name,
                        **transport_metrics(classes),
                    }
                )
    return rows


def real_dataset(source_root: Path, split: str) -> LayeredBlockDataset:
    config = load_layered_diffusion_config(source_root / "configs/layered-corpus-v2-1km.yaml")
    config = replace(
        config,
        train_manifest=source_root / "data/manifests/corpus-v2-1024/train.jsonl",
        validation_manifest=source_root / "data/manifests/corpus-v2-1024/validation.jsonl",
        resolution=(1024, 1024),
        crop_size_pixels=1024,
        crop_stride_pixels=1024,
        vertical_crop_repeat=1,
    )
    manifest = source_root / f"data/manifests/corpus-v2-1024/{split}.jsonl"
    return LayeredBlockDataset(config, manifest, augment=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-sweeps", type=Path)
    parser.add_argument("--real-preview-count", type=int, default=9)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    dataset = real_dataset(source_root, args.split)
    real_entries = []
    real_rows = []

    for index in range(len(dataset)):
        sample = dataset[index]
        classes = sample["x0"][:SURFACE_CLASS_COUNT].argmax(dim=0).numpy().astype(np.uint8)
        tile_id = str(sample["tile_id"]).rsplit(":", 2)[0]
        metrics = transport_metrics(classes)
        real_entries.append({"tile_id": tile_id, "classes": classes, "metrics": metrics})
        real_rows.append({"tile_id": tile_id, **metrics})

    write_csv(output / f"real-{args.split}.csv", real_rows)
    save_real_tiles(real_entries, output, min(args.real_preview_count, len(real_entries)))

    summary = {
        "split": args.split,
        "real_tiles": len(real_rows),
        "real_tiles_with_rail": sum(row["rail_fraction"] > 0 for row in real_rows),
    }

    if args.generated_sweeps:
        generated_rows = audit_generated_sweeps(args.generated_sweeps.expanduser().resolve())
        write_csv(output / "generated-sweeps.csv", generated_rows)
        summary["generated_sweep_rows"] = len(generated_rows)

    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
