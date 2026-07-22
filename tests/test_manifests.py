import csv
import json

from urban_dataset.manifests import build_manifests


def test_manifest_paths_are_relative_and_splits_are_populated(tmp_path):
    processed = tmp_path / "data" / "processed" / "city"
    tiles = processed / "tiles"
    processed.mkdir(parents=True)
    rows = []
    for row in range(4):
        for column in range(4):
            tile_id = f"city_x{column:+06d}_y{row:+06d}"
            tile_dir = tiles / tile_id
            tile_dir.mkdir(parents=True)
            (tile_dir / "layers.npz").write_bytes(b"test")
            (tile_dir / "metadata.json").write_text("{}")
            rows.append({"tile_id": tile_id, "city_id": "city", "column": column, "row": row})
    with (processed / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tile_id", "city_id", "column", "row"])
        writer.writeheader()
        writer.writerows(rows)

    config = tmp_path / "configs" / "dataset.yaml"
    config.parent.mkdir()
    config.write_text(
        "dataset_root: data/processed\n"
        "manifest_root: data/manifests\n"
        "seed: 5\n"
        "split_ratios: {train: 0.5, validation: 0.25, test: 0.25}\n"
        "spatial_group_tiles: 2\n"
    )
    summary = build_manifests(config)
    assert summary["train"] and summary["validation"] and summary["test"]
    manifest = tmp_path / "data" / "manifests" / "train.jsonl"
    row = json.loads(manifest.read_text().splitlines()[0])
    assert not row["sample_path"].startswith("/")
