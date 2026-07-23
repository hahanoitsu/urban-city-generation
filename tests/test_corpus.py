import pytest

from urban_dataset.corpus import build_corpus, load_corpus_config
from urban_dataset.demo import run_demo


def _write_config(path, gpkg, output_root, manifest_root, areas):
    area_text = "\n".join(
        f"      - id: {area_id}\n        bbox_wgs84: {list(bounds)}"
        for area_id, bounds in areas
    )
    path.write_text(
        f"output_root: {output_root}\n"
        f"manifest_root: {manifest_root}\n"
        "overwrite: true\n"
        "split_ratios: {train: 1.0, validation: 0.0, test: 0.0}\n"
        "quality:\n"
        "  minimum_buildings: 0\n"
        "  minimum_road_length_m: 0\n"
        "  minimum_nonempty_fraction: 0.0\n"
        "  minimum_valid_fraction: 0.95\n"
        "  reject_water_fraction_above: 1.0\n"
        "cities:\n"
        "  - id: demo_singapore\n"
        f"    gpkg: {gpkg}\n"
        "    areas:\n"
        f"{area_text}\n"
    )


def test_corpus_build_reuses_prepared_city(tmp_path):
    demo = tmp_path / "demo"
    run_demo(demo)
    config_path = tmp_path / "configs" / "corpus.yaml"
    config_path.parent.mkdir()
    _write_config(
        config_path,
        demo / "extracted_layers.gpkg",
        tmp_path / "processed",
        tmp_path / "manifests",
        [("centre", (103.8460, 1.2860, 103.8640, 1.3040))],
    )

    summary = build_corpus(load_corpus_config(config_path))

    assert summary["areas"] == 1
    assert summary["accepted_tiles"] > 0
    assert (tmp_path / "processed" / "demo_singapore-centre" / "atlas.png").exists()
    assert (tmp_path / "processed" / "audit.json").exists()
    assert (tmp_path / "manifests" / "train.jsonl").exists()


def test_overlapping_areas_are_rejected(tmp_path):
    demo = tmp_path / "demo"
    run_demo(demo)
    config_path = tmp_path / "configs" / "corpus.yaml"
    config_path.parent.mkdir()
    _write_config(
        config_path,
        demo / "extracted_layers.gpkg",
        tmp_path / "processed",
        tmp_path / "manifests",
        [
            ("first", (103.8460, 1.2860, 103.8580, 1.2990)),
            ("second", (103.8500, 1.2900, 103.8640, 1.3040)),
        ],
    )

    with pytest.raises(ValueError, match="overlap"):
        build_corpus(load_corpus_config(config_path))
