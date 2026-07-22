from urban_dataset.audit import audit_dataset, create_preview_atlas
from urban_dataset.demo import run_demo


def test_audit_and_atlas(tmp_path):
    output = tmp_path / "demo"
    run_demo(output)
    report = audit_dataset(output)
    assert report["tile_archives_found"] == 1
    assert report["invalid_tiles"] == []
    atlas = create_preview_atlas(output, tmp_path / "atlas.png")
    assert atlas.exists()
