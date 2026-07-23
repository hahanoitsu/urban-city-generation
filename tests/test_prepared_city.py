from urban_dataset.demo import run_demo
from urban_dataset.prepared import load_city_gpkg


def test_extracted_geopackage_can_be_reopened(tmp_path):
    output = tmp_path / "demo"
    run_demo(output)

    layers, metadata = load_city_gpkg(output / "extracted_layers.gpkg")

    assert metadata["city_id"] == "demo_singapore"
    assert metadata["metric_crs"].startswith("EPSG:")
    assert not layers.roads.empty
    assert not layers.buildings.empty
    assert "estimated_width_m" in layers.roads.columns
    assert "estimated_height_m" in layers.buildings.columns
