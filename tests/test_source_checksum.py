import pytest

from urban_dataset.config import (
    BuildConfig,
    HeightConfig,
    InputConfig,
    OutputConfig,
    ProjectConfig,
    QualityConfig,
    RasterConfig,
    RoadConfig,
)
from urban_dataset.demo import create_demo_layers
from urban_dataset.pipeline import run_build


def test_source_checksum_is_verified_before_output_is_created(tmp_path):
    pbf = tmp_path / "source.osm.pbf"
    pbf.write_bytes(b"not-the-recorded-source")
    output = tmp_path / "output"
    config = BuildConfig(
        project=ProjectConfig(city_id="demo", source_snapshot="sha256:" + "0" * 64),
        input=InputConfig(
            pbf_path=pbf,
            bbox_wgs84=(103.846, 1.286, 103.864, 1.304),
        ),
        output=OutputConfig(root=output),
        raster=RasterConfig(),
        roads=RoadConfig(),
        heights=HeightConfig(),
        quality=QualityConfig(minimum_buildings=1, minimum_road_length_m=1),
    )
    with pytest.raises(ValueError, match="checksum does not match"):
        run_build(config, extracted_layers=create_demo_layers())
    assert not output.exists()
