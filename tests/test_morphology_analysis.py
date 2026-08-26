import numpy as np
import pandas as pd
import pytest

from urban_analysis.analyse import _control_frame, _pca
from urban_analysis.morphology import (
    MORPHOLOGY_CONTROL_FEATURES,
    MORPHOLOGY_FEATURES,
    describe_tile,
)


def _tile():
    height = width = 32
    layers = np.zeros((12, height, width), dtype=np.float32)
    valid = np.ones((height, width), dtype=np.uint8)

    layers[0, :4, :4] = 1.0
    layers[7, 4:8, 4:8] = 1.0
    layers[8, 10:14, 10:14] = 1.0
    layers[9, 10:14, 10:14] = 0.5

    roads = np.zeros((3, height, width), dtype=np.uint8)
    roads[0, 16, 8:25] = 1
    roads[0, 8:25, 16] = 1

    road_vertical = np.zeros((4, height, width), dtype=np.uint8)
    road_vertical[0] = roads.any(axis=0)

    rail_vertical = np.zeros((4, height, width), dtype=np.uint8)
    rail_vertical[0, 20, 5:28] = 1

    profiles = np.zeros((3, height, width), dtype=np.float32)
    confidence = np.zeros((3, height, width), dtype=np.uint8)

    arrays = {
        "layers": layers,
        "valid_data_mask": valid,
        "road_centerlines": roads,
        "road_vertical_masks": road_vertical,
        "rail_vertical_masks": rail_vertical,
        "road_vertical_profiles_m": profiles,
        "rail_vertical_profiles_m": profiles.copy(),
        "road_vertical_profile_confidence": confidence,
        "rail_vertical_profile_confidence": confidence.copy(),
    }
    row = {
        "tile_id": "test",
        "city_id": "singapore",
        "building_count": "3",
        "mean_building_height_m": "12.0",
    }
    metadata = {
        "tile_size_m": 128,
        "metres_per_pixel": 4,
        "height": {"normalization_max_m": 180},
    }
    return row, arrays, metadata


def test_descriptor_measures_network_structure_and_density():
    row, arrays, metadata = _tile()
    result = describe_tile(row, arrays, metadata)

    assert result["road_components"] == 1
    assert result["road_junction_count"] == 1
    assert result["road_dead_end_count"] == 4
    assert result["road_junctions_per_km"] > 0
    assert result["road_dead_ends_per_km"] > result["road_junctions_per_km"]
    assert result["road_largest_component_fraction"] == pytest.approx(1.0)

    assert result["rail_present"] == 1
    assert result["rail_components"] == 1
    assert result["rail_largest_component_fraction"] == pytest.approx(1.0)

    assert result["building_count"] == 3
    assert result["building_height_mean_m"] == pytest.approx(12.0)
    assert result["building_height_area_mean_m"] == pytest.approx(90.0)
    assert result["building_density_per_km2"] > 0


def test_dead_ends_on_tile_boundary_are_not_counted():
    row, arrays, metadata = _tile()
    arrays["road_centerlines"][:] = 0
    arrays["road_centerlines"][0, 16, :] = 1
    result = describe_tile(row, arrays, metadata)

    assert result["road_components"] == 1
    assert result["road_dead_end_count"] == 0
    assert result["road_dead_ends_per_km"] == pytest.approx(0.0)
    assert result["road_boundary_sides"] == 2


def test_control_features_do_not_duplicate_validity_or_height_metrics():
    assert "road_junctions_per_km" in MORPHOLOGY_FEATURES
    assert "road_dead_ends_per_km" in MORPHOLOGY_FEATURES
    assert "road_junctions_per_km" in MORPHOLOGY_CONTROL_FEATURES
    assert "road_dead_ends_per_km" in MORPHOLOGY_CONTROL_FEATURES

    assert "road_components" not in MORPHOLOGY_CONTROL_FEATURES
    assert "road_largest_component_fraction" not in MORPHOLOGY_CONTROL_FEATURES
    assert "rail_components" not in MORPHOLOGY_CONTROL_FEATURES
    assert "rail_largest_component_fraction" not in MORPHOLOGY_CONTROL_FEATURES

    height_controls = [
        name for name in MORPHOLOGY_CONTROL_FEATURES if name.startswith("building_height")
    ]
    assert height_controls == ["building_height_mean_m"]


def test_control_frame_treats_rail_mode_share_as_missing_without_rail():
    frame = pd.DataFrame(
        {
            "rail_present": [0, 1],
            "rail_length_km_per_km2": [0.0, 2.0],
            "rail_underground_share": [0.0, 0.75],
            "rail_elevated_share": [0.0, 0.25],
        }
    )

    control = _control_frame(
        frame,
        [
            "rail_length_km_per_km2",
            "rail_underground_share",
            "rail_elevated_share",
        ],
    )

    assert np.isnan(control.loc[0, "rail_underground_share"])
    assert np.isnan(control.loc[0, "rail_elevated_share"])
    assert control.loc[1, "rail_underground_share"] == pytest.approx(0.75)
    assert control.loc[1, "rail_elevated_share"] == pytest.approx(0.25)


def test_pca_drops_constant_features_and_reports_variance():
    frame = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0],
            "b": [0.0, 2.0, 4.0],
            "constant": [1.0, 1.0, 1.0],
        }
    )

    scores, loadings, details = _pca(frame, ["a", "b", "constant"], 2)

    assert list(scores.columns) == ["PC1", "PC2"]
    assert set(loadings.index) == {"a", "b"}
    assert details["features"] == ["a", "b"]
    assert details["explained_variance_ratio"][0] == pytest.approx(1.0)
