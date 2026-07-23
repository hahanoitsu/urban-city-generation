import numpy as np

from urban_model.semantic_topology import (
    mask_topology_metrics,
    summarize_semantic_samples,
)


def test_connected_corridor_scores_as_one_component():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[6, 2:10] = 1

    metrics = mask_topology_metrics(mask)

    assert metrics["component_count"] == 1
    assert metrics["largest_component_fraction"] == 1.0
    assert metrics["small_component_pixel_fraction"] == 0.0


def test_fragmented_pixels_score_as_small_components():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[1, 1] = 1
    mask[5, 5] = 1
    mask[10, 10] = 1

    metrics = mask_topology_metrics(mask)

    assert metrics["component_count"] == 3
    assert metrics["largest_component_fraction"] == 1 / 3
    assert metrics["small_component_pixel_fraction"] == 1.0


def test_semantic_summary_reports_road_and_rail_topology():
    samples = np.zeros((2, 8, 8), dtype=np.uint8)
    samples[0, 4, 1:7] = 4
    samples[1, 1, 1] = 4
    samples[1, 3, 3] = 4
    samples[1, 6, 6] = 4
    samples[:, 2, :] = 3

    summary = summarize_semantic_samples(samples)

    assert summary["samples"] == 2
    assert summary["coverage"]["rail"]["max"] > summary["coverage"]["rail"]["min"]
    assert summary["topology"]["rail"]["component_count"]["max"] == 3
    assert summary["topology"]["road"]["largest_component_fraction"]["min"] == 1.0
