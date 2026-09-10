from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from urban_model.vectorize import _connected_components, _skeleton_paths


def _path_graph_components(paths):
    active = set()
    links = set()
    for path in paths:
        for pixel in path:
            active.add(tuple(pixel))
        for left, right in zip(path[:-1], path[1:], strict=True):
            links.add((tuple(left), tuple(right)))
            links.add((tuple(right), tuple(left)))

    remaining = set(active)
    components = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component = {start}
        remaining.remove(start)
        while stack:
            current = stack.pop()
            for left, right in links:
                if left == current and right in remaining:
                    remaining.remove(right)
                    component.add(right)
                    stack.append(right)
        components.append(component)
    return components


def test_short_junction_branch_survives_component_noise_filter():
    mask = np.zeros((15, 15), dtype=bool)
    mask[7, 2:13] = True
    mask[4:8, 7] = True

    paths, _cleaned = _skeleton_paths(mask, minimum_pixels=4)

    # The whole T junction is one legitimate component. The short upper branch
    # is only three skeleton pixels after thinning, so branch-level filtering at
    # four pixels would incorrectly delete it.
    assert len(_path_graph_components(paths)) == 1
    assert any(tuple(path[0]) == (4, 7) or tuple(path[-1]) == (4, 7) for path in paths)


def test_small_isolated_noise_component_is_still_removed():
    mask = np.zeros((20, 20), dtype=bool)
    mask[10, 2:16] = True
    mask[2, 2] = True
    mask[2, 3] = True

    paths, _cleaned = _skeleton_paths(mask, minimum_pixels=4)
    active = {tuple(pixel) for path in paths for pixel in path}

    assert all(row >= 8 for row, _column in active)
    assert len(_path_graph_components(paths)) == 1


def test_parallel_roads_are_not_joined_by_short_branch_fix():
    mask = np.zeros((24, 24), dtype=bool)
    mask[5:19, 7] = True
    mask[5:19, 15] = True

    paths, cleaned = _skeleton_paths(mask, minimum_pixels=4)
    active = {tuple(value) for value in np.argwhere(cleaned)}

    # The repair only changes branch retention inside an existing skeleton
    # component; it must not add a proximity-based bridge between roads.
    assert len(_connected_components(active)) == 2
    assert len(_path_graph_components(paths)) == 2
