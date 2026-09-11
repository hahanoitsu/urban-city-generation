from __future__ import annotations

from urban_ai.generate import _segment_is_clear


def test_new_segment_can_leave_existing_junction():
    positions = [(5, 5), (10, 5)]
    edges = {(0, 1)}

    assert _segment_is_clear(0, None, (5, 10), positions, edges)


def test_new_segment_cannot_cross_nonincident_road():
    positions = [(0, 5), (10, 5), (5, 0)]
    edges = {(0, 1)}

    assert not _segment_is_clear(2, None, (5, 10), positions, edges)


def test_new_segment_cannot_overlap_existing_incident_road():
    positions = [(5, 5), (10, 5)]
    edges = {(0, 1)}

    assert not _segment_is_clear(0, None, (8, 5), positions, edges)
