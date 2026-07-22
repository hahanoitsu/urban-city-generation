from urban_dataset.tile import iter_tile_specs


def test_grid_is_globally_anchored_and_reproducible():
    first = list(iter_tile_specs("city", (1050, 2050, 4200, 5200), 1024, 1024, include_partial_tiles=False))
    expanded = list(iter_tile_specs("city", (1000, 2000, 4300, 5300), 1024, 1024, include_partial_tiles=False))
    first_ids = {tile.tile_id for tile in first}
    expanded_ids = {tile.tile_id for tile in expanded}
    assert first_ids <= expanded_ids
    assert all(tile.minx % 1024 == 0 and tile.miny % 1024 == 0 for tile in expanded)
