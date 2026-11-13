import math

import numpy as np
import pytest
import torch
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon, box

from urban_dataset.extract import CityLayers
from urban_dataset.prepared import save_city_gpkg
from urban_layers.audit import audit_city
from urban_layers.buildings import BuildingDataset, BuildingModel
from urban_layers.city import generate_city, snap_lines
from urban_layers.data import CHANNELS, LINE_CLASSES, TRUNK, build_city, line_class, load_cities, pool_plan
from urban_layers.decoder import TokenDecoder, sequence_loss
from urban_layers.plan import PlanCrops, PlanNet, plan_loss, plan_statistics, urban_cells
from urban_layers.streets import D4, StreetDataset, StreetModel, collate, condition_arrays, find_ports, guide_crossings, move_grid, move_points, order_lines, port_key
from urban_layers.tokens import BuildingTokens, StreetGrammar, StreetTokens


def small_city(tmp_path):
    crs = "EPSG:3857"
    roads = []
    for index, value in enumerate(np.arange(60.0, 1536.0, 140.0)):
        kind = "major" if index == 5 else "local"
        roads.append({"road_class": kind, "vertical_mode": "surface", "estimated_width_m": 8.0, "geometry": LineString([(0, value), (1536, value)])})
        roads.append({"road_class": kind, "vertical_mode": "surface", "estimated_width_m": 8.0, "geometry": LineString([(value, 0), (value, 1536)])})
    buildings = []
    for x in np.arange(80.0, 1500.0, 35.0):
        for y in np.arange(80.0, 1500.0, 35.0):
            if (x - 60) % 140 > 20 and (y - 60) % 140 > 20:
                buildings.append({"building": "apartments", "estimated_height_m": 30.0, "geometry": box(x, y, x + 15, y + 12)})
    frame = lambda rows: gpd.GeoDataFrame(rows, geometry="geometry", crs=crs) if rows else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    layers = CityLayers(
        roads=frame(roads),
        buildings=frame(buildings),
        landuse=frame([{"landuse_class": "residential", "geometry": box(0, 0, 1536, 1536)}]),
        landuse_known=frame([]),
        water=frame([{"geometry": box(700, 1300, 1000, 1400)}]),
        green=frame([]),
        rail=frame([{"railway": "subway", "vertical_mode": "elevated", "geometry": LineString([(0, 500), (1536, 520)])}]),
        stations=frame(
            [
                {"station_kind": "subway", "geometry": Point(300, 504)},
                {"station_kind": "subway", "geometry": Point(320, 504)},
                {"station_kind": "subway", "geometry": Point(900, 512)},
                {"station_kind": "tram", "geometry": Point(600, 700)},
            ]
        ),
    )
    path = tmp_path / "small.gpkg"
    save_city_gpkg(layers, path, {"city_id": "small"}, overwrite=True)
    return path


def test_build_city_layers(tmp_path):
    summary = build_city(small_city(tmp_path), "small", tmp_path / "layers", stride=256.0)
    assert summary["tiles"] > 10
    assert summary["blocks"] > 50
    assert summary["buildings_in_blocks"] > 0.8 * summary["buildings"]
    cities = load_cities(tmp_path / "layers")
    plan = cities[0]["plan"]
    assert plan.shape[0] == len(CHANNELS)
    assert plan[CHANNELS.index("local")].max() > 0
    assert plan[CHANNELS.index("water")].max() > 0.5


def test_stations_are_saved_with_the_city(tmp_path):
    summary = build_city(small_city(tmp_path), "small", tmp_path / "layers", stride=256.0)
    assert summary["stations"]["subway"] == 2 and summary["stations"]["tram"] == 1 and summary["stations"]["rail"] == 0
    trunk = torch.load(tmp_path / "layers" / "small" / "trunk.pt", weights_only=False)
    assert sum(len(stops) for tile in trunk for stops in tile["stops"]) == 2
    stations = load_cities(tmp_path / "layers")[0]["stations"]
    assert stations.shape == (3, 3)
    assert sorted(stations[:, 0].tolist()) == [300.0, 600.0, 900.0]


def test_street_tokens_round_trip_and_grammar():
    tokens = StreetTokens()
    lines = [
        (line_class("road", "major", "surface"), np.array([[0, 10], [256, 10.4], [512, 40]])),
        (line_class("rail", "subway", "elevated"), np.array([[3, 3], [300, 400]])),
        (line_class("road", "local", "surface"), np.array([[100, 10], [100, 200]]), (True, True)),
    ]
    back = tokens.decode(tokens.encode(lines))
    assert [item[0] for item in back] == [item[0] for item in lines]
    assert back[0][1].tolist() == [[0, 10], [256, 10], [512, 40]]
    assert [item[2] for item in back] == [(False, False), (False, False), (True, True)]
    grammar = StreetGrammar(tokens, max_points=3, max_lines=3)
    state = grammar.start(1, "cpu")
    sequence = tokens.encode(lines)[1:]
    for token in sequence:
        assert bool(grammar.allowed(state)[0, token])
        state = grammar.advance(state, torch.tensor([token]))
    assert bool(grammar.done(state)[0])


def test_station_marks_survive_the_tokens_and_the_grammar():
    tokens = StreetTokens(stations=True)
    subway = line_class("rail", "subway", "elevated")
    major = line_class("road", "major", "surface")
    lines = [
        (subway, np.array([[0, 10], [100, 10], [100.2, 10.1], [300, 40]]), (False, False), [1, 3]),
        (major, np.array([[5, 5], [50, 80]]), (True, False), []),
    ]
    encoded = tokens.encode(lines)
    back = tokens.decode(encoded)
    assert back[0][3] == (1, 2) and back[1][3] == ()
    assert [len(item[1]) for item in back] == [3, 2]
    grammar = StreetGrammar(tokens, max_points=8, max_lines=8)
    state = grammar.start(1, "cpu")
    for token in encoded[1:]:
        allowed = grammar.allowed(state)[0]
        assert bool(allowed[token])
        if tokens.kinds[token] == 6 and not bool(state["rail"][0]):
            assert not bool(allowed[tokens.station])
        state = grammar.advance(state, torch.tensor([token]))
    assert bool(grammar.done(state)[0])
    assert StreetTokens().vocab == tokens.vocab - 1 and len(StreetTokens().decode(StreetTokens().encode(lines))[0]) == 3


def test_main_road_corridors_leave_out_ramps_and_merge_twin_carriageways():
    from urban_layers.data import city_lines, guide_lines, street_tiles

    crs = "EPSG:3857"
    road = lambda kind, tag, points, oneway=None: {"road_class": kind, "highway": tag, "oneway": oneway, "vertical_mode": "surface", "estimated_width_m": 8.0, "geometry": LineString(points)}
    roads = [
        road("major", "primary", [(0, 300), (1100, 300)], "yes"),
        road("major", "primary", [(1100, 300), (2000, 300)], "yes"),
        road("major", "primary", [(2000, 316), (900, 316)], "yes"),
        road("major", "primary", [(900, 316), (0, 316)], "yes"),
        road("major", "primary_link", [(900, 316), (950, 400), (1000, 480)]),
        road("major", "trunk", [(1000, 0), (1000, 2000)]),
        road("major", "primary", [(500, 316), (500, 700)]),
        road("major", "primary", [(1900, 1700), (100, 1700)], "yes"),
        road("local", "residential", [(0, 800), (2000, 800)]),
    ]
    frame = lambda rows: gpd.GeoDataFrame(rows, geometry="geometry", crs=crs) if rows else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    rail = [{"railway": "subway", "vertical_mode": "elevated", "geometry": LineString([(0, 1200 + shift), (2000, 1200 + shift)])} for shift in (0.0, 4.0)]
    layers = CityLayers(roads=frame(roads), buildings=frame([]), landuse=frame([]), landuse_known=frame([]), water=frame([]), green=frame([]), rail=frame(rail))
    lines = city_lines(layers)
    guide = guide_lines(layers, lines)
    main = [geometry for class_id, geometry, _width in guide if LINE_CLASSES[class_id][0] == "road"]
    assert sorted(round(geometry.length) for geometry in main) == [400, 900, 1100, 1800, 2000]
    assert next(geometry for geometry in main if round(geometry.length) == 400).coords[0] == (500.0, 300.0)
    assert len([1 for class_id, _geometry, _width in guide if LINE_CLASSES[class_id][0] == "rail"]) == 1
    tiles = street_tiles([line for line in lines if LINE_CLASSES[line[0]][0] == "road"], (0.0, 0.0, 2048.0, 2048.0), guide=guide)
    tile = next(tile for tile in tiles if tile["origin"] == (512.0, 0.0))
    assert {LINE_CLASSES[class_id][1] for class_id, _points in tile["targets"]} == {"major"}
    assert len(tile["targets"]) == 5 and len(tile["corridors"]) == 4
    data = StreetDataset([(0, tile)], [np.zeros((len(CHANNELS), 16, 16), dtype=np.float32)], 1, np.ones(len(CHANNELS), dtype=np.float32), augment=False, max_len=256, exits=True)
    item = data[0]
    assert item["guide_mask"].sum() > 20 and item["guide_xy"][item["guide_mask"]].min() >= -0.26
    model = StreetModel(1, d=32, layers=1, heads=4, max_len=256, exits=True, guided=True)
    batch = collate([item, item])
    loss = sequence_loss(model(batch), batch["tokens"])
    loss.backward()
    assert torch.isfinite(loss) and model.condition(batch)[0].shape[1] == StreetModel(1, d=32, layers=1, heads=4, max_len=256).condition(batch)[0].shape[1] + 192

    done = {(-1, 0), (-1, -1), (0, -1), (1, -1)}
    crossings = guide_crossings(tile["corridors"], done)
    assert sorted((x, y) for x, y, *_rest in crossings) == [(488.0, 512.0), (512.0, 300.0)]
    assert all(hx < -0.9 if x == 512.0 else hy < -0.9 for x, y, hx, hy, _class_id in crossings)
    halves = [(crossings[0][4], np.array([[300.0, 300.0], [512.0, 300.0]])), (crossings[0][4], np.array([[512.0, 300.0], [600.0, 300.0]]))]
    assert len(guide_crossings(halves, done)) == 1
    plain = find_ports(tile["targets"], done)
    pinned = find_ports(tile["targets"], done, tile["corridors"])
    assert len(plain) == 3 and sorted(set(pinned) - set(plain)) == [(488.0, 512.0, 0.0, -1.0, crossings[0][4]), (512.0, 300.0, -1.0, 0.0, crossings[0][4])]
    held = StreetDataset([(0, tile)], [np.zeros((len(CHANNELS), 16, 16), dtype=np.float32)], 1, np.ones(len(CHANNELS), dtype=np.float32), augment=False, max_len=256, exits=True, pinned=True)
    leave = lambda tokens: int((tokens == held.tokens.exit).sum())
    assert leave(item["tokens"]) == 0 and leave(held[0]["tokens"]) == 2 and int(held[0]["port_mask"].sum()) == 5
    seen = [np.ones((len(CHANNELS), 16, 16), dtype=np.float32)]
    blind = StreetDataset([(0, tile)], seen, 1, np.ones(len(CHANNELS), dtype=np.float32), augment=False, max_len=256, exits=True, hidden=("major", "rail"))[0]["plan"]
    assert blind[:, [CHANNELS.index("major"), CHANNELS.index("rail")]].max() == 0.0 and blind[:, CHANNELS.index("local")].max() == 1.0


def test_two_tracks_become_one_corridor_and_a_branch_stays_joined():
    from urban_layers.audit import road_pieces
    from urban_layers.data import rail_corridors

    over = line_class("rail", "subway", "elevated")
    under = line_class("rail", "subway", "underground")
    lines = [
        (over, LineString([(0, 100), (3000, 100)]), 0.0),
        (over, LineString([(0, 104), (1500, 104), (1800, 400), (3000, 400)]), 0.0),
        (over, LineString([(1000, -500), (1000, 700)]), 0.0),
        (under, LineString([(3000, 100), (4000, 100)]), 0.0),
        (under, LineString([(3000, 104), (4000, 104)]), 0.0),
        (over, LineString([(0, 1600), (2000, 2400)]), 0.0),
        (over, LineString([(0, 2000), (2000, 2000)]), 0.0),
        (line_class("road", "major", "surface"), LineString([(0, 96), (3000, 96)]), 0.0),
    ]
    kept = rail_corridors(lines)
    rail = [(class_id, geometry) for class_id, geometry, _width in kept if LINE_CLASSES[class_id][0] == "rail"]
    assert len(kept) - len(rail) == 1 and len(rail) == 6
    lengths = sorted(round(geometry.length) for _class_id, geometry in rail)
    assert lengths == [1000, 1200, 1504, 2000, 2154, 3121]
    branch = next(geometry for _class_id, geometry in rail if round(geometry.length) == 1504)
    assert branch.coords[0] == (1500.0, 104.0)
    _shapes, label = road_pieces([{"points": np.asarray(geometry.coords), "vertical": LINE_CLASSES[class_id][2]} for class_id, geometry in rail])
    assert len(set(label.tolist())) == 2


def test_some_empty_tiles_are_kept_so_the_model_learns_to_draw_nothing():
    from urban_layers.data import street_tiles

    local = line_class("road", "local", "surface")
    tiles = street_tiles([(local, LineString([(100, 100), (400, 300)]), 6.0)], (0.0, 0.0, 4096.0, 4096.0))
    empty = [tile for tile in tiles if not tile["targets"]]
    assert len(tiles) - len(empty) == 1 and len(empty) == 20
    data = StreetDataset([(0, tile) for tile in empty], [np.zeros((len(CHANNELS), 32, 32), dtype=np.float32)], 1, np.ones(len(CHANNELS), dtype=np.float32), max_len=64)
    assert data[0]["tokens"].tolist() == [1, 2]


def test_a_block_knows_the_roads_on_its_edge_and_the_lines_across_it():
    from urban_layers.buildings import block_arrays, flip
    from urban_layers.data import block_record, line_index, road_index

    lines = [
        (line_class("road", "major", "surface"), LineString([(-50, -3), (250, -3)]), 14.0),
        (line_class("road", "local", "surface"), LineString([(-3, -50), (-3, 150)]), 6.0),
        (line_class("road", "local", "surface"), LineString([(100, -3), (100, 60)]), 6.0),
        (line_class("rail", "subway", "elevated"), LineString([(-50, 80), (250, 80)]), 0.0),
        (line_class("rail", "subway", "underground"), LineString([(-50, 40), (250, 40)]), 0.0),
    ]
    block = box(0, 0, 200, 100)
    record = block_record(block, np.zeros((len(CHANNELS), 4, 4), dtype=np.float32), (0.0, 0.0, 512.0, 512.0), *road_index(lines), *line_index(lines))
    edge = record["edge"]
    assert edge.shape == (48, 5)
    assert np.all(edge[34:40, 0] == 1.0) and np.allclose(edge[34:40, 4], 0.1) and np.all(edge[41:47, 0] == 1.0)
    assert edge[40, 2] == 1.0 and edge[40, 4] == 0.0
    assert np.all(edge[26:31, 2] == 1.0) and np.allclose(edge[26:31, 4], 0.1)
    assert np.all(edge[10:23, 3] == 1.0) and np.allclose(edge[10:23, 4], 20.0 / 30.0)
    kinds = {LINE_CLASSES[int(value)][1:] for value in record["inner_class"][record["inner_mask"]]}
    assert kinds == {("local", "surface"), ("subway", "elevated")}
    arrays = block_arrays({**record, "buildings": np.zeros((0, 7)), "landuse": 1}, np.ones(len(CHANNELS), dtype=np.float32), np.ones(1))
    assert arrays["edge"].shape == (48, 5) and arrays["inner_mask"].sum() == record["inner_mask"].sum()
    mirrored = flip({**record, "buildings": np.zeros((0, 7))}, True, False)
    assert np.allclose(mirrored["inner_xy"][:, 0], -record["inner_xy"][:, 0]) and np.allclose(mirrored["inner_heading"][:, 1], record["inner_heading"][:, 1])
    model = BuildingModel(1, d=32, layers=1, heads=4, max_len=64, roads=True)
    batch = {name: torch.from_numpy(np.stack([value, value])) for name, value in arrays.items()}
    batch["tokens"] = torch.tensor([[1, 4, 2], [1, 4, 2]])
    loss = sequence_loss(model(batch), batch["tokens"])
    loss.backward()
    assert torch.isfinite(loss) and model.edge.weight.grad is not None
    old = BuildingModel(1, d=32, layers=1, heads=4, max_len=64)
    assert old.condition(batch)[0].shape[1] == 2 + 48 and model.condition(batch)[0].shape[1] == 2 + 48 + 48


def test_building_angles_are_kept_relative_to_the_nearest_road():
    from urban_layers.buildings import place_buildings
    from urban_layers.data import block_frame, block_records, city_lines, road_segments

    crs = "EPSG:3857"
    frame = lambda rows: gpd.GeoDataFrame(rows, geometry="geometry", crs=crs) if rows else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    road = lambda points: {"road_class": "local", "vertical_mode": "surface", "estimated_width_m": 6.0, "geometry": LineString(points)}
    roads = [road([(0, 0), (200, 0)]), road([(200, 0), (200, 100)]), road([(200, 100), (0, 100)]), road([(0, 100), (0, 0)])]
    homes = [box(170, 30, 182, 70), box(40, 10, 80, 22), Polygon([(100, 40), (130, 57), (125, 66), (95, 49)])]
    buildings = [{"building": "house", "estimated_height_m": 9.0, "geometry": home} for home in homes]
    layers = CityLayers(roads=frame(roads), buildings=frame(buildings), landuse=frame([]), landuse_known=frame([]), water=frame([]), green=frame([]), rail=frame([]))
    lines = city_lines(layers)
    records, assigned, _total = block_records(layers, lines, np.zeros((len(CHANNELS), 2, 2), dtype=np.float32), (0.0, 0.0, 256.0, 256.0))
    assert assigned == 3
    record = next(record for record in records if len(record["buildings"]) == 3)
    turns = sorted(abs(float(turn)) for turn in record["buildings"][:, 5])
    assert turns[0] < 0.02 and turns[1] < 0.02 and abs(turns[2] - math.atan2(17, 30)) < 0.03
    block = box(3, 3, 197, 97)
    centre, angle = block_frame(block)
    placed = place_buildings(block, centre, angle, [tuple(row) for row in record["buildings"]], nudge=0.0, roads=road_segments(lines), face=False, relative=True)
    assert len(placed) == 3
    for item, home in zip(sorted(placed, key=lambda item: Polygon(item["footprint"]).centroid.x), sorted(homes, key=lambda home: home.centroid.x)):
        assert Polygon(item["footprint"]).intersection(home.minimum_rotated_rectangle).area > 0.9 * home.area


def test_building_tokens_round_trip():
    tokens = BuildingTokens()
    landuse, items = tokens.decode(tokens.encode(2, [(1, 10.2, -40.7, 25.0, 12.0, 0.4, 33.0)]))
    assert landuse == 2
    kind, x, y, width, depth, turn, height = items[0]
    assert kind == 1 and abs(x - 10.2) < 1 and abs(y + 40.7) < 1 and width == 25 and depth == 12 and abs(turn - 0.4) < 0.06 and height == 33


def test_decoder_cache_matches_full_forward():
    torch.manual_seed(0)
    tokens = StreetTokens()
    decoder = TokenDecoder(tokens.vocab, tokens.features(), d=32, layers=2, heads=4, max_len=32).eval()
    memory = torch.randn(2, 5, 32)
    padding = torch.zeros(2, 5, dtype=torch.bool)
    padding[1, 3:] = True
    sequence = torch.randint(4, tokens.vocab, (2, 10))
    full = decoder(sequence, memory, padding)
    caches = [{"self": {}, "cross": {}} for _ in decoder.layers]
    steps = []
    for position in range(sequence.shape[1]):
        x = decoder.embed(sequence[:, position : position + 1], position)
        for layer, cache in zip(decoder.layers, caches):
            x = layer(x, memory, (~padding)[:, None, None, :], cache)
        steps.append(decoder.head(decoder.norm(x)))
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-4)


def test_rotations_keep_plan_and_points_together():
    grid = np.zeros((1, 8, 8), dtype=np.float32)
    grid[0, 2, 6] = 1.0
    point = np.array([[(6 - 2 + 0.5) * 128.0, (2 - 2 + 0.5) * 128.0]])
    for matrix in D4:
        moved = move_points(point, matrix)[0]
        turned = move_grid(grid, matrix)
        row, column = np.argwhere(turned[0] == 1.0)[0]
        assert np.allclose(moved, [(column - 2 + 0.5) * 128.0, (row - 2 + 0.5) * 128.0])


def test_ports_start_lines_on_visible_sides():
    major = line_class("road", "major", "surface")
    lines = [(major, np.array([[300.0, 200.0], [0.0, 200.0]])), (major, np.array([[100.0, 512.0], [100.0, 300.0]]))]
    ports = find_ports(lines, {(-1, 0)})
    assert len(ports) == 1 and ports[0][0] == 0.0
    ordered = order_lines(lines, {(-1, 0)})
    assert ordered[0][1][0].tolist() == [0.0, 200.0]


def test_lines_grow_from_what_is_drawn_and_mark_joins():
    local = line_class("road", "local", "surface")
    major = line_class("road", "major", "surface")
    lines = [
        (local, np.array([[200.0, 300.0], [200.0, 100.0]])),
        (major, np.array([[512.0, 100.0], [100.0, 100.0]])),
        (local, np.array([[260.0, 250.0], [200.0, 250.0]])),
        (local, np.array([[0.0, 400.0], [200.0, 300.0]])),
    ]
    ordered = order_lines(lines, {(-1, 0)})
    assert [item[1][0].tolist() for item in ordered] == [[0.0, 400.0], [100.0, 100.0], [200.0, 300.0], [200.0, 250.0]]
    assert [item[2] for item in ordered] == [(False, True), (False, False), (True, True), (True, False)]


def test_snap_moves_joined_ends_and_keeps_chains_together():
    local = line_class("road", "local", "surface")
    lines = [
        (local, np.array([[0.0, 0.0], [100.0, 0.0]])),
        (local, np.array([[50.0, 20.0], [50.0, 80.0]]), (True, False)),
        (local, np.array([[80.0, 10.0], [80.0, 60.0]])),
        (local, np.array([[50.0, 81.0], [10.0, 81.0]])),
    ]
    snapped = snap_lines(lines)
    assert snapped[1][1][0].tolist() == [50.0, 0.0]
    assert [50.0, 0.0] in snapped[0][1].tolist()
    assert snapped[2][1][0].tolist() == [80.0, 10.0]
    assert snapped[1][1][-1].tolist() == snapped[3][1][0].tolist()
    chained = snap_lines(
        [
            (local, np.array([[0.0, 0.0], [0.0, -50.0]])),
            (local, np.array([[10.0, 0.0], [10.0, 50.0]]), (True, False)),
            (local, np.array([[20.5, 0.0], [40.0, 30.0]]), (True, False)),
        ]
    )
    assert chained[1][1][0].tolist() == [0.0, 0.0] and chained[2][1][0].tolist() == [0.0, 0.0]


def test_models_train_and_generate_a_city(tmp_path):
    torch.manual_seed(0)
    build_city(small_city(tmp_path), "small", tmp_path / "layers", stride=256.0)
    cities = load_cities(tmp_path / "layers")
    plans = [city["plan"] for city in cities]
    mean, std, control_scale = plan_statistics(plans)
    scale = 1.0 / std

    plan_model = PlanNet(1, width=16)
    keep = [urban_cells(plan) for plan in plans]
    assert keep[0].any()
    crops = PlanCrops(plans, mean, std, control_scale, size=8, length=4, keep=keep)
    loss = plan_loss(plan_model, torch.utils.data.default_collate([crops[index] for index in range(4)]))
    loss.backward()
    assert torch.isfinite(loss)

    tiles = [(0, tile) for tile in torch.load(cities[0]["folder"] / "streets.pt", weights_only=False)]
    street_model = StreetModel(1, d=32, layers=1, heads=4, max_len=512)
    batch = collate([StreetDataset(tiles, plans, 1, scale, max_len=512)[index] for index in range(4)])
    loss = sequence_loss(street_model(batch), batch["tokens"])
    loss.backward()
    assert torch.isfinite(loss)

    blocks = [(0, block) for block in torch.load(cities[0]["folder"] / "blocks.pt", weights_only=False)]
    building_model = BuildingModel(1, d=32, layers=1, heads=4, max_len=256)
    batch = collate([BuildingDataset(blocks, 1, scale, max_len=256)[index] for index in range(4)])
    loss = sequence_loss(building_model(batch), batch["tokens"])
    loss.backward()
    assert torch.isfinite(loss)

    checkpoints = {
        "plan": (plan_model.eval(), {"cities": ["small"], "mean": mean, "std": std, "control_scale": control_scale, "city_controls": [plans[0].mean(axis=(1, 2))]}),
        "streets": (street_model.eval(), {"cities": ["small"], "scale": scale}),
        "buildings": (building_model.eval(), {"cities": ["small"], "scale": scale}),
    }
    city, plan = generate_city(checkpoints, size_m=(512.0, 512.0), style={"small": 1.0}, sliders={"roads": 1.5}, plan_steps=2, log=lambda _text: None)
    assert city["format"] == "urban-layers-city"
    assert plan.shape[0] == len(CHANNELS)
    assert all(LINE_CLASSES[line_class("road", road["class"], road["vertical"])][0] == "road" for road in city["roads"])
    assert all(item["tracks"] == 2 for item in city["rail"])
    report = audit_city(city, plan)
    assert {"network", "follows_plan", "rail", "cover"} <= set(report)
    from urban_layers.check import check_city, failures

    assert failures(check_city(city, plan)) == []
    assert {"rail", "stations", "road joins"} <= set(city["rules"])
    bare, plan = generate_city(checkpoints, size_m=(512.0, 512.0), style={"small": 1.0}, plan_steps=2, transport_rules=False, log=lambda _text: None)
    assert bare["rules"] == ["plan totals", "natural cover", "building placement"] and bare["stations"] == []
    assert [item for item in failures(check_city(bare, plan)) if not item.startswith("plan_orientation")] == []


def test_rail_stations_and_main_roads_can_come_from_a_model(tmp_path):
    torch.manual_seed(0)
    build_city(small_city(tmp_path), "small", tmp_path / "layers", stride=256.0)
    cities = load_cities(tmp_path / "layers")
    plans = [city["plan"] for city in cities]
    mean, std, control_scale = plan_statistics(plans)
    scale = 1.0 / std
    coarse = [pool_plan(plan, TRUNK) for plan in plans]
    coarse_scale = 1.0 / plan_statistics(coarse)[1]

    trunk = [(0, tile) for tile in torch.load(cities[0]["folder"] / "trunk.pt", weights_only=False)]
    assert sum(len(stops) for _city, tile in trunk for stops in tile["stops"]) == 2
    assert all(LINE_CLASSES[class_id][1] in ("major", "subway") for _city, tile in trunk for class_id, _points in tile["targets"])
    trunk_model = StreetModel(1, d=32, layers=1, heads=4, max_len=512, stations=True, exits=True)
    data = StreetDataset(trunk, coarse, 1, coarse_scale, max_len=512, stations=True, plan_cells=16, exits=True)
    batch = collate([data[0], data[0]])
    assert int((batch["tokens"] == trunk_model.tokens.station).sum()) == 4
    loss = sequence_loss(trunk_model(batch), batch["tokens"])
    loss.backward()
    assert torch.isfinite(loss)

    tiles = [(0, tile) for tile in torch.load(cities[0]["folder"] / "streets.pt", weights_only=False)]
    assert all(LINE_CLASSES[class_id][0] == "road" for _city, tile in tiles for class_id, _points in tile["targets"])
    assert sum(len(tile["corridors"]) for _city, tile in tiles) > 0
    street_model = StreetModel(1, d=32, layers=1, heads=4, max_len=512, exits=True, guided=True)
    batch = collate([StreetDataset(tiles, plans, 1, scale, max_len=512, exits=True)[index] for index in range(4)])
    assert bool(batch["guide_mask"].any())
    loss = sequence_loss(street_model(batch), batch["tokens"])
    loss.backward()
    assert torch.isfinite(loss)

    plan_model = PlanNet(1, width=16)
    blocks = [(0, block) for block in torch.load(cities[0]["folder"] / "blocks.pt", weights_only=False)]
    assert sum(float(block["edge"][:, :4].sum()) for _city, block in blocks) > 0
    building_model = BuildingModel(1, d=32, layers=1, heads=4, max_len=256, roads=True)
    batch = collate([BuildingDataset(blocks, 1, scale, max_len=256)[index] for index in range(4)])
    loss = sequence_loss(building_model(batch), batch["tokens"])
    loss.backward()
    assert torch.isfinite(loss)
    checkpoints = {
        "plan": (plan_model.eval(), {"cities": ["small"], "mean": mean, "std": std, "control_scale": control_scale, "city_controls": [plans[0].mean(axis=(1, 2))]}),
        "trunk": (trunk_model.eval(), {"cities": ["small"], "scale": coarse_scale, "unit": TRUNK, "plan_cells": 16}),
        "streets": (street_model.eval(), {"cities": ["small"], "scale": scale}),
        "buildings": (building_model.eval(), {"cities": ["small"], "scale": scale}),
    }
    city, plan = generate_city(checkpoints, size_m=(1024.0, 1024.0), style={"small": 1.0}, plan_steps=2, seed=3, log=lambda _text: None)
    assert city["rules"] == ["plan totals", "natural cover", "building clearance", "road and rail repairs"]
    assert all(LINE_CLASSES[line_class("rail", item["class"], item["vertical"])][0] == "rail" for item in city["rail"])
    from urban_layers.check import check_city, failures

    assert [item for item in failures(check_city(city, plan)) if not item.startswith("plan_orientation")] == []


def test_a_corridor_is_handed_from_tile_to_tile():
    from urban_layers.city import generate_streets

    torch.manual_seed(0)
    major = line_class("road", "major", "surface")
    model = StreetModel(1, d=32, layers=1, heads=4, max_len=512, exits=True, guided=True).eval()
    plan = np.zeros((len(CHANNELS), 8, 8), dtype=np.float32)
    checkpoint = {"scale": np.ones(len(CHANNELS), dtype=np.float32)}
    given = [(major, np.array([[0.0, 300.0], [1024.0, 700.0]]))]
    lines = generate_streets(model, checkpoint, plan, np.ones(1, dtype=np.float32), torch.Generator().manual_seed(0), given=given, pinned=True, log=lambda _text: None)
    ends = np.asarray([points[slot] for class_id, points, _joins in lines if class_id == major for slot in (0, -1)]).reshape(-1, 2)
    for x, y in ((0.0, 300.0), (512.0, 500.0), (542.72, 512.0), (1024.0, 700.0)):
        assert np.hypot(ends[:, 0] - x, ends[:, 1] - y).min() < 1.0
    assert (np.hypot(ends[:, 0] - 512.0, ends[:, 1] - 500.0) < 1.0).sum() >= 2


@pytest.mark.parametrize("max_len", [40, 512])
def test_every_port_gets_a_line_even_from_an_untrained_model(max_len):
    torch.manual_seed(0)
    model = StreetModel(1, d=32, layers=1, heads=4, max_len=max_len).eval()
    local = line_class("road", "local", "surface")
    ports = [(0.0, 100.0, 1.0, 0.0, local), (250.0, 0.0, 0.0, 1.0, local), (0.0, 400.0, 1.0, 0.0, local)]
    from urban_layers.streets import condition_arrays

    item = condition_arrays(np.zeros((len(CHANNELS), 8, 8), dtype=np.float32), [], sorted(ports, key=port_key), np.ones(1, dtype=np.float32))
    batch = collate([item, item])
    for lines in model.generate(batch, generator=torch.Generator().manual_seed(1), max_lines=8):
        ends = np.asarray([points[index] for _class, points, _joins in lines for index in (0, -1)])
        for x, y, *_rest in ports:
            assert np.abs(ends - [x, y]).max(axis=1).min() <= 8


def test_free_lines_do_not_start_or_end_on_a_done_side():
    torch.manual_seed(0)
    model = StreetModel(1, d=32, layers=1, heads=4, max_len=512).eval()
    from urban_layers.streets import condition_arrays

    item = condition_arrays(np.zeros((len(CHANNELS), 8, 8), dtype=np.float32), [], [], np.ones(1, dtype=np.float32))
    batch = collate([item, item])
    batch["sides"] = torch.tensor([[True, True, True, True], [True, False, False, True]])
    results = model.generate(batch, generator=torch.Generator().manual_seed(2), max_points=20, max_lines=6)
    assert sum(len(lines) for lines in results) > 0
    for lines, sides in zip(results, batch["sides"].tolist()):
        for _class_id, points, joins in lines:
            ends = ([points[0]] if not joins[0] else []) + ([points[-1]] if len(points) < 20 and not joins[1] else [])
            for x, y in ends:
                near = [x <= 4, y <= 4, x >= 508, y >= 508]
                assert not any(side and close for side, close in zip(sides, near))


def test_land_use_follows_the_plan_not_a_grid():
    from urban_layers.city import landuse_blocks, landuse_regions
    from urban_layers.data import LANDUSE

    plan = np.zeros((len(CHANNELS), 16, 16), dtype=np.float32)
    plan[CHANNELS.index("residential"), :, :8] = 0.9
    plan[CHANNELS.index("industrial"), :, 8:] = 0.9
    regions = landuse_regions(plan)
    assert {LANDUSE[use] for _polygon, use in regions} == {"residential", "industrial"}
    local = line_class("road", "local", "surface")
    lines = [(local, LineString([(100, 100), (1900, 100)]), 0.0), (local, LineString([(100, 1900), (1900, 1900)]), 0.0)]
    pieces = landuse_blocks(lines, None, regions, box(0, 0, 2048, 2048).exterior)
    middle = box(100, 100, 1900, 1900)
    big = [(polygon, LANDUSE[use]) for polygon, use in pieces if polygon.intersection(middle).area > 100000]
    assert {"residential", "industrial"} <= {use for _polygon, use in big}
    used = sum(polygon.intersection(middle).area for polygon, use in big if use != "none")
    assert used > 0.8 * middle.area
    for polygon, use in big:
        if use != "none":
            assert (polygon.centroid.x < 1024) == (use == "residential")


def test_building_land_use_can_be_forced():
    torch.manual_seed(0)
    model = BuildingModel(1, d=32, layers=1, heads=4, max_len=64).eval()
    ring = np.stack([np.cos(np.linspace(0, 6.28, 48)), np.sin(np.linspace(0, 6.28, 48))], axis=1).astype(np.float32) * 50
    from urban_layers.buildings import block_arrays

    record = {"ring": ring, "plan": np.zeros(len(CHANNELS), dtype=np.float32), "frontage": np.zeros(3, dtype=np.float32), "area": 8000.0}
    item = block_arrays(record, np.ones(len(CHANNELS), dtype=np.float32), np.ones(1, dtype=np.float32))
    batch = {name: torch.from_numpy(np.stack([item[name]] * 3)) for name in item}
    results = model.generate(batch, generator=torch.Generator().manual_seed(0), landuse=torch.tensor([1, 3, 5]))
    assert [landuse for landuse, _items in results] == [1, 3, 5]


def test_pick_samples_matches_the_planned_total():
    from urban_layers.city import pick_samples

    areas = [np.array([0.0, 100.0, 400.0]), np.array([50.0, 200.0, 900.0]), np.array([0.0, 10.0, 20.0])]
    targets = np.array([300.0, 300.0, 10.0])
    picks = pick_samples(areas, targets, goal=620.0)
    assert sum(values[pick] for values, pick in zip(areas, picks)) == 620.0
    assert pick_samples(areas, targets, goal=0.0) == [0, 0, 0]
    assert pick_samples(areas, targets, goal=1e9) == [2, 2, 2]


def test_land_use_keeps_the_planned_proportions():
    from urban_layers.city import landuse_regions
    from urban_layers.data import LANDUSE

    plan = np.zeros((len(CHANNELS), 16, 16), dtype=np.float32)
    plan[CHANNELS.index("green")] = 0.5
    plan[CHANNELS.index("residential")] = np.linspace(0.1, 0.4, 16)[None, :]
    regions = landuse_regions(plan)
    total = 16 * 128.0 * 16 * 128.0
    residential = [polygon for polygon, use in regions if LANDUSE[use] == "residential"]
    share = sum(polygon.area for polygon in residential) / total
    assert 0.15 < share < 0.35
    assert np.mean([polygon.centroid.x for polygon in residential]) > 1024


def test_buildings_need_a_road_nearby():
    from urban_layers.city import reachable
    from urban_layers.data import road_index

    local = line_class("road", "local", "surface")
    tree, _classes = road_index([(local, LineString([(0, 0), (500, 0)]), 0.0)])
    near = {"footprint": [[100, 20], [110, 20], [110, 30], [100, 30]]}
    far = {"footprint": [[100, 300], [110, 300], [110, 310], [100, 310]]}
    assert reachable([near, far], tree) == [near]
    assert reachable([near, far], None) == [near, far]


def test_rail_runs_right_across_the_map():
    from urban_layers.rail import rail_corridors, rail_runs, rail_stations

    plan = np.zeros((len(CHANNELS), 32, 32), dtype=np.float32)
    plan[CHANNELS.index("rail"), 15:17, 12:20] = 1.0
    lines = rail_corridors(plan, (4096.0, 4096.0))
    assert len(lines) == 1
    assert LineString(lines[0]).length > 3500.0
    for x, y in (lines[0][0], lines[0][-1]):
        assert min(x, y, 4096.0 - x, 4096.0 - y) < 1.0
    runs = rail_runs(lines, plan)
    assert {LINE_CLASSES[class_id][:2] for class_id, _points, _number in runs} == {("rail", "subway")}
    stations = np.asarray([station["xy"] for station in rail_stations(lines, plan)])
    assert len(stations) >= 2
    assert np.linalg.norm(np.diff(stations, axis=0), axis=1).min() > 700.0
    assert rail_corridors(np.zeros_like(plan), (4096.0, 4096.0)) == []


def test_crossing_lines_share_one_interchange():
    from urban_layers.rail import rail_stations

    plan = np.zeros((len(CHANNELS), 32, 32), dtype=np.float32)
    lines = [np.array([[0.0, 2000.0], [4096.0, 2000.0]]), np.array([[2100.0, 0.0], [2100.0, 4096.0]])]
    stations = rail_stations(lines, plan)
    shared = [station for station in stations if len(station["lines"]) == 2]
    assert len(shared) == 1 and np.allclose(shared[0]["xy"], [2100.0, 2000.0])
    others = np.asarray([station["xy"] for station in stations if len(station["lines"]) == 1])
    assert np.linalg.norm(others - [2100.0, 2000.0], axis=1).min() >= 599.0
    for number in (0, 1):
        assert sum(number in station["lines"] for station in stations) >= 3


def test_street_grammar_can_leave_out_rail():
    tokens = StreetTokens()
    grammar = StreetGrammar(tokens, classes=[mode == "road" for mode, _name, _level in LINE_CLASSES])
    allowed = grammar.allowed(grammar.start(1, "cpu"))[0]
    for index, (mode, _name, _level) in enumerate(LINE_CLASSES):
        assert bool(allowed[tokens.first_class + index]) == (mode == "road")


def test_buildings_keep_off_roads_and_each_other():
    from urban_layers.buildings import place_buildings

    block = box(0, 0, 200, 200)
    road = LineString([(100, 0), (100, 120)]).buffer(6.0)
    items = [
        (0, -15.0, -40.0, 30.0, 20.0, 0.0, 9.0),
        (0, 15.0, -40.0, 30.0, 20.0, 0.0, 9.0),
        (0, -40.0, -40.0, 20.0, 20.0, 0.0, 9.0),
    ]
    placed = place_buildings(block, np.array([100.0, 100.0]), 0.0, items, blocked=[road])
    shapes = [Polygon(item["footprint"]) for item in placed]
    assert len(shapes) == 3
    assert all(shape.intersection(road).area <= 1.0 for shape in shapes)
    assert all(one.intersection(other).area <= 1.0 for index, one in enumerate(shapes) for other in shapes[index + 1 :])
    assert [round(shape.area) for shape in shapes] == [600, 600, 400]


def test_a_box_too_deep_for_its_strip_loses_depth_not_length():
    from urban_layers.buildings import place_buildings

    block = box(0, 0, 100, 18)
    roads = box(-50, 18, 150, 40).union(box(-50, -20, 150, 0))
    placed = place_buildings(block, np.array([0.0, 0.0]), 0.0, [(0, 50.0, 10.5, 80.0, 17.0, 0.0, 30.0)], blocked=[roads])
    shape = Polygon(placed[0]["footprint"])
    assert shape.intersection(roads).area <= 1.0
    assert shape.area > 0.9 * 80.0 * 17.0
    corners = np.asarray(placed[0]["footprint"])
    short, long = sorted([np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[1])])
    assert np.isclose(long, 80.0) and 15.0 < short < 17.0


def test_boxes_turn_to_face_the_road():
    from urban_layers.buildings import face_road
    from urban_layers.data import box_polygon

    roads = np.array([[[0.0, 0.0], [100.0, 0.0]]])
    corners = box_polygon(50.0, 20.0, 20.0, 10.0, math.radians(15.0))
    turned = face_road(corners, roads)
    edge = turned[1] - turned[0]
    assert abs(math.degrees(math.atan2(edge[1], edge[0]))) < 0.5
    assert np.allclose(turned.mean(axis=0), corners.mean(axis=0))
    far = box_polygon(50.0, 200.0, 20.0, 10.0, math.radians(15.0))
    assert np.allclose(face_road(far, roads), far)


def test_thin_unused_ground_becomes_natural_cover():
    from urban_layers.city import landuse_labels, natural_cover

    plan = np.zeros((len(CHANNELS), 16, 16), dtype=np.float32)
    plan[CHANNELS.index("coverage"), :, :8] = 0.2
    plan[CHANNELS.index("local"), :, :8] = 0.8
    plan[CHANNELS.index("residential"), :, :8] = 0.9
    natural = natural_cover(plan, landuse_labels(plan))
    assert sum(polygon.area for polygon in natural) > 0.3 * (16 * 128.0) ** 2
    assert all(polygon.centroid.x > 1024 for polygon in natural)


def test_rail_audit_takes_a_single_line():
    from urban_layers.audit import rail_stats

    city = {"size_m": [1000.0, 1000.0], "rail": [{"points": [[0.0, 500.0], [1000.0, 500.0]], "tracks": 2}], "stations": []}
    assert rail_stats(city) == {"track_km": 2.0, "pieces": 1, "loose_ends": 0, "stations": 0}


def test_function_audit_finds_a_town_you_cannot_drive_to():
    from urban_layers.audit import function_stats

    def town(left):
        roads = [{"class": "major", "vertical": "surface", "points": [[left, 500.0], [left + 1000.0, 500.0]]}]
        roads += [{"class": "local", "vertical": "surface", "points": [[left + x, 200.0], [left + x, 800.0]]} for x in (200.0, 500.0, 800.0)]
        house = {"height_m": 9.0, "footprint": [[left + 300.0, 600.0], [left + 340.0, 600.0], [left + 340.0, 630.0], [left + 300.0, 630.0]]}
        return roads, house

    west, west_house = town(0.0)
    east, east_house = town(2000.0)
    flyover = {"class": "major", "vertical": "elevated", "points": [[100.0, 300.0], [900.0, 300.0]]}
    city = {
        "size_m": [3000.0, 1000.0],
        "roads": west + east + [flyover],
        "nodes": [],
        "edges": [],
        "buildings": [west_house, east_house],
        "blocks": [],
        "stations": [{"xy": [320.0, 450.0]}],
    }
    found = function_stats(city)
    assert found["largest_piece_road_share"] < 0.55
    assert found["floor_on_largest_piece"] == 0.5
    assert found["floor_within_800m_of_a_station"] == 0.5
    city["stations"].append({"xy": [2320.0, 450.0], "kind": "tram"})
    assert function_stats(city)["floor_within_800m_of_a_station"] == 0.5
    city["stations"][-1]["kind"] = "subway"
    assert function_stats(city)["floor_within_800m_of_a_station"] == 1.0
    city["stations"].pop()
    city["roads"] = west + east + [{"class": "major", "vertical": "elevated", "points": [[1000.0, 500.0], [2000.0, 500.0]]}]
    found = function_stats(city)
    assert found["largest_piece_road_share"] == 1.0 and found["major_largest_piece_share"] == 1.0


def road(name, level, points):
    return (line_class("road", name, level), np.asarray(points, dtype=np.float64))


def test_a_hanging_flyover_comes_down_onto_the_road():
    from urban_layers.city import lines_to_layers
    from urban_layers.network import join_network

    lines = [road("major", "surface", [[0.0, 0.0], [400.0, 0.0]]), road("major", "elevated", [[200.0, 300.0], [200.0, 25.0]])]
    joined, counts = join_network(lines, np.zeros((len(CHANNELS), 8, 8)), (1024.0, 1024.0))
    assert counts["ramps"] == 1
    flyover = [points for class_id, points in joined if LINE_CLASSES[class_id][2] == "elevated"][0]
    assert np.allclose(flyover[-1], [200.0, 0.0])
    _roads, _rail, graph = lines_to_layers(joined, 1024.0, 1024.0)
    landing = [node for node in graph["nodes"] if np.allclose(node["position_local_m"][:2], [200.0, 0.0])]
    assert len(landing) == 1 and landing[0]["degree"] == 3


def test_cut_off_towns_get_joined_and_lone_stubs_go():
    from urban_layers.audit import road_pieces
    from urban_layers.network import join_network

    def town(left):
        lines = [road("major", "surface", [[left, 500.0], [left + 1000.0, 500.0]])]
        return lines + [road("local", "surface", [[left + x, 200.0], [left + x, 800.0]]) for x in (200.0, 500.0, 800.0)]

    stub_far = road("local", "surface", [[1400.0, 1600.0], [1500.0, 1600.0]])
    stub_near = road("local", "surface", [[200.0, 820.0], [200.0, 900.0]])
    lines = town(0.0) + town(1600.0) + [stub_far, stub_near]
    joined, counts = join_network(lines, np.zeros((len(CHANNELS), 16, 24)), (3072.0, 2048.0))
    assert counts["dropped"] == 1 and counts["major_links"] >= 1
    assert not any(np.allclose(points, stub_far[1]) for _class_id, points in joined)
    _shapes, label = road_pieces([{"points": points, "vertical": LINE_CLASSES[class_id][2]} for class_id, points in joined])
    assert len(set(label.tolist())) == 1
    majors = [(class_id, points) for class_id, points in joined if LINE_CLASSES[class_id][1] == "major"]
    _shapes, label = road_pieces([{"points": points, "vertical": "surface"} for _class_id, points in majors])
    assert len(set(label.tolist())) == 1


def test_buffer_of_a_road_that_crosses_itself():
    from urban_layers.data import corridor

    loop = [[1991.0, 2056.0], [1993.0, 2052.0], [1997.0, 2048.0], [1998.0, 2048.0], [1993.0, 2050.0], [1991.0, 2056.0]]
    assert corridor(loop, 3.0).difference(corridor(loop, 6.0)).area < 0.01


def test_blocks_stop_at_the_road_and_keep_their_holes():
    from urban_layers.city import cut_off_roads, rings
    from urban_layers.data import corridor, polygon_of

    local = line_class("road", "local", "surface")
    road = np.array([[0.0, 100.0], [300.0, 100.0]])
    parts = cut_off_roads([(box(0, 0, 300, 200), 1)], [(local, road)])
    assert len(parts) == 2 and all(use == 1 for _part, use in parts)
    assert all(part.intersection(corridor(road, 3.0)).area < 0.01 for part, _use in parts)
    holed = box(0, 0, 100, 100).difference(box(40, 40, 60, 60))
    assert polygon_of(rings(holed)).equals(holed)


def test_a_split_block_keeps_its_holes():
    from urban_layers.buildings import place_buildings
    from urban_layers.data import block_frame, split_block, to_frame

    island = box(180, 180, 220, 220)
    block = box(0, 0, 800, 800).difference(island)
    parts = split_block(block)
    assert len(parts) > 1
    assert np.isclose(sum(part.area for part in parts), block.area)
    assert not any(part.intersects(Point(200, 200)) for part in parts)
    part = next(part for part in parts if part.distance(island) < 1.0)
    centre, angle = block_frame(part)
    x, y = to_frame(np.array([[200.0, 200.0]]), centre, angle)[0]
    assert place_buildings(part, centre, angle, [(0, x, y, 6.0, 6.0, 0.0, 9.0)], nudge=0.0) == []


def test_roads_over_water_are_marked_as_bridges():
    from urban_layers.city import bridge_spans

    water = box(100, -50, 160, 50)
    spans = bridge_spans([[0.0, 0.0], [300.0, 0.0]], water)
    assert len(spans) == 1 and np.allclose(spans[0], [100.0, 160.0])
    assert bridge_spans([[0.0, 0.0], [300.0, 0.0]], None) == []
    loop = [[110.0, 0.0], [150.0, 0.0], [150.0, 30.0], [110.0, 0.0]]
    spans = bridge_spans(loop, water)
    assert len(spans) == 1 and np.isclose(spans[0][0], 0.0) and np.isclose(spans[0][1], LineString(loop).length)


def test_the_pattern_map_says_which_way_the_streets_run():
    from urban_layers.data import PATTERN, street_pattern

    rise = math.tan(math.radians(22.5))
    slanted = np.array([[10.0, 10.0], [500.0, 10.0 + 490.0 * rise]])
    tile = {"cell": (4, 4), "targets": [(0, slanted), (0, slanted + (0.0, 150.0))], "context": []}
    pattern = street_pattern([tile], (16, 16))
    assert pattern.shape == (len(PATTERN), 16, 16)
    assert pattern[2, 5, 6] == 1.0 and abs(pattern[0, 5, 6]) < 1e-3 and pattern[1, 5, 6] > 0.999 and not pattern[:, 14, 14].any()

    mirror = D4[4]
    turned = street_pattern([{"cell": (4, 4), "targets": [(0, move_points(points, mirror)) for _class_id, points in tile["targets"]]}], (16, 16))
    moved = move_grid(pattern[:, 2:10, 2:10], mirror)
    moved[1] = -moved[1]
    assert np.allclose(moved, turned[:, 2:10, 2:10], atol=1e-4) and turned[1, 5, 5] < -0.999

    plans = [np.zeros((len(CHANNELS), 16, 16), dtype=np.float32)]
    data = StreetDataset([(0, tile)], plans, 1, np.ones(len(CHANNELS), dtype=np.float32), augment=False, max_len=256, exits=True, patterns=[pattern])
    item = data[0]
    assert item["plan"].shape[1] == len(CHANNELS) + len(PATTERN) and item["plan"][:, len(CHANNELS) + 1].max() > 0.999
    batch = collate([item])
    model = StreetModel(1, d=32, layers=1, heads=4, max_len=256, exits=True, guided=True, pattern=True)
    assert torch.isfinite(sequence_loss(model(batch), batch["tokens"]))
    torch.manual_seed(0)
    free = model.eval().generate({name: value for name, value in batch.items() if name != "tokens"}, max_lines=4)
    torch.manual_seed(0)
    led = model.generate({name: value for name, value in batch.items() if name != "tokens"}, max_lines=4, guidance=3.0)
    assert len(free) == 1 and len(led) == 1
    plain = StreetDataset([(0, tile)], plans, 1, np.ones(len(CHANNELS), dtype=np.float32), augment=False, max_len=256, exits=True)[0]
    assert plain["plan"].shape[1] == len(CHANNELS)


def test_a_pattern_map_can_be_drawn_from_a_plan():
    from urban_layers.data import PATTERN
    from urban_layers.plan import PatternCrops, coarse, pattern_controls, sample_pattern

    torch.manual_seed(0)
    np.random.seed(0)
    plans = [np.random.rand(len(CHANNELS), 40, 40).astype(np.float32)]
    patterns = [np.random.uniform(-1.0, 1.0, (len(PATTERN), 40, 40)).astype(np.float32)]
    mean, std, _scale = plan_statistics(plans)
    told_mean, told_std, _scale = plan_statistics(patterns)
    data = PatternCrops(patterns, plans, told_mean, told_std, mean, std, size=16, length=4)
    items = [data[index] for index in range(4)]
    batch = {name: torch.from_numpy(np.stack([item[name] for item in items])) for name in ("plan", "given", "controls", "city")}
    assert batch["plan"].shape == (4, len(PATTERN), 16, 16) and batch["given"].shape == (4, len(CHANNELS), 16, 16)
    assert batch["controls"].shape == (4, len(PATTERN) + 1) and np.allclose(coarse(plans[0])[:, :4, :4], plans[0][:, :4, :4].mean(axis=(1, 2), keepdims=True), atol=1e-6)
    whole = np.ones((40, 40), dtype=bool)
    assert np.allclose(pattern_controls(patterns[0], whole)[:3], patterns[0].mean(axis=(1, 2)), atol=1e-5)
    model = PlanNet(1, channels=len(PATTERN), width=16, given=len(CHANNELS), controls=len(PATTERN) + 1)
    assert torch.isfinite(plan_loss(model, batch))
    drawn = sample_pattern(model, plans[0][:, :16, :16], np.ones(1, dtype=np.float32), np.zeros(len(PATTERN) + 1, dtype=np.float32), told_mean, told_std, mean, std, steps=2)
    assert drawn.shape == (len(PATTERN), 16, 16) and np.hypot(drawn[0], drawn[1]).max() <= 1.0 + 1e-5 and set(np.unique(drawn[2])) <= {0.0, 1.0}


def test_plan_cells_far_apart_get_different_position_codes():
    from urban_layers.decoder import fourier

    crop = np.zeros((len(CHANNELS), 16, 16), dtype=np.float32)
    spots = torch.from_numpy(condition_arrays(crop, [], [], np.ones(1, dtype=np.float32))["plan_xy"])
    close = torch.unique(fourier(spots, 8).round(decimals=4), dim=0)
    wide = torch.unique(fourier(spots / 2.0, 8).round(decimals=4), dim=0)
    assert len(close) < len(spots) == len(wide)
    small = torch.from_numpy(condition_arrays(crop[:, :8, :8], [], [], np.ones(1, dtype=np.float32))["plan_xy"])
    assert len(torch.unique(fourier(small, 8).round(decimals=4), dim=0)) == len(small)
    model = StreetModel(1, d=32, layers=1, heads=4, max_len=64, stations=True, exits=True, reach=2.0)
    assert len(torch.unique(model.where(spots).detach().round(decimals=4), dim=0)) == len(spots)


def test_repairs_join_trim_link_and_drop():
    from urban_layers.network import repair_network, repair_rail

    local = line_class("road", "local", "surface")
    major = line_class("road", "major", "surface")
    rail = line_class("rail", "rail", "surface")
    lines = [
        (major, np.array([[0.0, 100.0], [1000.0, 100.0]])),
        (local, np.array([[200.0, 106.0], [200.0, 400.0]])),
        (local, np.array([[400.0, 100.0], [400.0, 108.0]])),
        (local, np.array([[600.0, 100.0], [600.0, 300.0]])),
        (local, np.array([[600.0, 300.0], [900.0, 300.0]])),
        (local, np.array([[900.0, 380.0], [900.0, 700.0], [700.0, 700.0]])),
        (local, np.array([[50.0, 900.0], [120.0, 900.0]])),
    ]
    fixed, told = repair_network(lines, (1024.0, 1024.0))
    assert (told["near_misses_joined"], told["stubs_removed"], told["links_added"], told["strays_dropped"]) == (1, 1, 1, 1)
    assert told["links"] == [[[900.0, 300.0], [900.0, 380.0]]] and abs(told["link_km"] - 0.08) < 1e-6
    shapes = [LineString(points) for _class_id, points in fixed]
    assert all(min(shape.distance(other) for other in shapes if other is not shape) < 1e-6 for shape in shapes)
    assert not any(LineString(points).length < 10.0 for _class_id, points in fixed)
    tracks = [(rail, np.array([[0.0, 0.0], [500.0, 0.0]])), (rail, np.array([[520.0, 5.0], [900.0, 5.0]])), (rail, np.array([[300.0, 600.0], [360.0, 600.0]])), (major, np.array([[0.0, 50.0], [10.0, 50.0]]))]
    mended, said = repair_rail(tracks)
    kept = [LineString(points) for class_id, points in mended if class_id == rail]
    assert said["rail_strays_dropped"] == 1 and len(kept) == 2 and kept[0].distance(kept[1]) < 1e-6 and sum(1 for class_id, _points in mended if class_id == major) == 1
    weave = [(rail, np.array([[0.0, 0.0], [1000.0, 0.0]])), (rail, np.array([[100.0, 12.0], [400.0, -10.0], [700.0, 14.0], [800.0, 200.0]])), (rail, np.array([[0.0, 300.0], [400.0, 300.0], [400.0, 700.0]]))]
    mended, said = repair_rail(weave)
    through, branch, bend = [LineString(points) for _class_id, points in mended]
    assert 0.5 < said["rail_side_by_side_km"] < 0.7 and through.length == 1000.0 and not through.crosses(branch)
    assert branch.distance(through) < 1e-6 and branch.distance(Point(800.0, 200.0)) < 1e-6
    turns = np.abs(np.diff(np.unwrap(np.arctan2(*np.diff(np.asarray(bend.coords), axis=0).T[::-1]))))
    assert turns.max() < np.deg2rad(40) and bend.distance(Point(400.0, 300.0)) < 30.0
