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
from urban_layers.data import CHANNELS, LINE_CLASSES, build_city, line_class, load_cities
from urban_layers.decoder import TokenDecoder, sequence_loss
from urban_layers.plan import PlanCrops, PlanNet, plan_loss, plan_statistics, urban_cells
from urban_layers.streets import D4, StreetDataset, StreetModel, collate, find_ports, move_grid, move_points, order_lines, port_key
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
