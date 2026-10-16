import numpy as np
import pytest
import torch
import geopandas as gpd
from shapely.geometry import LineString, box

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
    # the port line, then the major road, then locals that start on something already drawn
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
    # joined end 20 m away lands on the road and the road gets the vertex
    assert snapped[1][1][0].tolist() == [50.0, 0.0]
    assert [50.0, 0.0] in snapped[0][1].tolist()
    # a free end 10 m away is a real dead end and stays
    assert snapped[2][1][0].tolist() == [80.0, 10.0]
    # ends 1 m apart become one junction
    assert snapped[1][1][-1].tolist() == snapped[3][1][0].tolist()
    chained = snap_lines(
        [
            (local, np.array([[0.0, 0.0], [0.0, -50.0]])),
            (local, np.array([[10.0, 0.0], [10.0, 50.0]]), (True, False)),
            (local, np.array([[20.5, 0.0], [40.0, 30.0]]), (True, False)),
        ]
    )
    # the third line goes to the second line's end, which itself goes to the first line's end, so all three meet
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
    report = audit_city(city, plan)
    assert "network" in report and "follows_plan" in report


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
            # a line at the point limit is allowed to end anywhere, and joined ends get snapped onto a line
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
    # one big block with no roads inside crosses the land use edge at x = 1024
    lines = [(local, LineString([(100, 100), (1900, 100)]), 0.0), (local, LineString([(100, 1900), (1900, 1900)]), 0.0)]
    pieces = landuse_blocks(lines, None, regions, box(0, 0, 2048, 2048).exterior)
    middle = box(100, 100, 1900, 1900)
    big = [(polygon, LANDUSE[use]) for polygon, use in pieces if polygon.intersection(middle).area > 100000]
    assert {use for _polygon, use in big} == {"residential", "industrial"}
    for polygon, use in big:
        x = polygon.centroid.x
        assert (x < 1024) == (use == "residential")


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
