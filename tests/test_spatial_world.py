from urban_dataset.spatial_world import simplify_transport_graph


def node(node_id, x, boundary=False):
    return {
        "id": node_id,
        "position_local_m": [x, 0.0, 0.0],
        "transport_mode": "road",
        "vertical_mode": "surface",
        "boundary_port_key": node_id if boundary else None,
    }


def edge(edge_id, left, right, x0, x1, road_class="local"):
    return {
        "id": edge_id,
        "from_node": left,
        "to_node": right,
        "transport_mode": "road",
        "class": road_class,
        "vertical_mode": "surface",
        "width_m": 6.0,
        "length_m": abs(x1 - x0),
        "geometry_local_m": [[x0, 0.0, 0.0], [x1, 0.0, 0.0]],
    }


def test_simplify_transport_graph_merges_degree_two_chain():
    graph = {
        "nodes": [
            node("a", 0.0, boundary=True),
            node("b", 10.0),
            node("c", 20.0),
            node("d", 30.0, boundary=True),
        ],
        "edges": [
            edge("ab", "a", "b", 0.0, 10.0),
            edge("bc", "b", "c", 10.0, 20.0),
            edge("cd", "c", "d", 20.0, 30.0),
        ],
    }
    simplified = simplify_transport_graph(graph)
    assert len(simplified["nodes"]) == 2
    assert len(simplified["edges"]) == 1
    result = simplified["edges"][0]
    assert result["from_node"] == "a"
    assert result["to_node"] == "d"
    assert result["geometry_local_m"][0][0] == 0.0
    assert result["geometry_local_m"][-1][0] == 30.0


def test_simplify_transport_graph_keeps_class_change():
    graph = {
        "nodes": [
            node("a", 0.0, boundary=True),
            node("b", 10.0),
            node("c", 20.0, boundary=True),
        ],
        "edges": [
            edge("ab", "a", "b", 0.0, 10.0, "major"),
            edge("bc", "b", "c", 10.0, 20.0, "local"),
        ],
    }
    simplified = simplify_transport_graph(graph)
    assert len(simplified["edges"]) == 2
