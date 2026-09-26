"""Unit tests for modems.network: node naming/types and the RoadNetwork container"""

from __future__ import annotations

import os

import matplotlib
import numpy as np
import pytest

from modems.core import ModemsAgent, ModemsRequest
from modems.network import (
    AVERAGE_SPEED_MS,
    METERS_PER_UNIT,
    NetworkNodeName,
    NetworkNodeType,
    RoadNetwork,
)

from .builders import line_network

# --------------------------------------------------------------------------------------
# NetworkNodeType / NetworkNodeName
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node_type, hub, station, pickup, delivery",
    [
        ("h", True, False, False, False),
        ("s", False, True, False, False),
        ("p", False, True, True, False),
        ("d", False, True, False, True),
        ("a", False, False, False, False),
        ("r", False, False, False, False),
    ],
)
def test_node_type_classification(
    node_type: str, hub: bool, station: bool, pickup: bool, delivery: bool
) -> None:
    assert NetworkNodeType.is_hub(node_type) == hub
    assert NetworkNodeType.is_station(node_type) == station
    assert NetworkNodeType.is_pickup(node_type) == pickup
    assert NetworkNodeType.is_delivery(node_type) == delivery


@pytest.mark.parametrize(
    "name, node_type, node_index",
    [
        (NetworkNodeName.make_agent_node_name(1, "h", 2), "h", 2),
        (NetworkNodeName.make_agent_node_name(4, "p", 3), "s", 3),
        (NetworkNodeName.make_request_pickup_node_name(12, 5), "p", 5),
        (NetworkNodeName.make_request_delivery_node_name(12, 7), "d", 7),
        (NetworkNodeName.make_hub_name(3), "h", 3),
        (NetworkNodeName.make_station_name(11), "s", 11),
    ],
)
def test_node_names_round_trip_through_parsers(
    name: str, node_type: str, node_index: int
) -> None:
    assert NetworkNodeName.get_node_type(name) == node_type
    assert NetworkNodeName.get_node_index(name) == node_index


def test_node_name_formats_and_request_index() -> None:
    assert NetworkNodeName.make_agent_node_name(4, "p", 3) == "a_4_s_3"
    assert NetworkNodeName.make_request_pickup_node_name(12, 5) == "r_12_p_5"
    assert NetworkNodeName.make_hub_name(3) == "h_3"
    assert NetworkNodeName.get_request_index("r_12_d_7") == 12


# --------------------------------------------------------------------------------------
# RoadNetwork: generation
# --------------------------------------------------------------------------------------


def test_generated_network_shape_bounds_and_symmetry() -> None:
    net = RoadNetwork(nr_hubs=3, nr_stations=10, rng=np.random.default_rng(1))
    assert net.nr_nodes == 13
    assert net.locations.shape == (13, 2)
    assert np.all((net.locations >= 40.0) & (net.locations <= 65.0))
    assert len(np.unique(net.locations, axis=0)) == 13
    assert np.allclose(net.travel_times, net.travel_times.T)
    assert np.all(np.diag(net.travel_times) == 0.0)


def test_generated_travel_times_are_scaled_distances_with_one_shared_factor() -> None:
    net = RoadNetwork(nr_hubs=2, nr_stations=4, rng=np.random.default_rng(3))
    dist = np.linalg.norm(net.locations[:, None] - net.locations[None, :], axis=2)
    base = dist * METERS_PER_UNIT / AVERAGE_SPEED_MS / 60.0
    off_diag = ~np.eye(net.nr_nodes, dtype=bool)
    factors = net.travel_times[off_diag] / base[off_diag]
    assert np.allclose(factors, factors[0])
    assert 1.1 <= factors[0] <= 1.25


def test_generated_network_is_seed_reproducible() -> None:
    nets = [RoadNetwork(2, 4, rng=np.random.default_rng(5)) for _ in range(2)]
    assert np.array_equal(nets[0].locations, nets[1].locations)
    assert np.array_equal(nets[0].travel_times, nets[1].travel_times)


def test_given_locations_are_kept_and_only_times_generated() -> None:
    locations = np.array([[40.0, 40.0], [50.0, 50.0], [60.0, 60.0]])
    net = RoadNetwork(1, 2, locations=locations, rng=np.random.default_rng(1))
    assert np.array_equal(net.locations, locations)
    assert net.travel_times[0, 2] == pytest.approx(2 * net.travel_times[0, 1])


# --------------------------------------------------------------------------------------
# RoadNetwork: validation of caller-provided data
# --------------------------------------------------------------------------------------

LINE3 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
TIMES3 = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"nr_hubs": 0}, "nr_hubs"),
        ({"nr_stations": 0}, "nr_stations"),
        ({"locations": LINE3[:2]}, "shape"),
        ({"locations": np.where(LINE3 == 2.0, np.inf, LINE3)}, "finite"),
        ({"locations": np.zeros((3, 2)), "travel_times": None}, "unique"),
        ({"travel_times": TIMES3[:2, :2]}, "square"),
        ({"travel_times": np.where(TIMES3 == 2.0, np.nan, TIMES3)}, "finite"),
        ({"travel_times": TIMES3 - 1.5}, "non-negative"),
        ({"travel_times": TIMES3 + np.eye(3)}, "diagonal"),
        ({"travel_times": np.where(TIMES3 == 1.0, 0.0, TIMES3)}, "positive"),
        ({"travel_times": np.where(TIMES3 == 2.0, 5.0, TIMES3)}, "triangle"),
    ],
)
def test_network_rejects_invalid_inputs(kwargs: dict, match: str) -> None:
    arguments = {
        "nr_hubs": 1,
        "nr_stations": 2,
        "locations": LINE3,
        "travel_times": TIMES3,
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        RoadNetwork(**arguments)


def test_duplicate_locations_are_allowed_when_times_are_given() -> None:
    net = RoadNetwork(1, 2, locations=np.zeros((3, 2)), travel_times=TIMES3)
    assert np.array_equal(net.travel_times, TIMES3)


# --------------------------------------------------------------------------------------
# RoadNetwork: lookups
# --------------------------------------------------------------------------------------


def test_travel_time_lookup_maps_hubs_before_stations() -> None:
    net = line_network([0.0, 10.0, 1.0, 5.0], nr_hubs=2)  # h_1, h_2, s_1, s_2
    assert net.get_travel_time_by_index("h", 2, "s", 1) == 9.0
    assert net.get_travel_time_by_index("s", 2, "h", 1) == 5.0
    assert net.get_travel_time_by_name("a_1_h_2", "r_1_p_2") == 5.0
    assert net.get_travel_time_by_name("r_1_d_1", "h_1") == 1.0


def test_virtual_travel_graph_is_complete_and_keeps_direction() -> None:
    times = np.array([[0.0, 2.0, 3.0], [4.0, 0.0, 1.0], [3.0, 3.0, 0.0]])
    net = RoadNetwork(1, 2, locations=LINE3, travel_times=times)
    graph = net.get_travel_times_dict([ModemsAgent("h", 1)], [ModemsRequest(1, 2)])
    nodes = ["a_1_h_1", "r_1_p_1", "r_1_d_2", "h_1"]
    assert set(graph) == {(i, j) for i in nodes for j in nodes}
    assert graph[("a_1_h_1", "r_1_p_1")] == 2.0
    assert graph[("r_1_p_1", "a_1_h_1")] == 4.0
    assert graph[("r_1_p_1", "r_1_d_2")] == 1.0
    assert graph[("r_1_d_2", "r_1_p_1")] == 3.0


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------


def test_agent_colors_are_valid_and_cycle() -> None:
    colors = [RoadNetwork._agent_color(i) for i in range(12)]
    assert all(matplotlib.colors.is_color_like(c) for c in colors)
    assert colors[:6] == colors[6:]


def test_plots_write_only_the_requested_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    net = line_network([0.0, 1.0, 2.0, 3.0])
    routes = {
        "agent_1": ["a_1_h_1", "r_1_p_1", "r_2_p_2", "r_1_d_2", "r_2_d_3", "h_1"],
        "agent_2": ["a_2_s_1", "r_3_p_1", "r_3_d_3", "h_1"],
    }
    net.plot_network(show=False)
    net.plot_routes(routes, show=False)
    assert os.listdir(tmp_path) == []
    net.plot_network(show=False, outfile="network.png")
    net.plot_routes(routes, show=False, outfile="routes.png")
    assert sorted(os.listdir(tmp_path)) == ["network.png", "routes.png"]
    assert all(os.path.getsize(tmp_path / f) > 0 for f in os.listdir(tmp_path))
