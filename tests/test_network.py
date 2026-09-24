from __future__ import annotations

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

# --------------------------------------------------------------------------------------
# NetworkNodeType
# --------------------------------------------------------------------------------------


def test_node_type_classification() -> None:
    """is_hub/is_station/is_pickup/is_delivery correctly classify each type"""
    assert NetworkNodeType.hub == "h"
    assert NetworkNodeType.station == "s"
    assert str(NetworkNodeType.hub) == "h"
    assert str(NetworkNodeType.station) == "s"
    assert NetworkNodeType.is_hub(NetworkNodeType.hub)
    assert NetworkNodeType.is_hub("h")
    assert not NetworkNodeType.is_hub(NetworkNodeType.station)
    assert NetworkNodeType.is_station(NetworkNodeType.station)
    assert NetworkNodeType.is_pickup(NetworkNodeType.pickup)
    assert NetworkNodeType.is_delivery(NetworkNodeType.delivery)
    assert not NetworkNodeType.is_pickup(NetworkNodeType.delivery)


# --------------------------------------------------------------------------------------
# NetworkNodeName -- 1-based naming/parsing round trip
# --------------------------------------------------------------------------------------


def test_hub_name_is_1_based() -> None:
    """make_hub_name() uses the 1-based hub index"""
    assert NetworkNodeName.make_hub_name(1) == "h_1"
    assert NetworkNodeName.make_hub_name(3) == "h_3"


def test_agent_node_name_and_parsing() -> None:
    """make_agent_node_name()'s output round-trips through the get_* parsers"""
    name = NetworkNodeName.make_agent_node_name(1, NetworkNodeType.station, 3)
    assert name == "a_1_s_3"
    assert NetworkNodeName.get_node_index(name) == 3
    assert NetworkNodeName.get_node_type(name) == NetworkNodeType.station
    assert NetworkNodeName.get_request_index(name) == 1


def test_request_pickup_delivery_names_and_parsing() -> None:
    """Pickup/delivery node names round-trip through the get_* parsers"""
    p = NetworkNodeName.make_request_pickup_node_name(2, 5)
    d = NetworkNodeName.make_request_delivery_node_name(2, 7)
    assert p == "r_2_p_5"
    assert d == "r_2_d_7"
    assert NetworkNodeName.get_node_index(p) == 5
    assert NetworkNodeName.get_node_index(d) == 7
    assert NetworkNodeName.get_request_index(p) == 2
    assert NetworkNodeName.get_node_type(p) == NetworkNodeType.pickup
    assert NetworkNodeName.get_node_type(d) == NetworkNodeType.delivery


# --------------------------------------------------------------------------------------
# RoadNetwork
# --------------------------------------------------------------------------------------


def test_road_network_seeded_reproducibility() -> None:
    """Two RoadNetworks built from equally-seeded generators are identical"""
    rng1 = np.random.default_rng(5)
    net1 = RoadNetwork(nr_hubs=2, nr_stations=4, rng=rng1)
    rng2 = np.random.default_rng(5)
    net2 = RoadNetwork(nr_hubs=2, nr_stations=4, rng=rng2)
    assert np.array_equal(net1.locations, net2.locations)
    assert np.array_equal(net1.travel_times, net2.travel_times)


def test_road_network_node_counts() -> None:
    """nr_nodes and the locations/travel_times array shapes match hubs+stations"""
    net = RoadNetwork(nr_hubs=3, nr_stations=10, rng=np.random.default_rng(1))
    assert net.nr_nodes == 13
    assert net.locations.shape == (13, 2)
    assert net.travel_times.shape == (13, 13)


def test_travel_time_lookup_matches_manual_calibration() -> None:
    """get_travel_time_by_index() falls within the [1.1, 1.25] variability band"""
    net = RoadNetwork(nr_hubs=2, nr_stations=2, rng=np.random.default_rng(1))
    # h_1 (array row 0) <-> s_1 (array row 2, since nr_hubs=2)
    looked_up = net.get_travel_time_by_index("h", 1, "s", 1)
    distance_units = np.linalg.norm(net.locations[0] - net.locations[2])
    lo = (distance_units * METERS_PER_UNIT / AVERAGE_SPEED_MS) / 60 * 1.1
    hi = (distance_units * METERS_PER_UNIT / AVERAGE_SPEED_MS) / 60 * 1.25
    assert lo - 1e-6 <= looked_up <= hi + 1e-6


def test_travel_times_are_symmetric_and_zero_diagonal() -> None:
    """A randomly generated travel-time matrix is symmetric with a zero diagonal"""
    net = RoadNetwork(nr_hubs=2, nr_stations=3, rng=np.random.default_rng(2))
    assert np.allclose(net.travel_times, net.travel_times.T)
    assert np.allclose(np.diag(net.travel_times), 0.0)


def test_provided_locations_are_not_regenerated() -> None:
    """Caller-supplied locations are used/stored, not overwritten"""
    locations = np.array([[40.0, 40.0], [50.0, 50.0], [60.0, 60.0]])
    net = RoadNetwork(
        nr_hubs=1, nr_stations=2, locations=locations, rng=np.random.default_rng(1)
    )
    assert np.array_equal(net.locations, locations)


def test_directed_travel_times_are_preserved_in_virtual_graph() -> None:
    """get_travel_times_dict() keeps an asymmetric physical matrix directed"""
    locations = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    travel_times = np.array(
        [
            [0.0, 2.0, 3.0],
            [4.0, 0.0, 1.0],
            [3.0, 3.0, 0.0],
        ]
    )
    net = RoadNetwork(
        nr_hubs=1,
        nr_stations=2,
        locations=locations,
        travel_times=travel_times,
    )
    agent = ModemsAgent(NetworkNodeType.hub, 1)
    request = ModemsRequest(1, 2)

    virtual = net.get_travel_times_dict([agent], [request])

    assert virtual[("a_1_h_1", "r_1_p_1")] == 2.0
    assert virtual[("r_1_p_1", "a_1_h_1")] == 4.0


def test_travel_times_must_satisfy_directed_triangle_inequality() -> None:
    """__init__ rejects a travel-time matrix that violates the triangle inequality"""
    locations = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    travel_times = np.array(
        [
            [0.0, 1.0, 5.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="triangle inequality"):
        RoadNetwork(
            nr_hubs=1,
            nr_stations=2,
            locations=locations,
            travel_times=travel_times,
        )
