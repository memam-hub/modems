from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .core import (
    DEFAULT_TIME_WINDOW,
    ModemsAgent,
    ModemsRequest,
    ModemsScenario,
    RequestStatus,
    ScenarioTiming,
    ScenarioType,
)
from .network import NetworkNodeType, RoadNetwork

# default request arrival base rate per hour
DEFAULT_BASE_RATE_PER_HOUR = 5.0

# Duration of a full simulated workday (minutes)
DEFAULT_WORKDAY_LENGTH = 480.0

# Buffer (minutes) to run the simulator longer than workday so that requests arriving
# near the end of the day can still be resolved (completed or rejected)
DEFAULT_WORKDAY_BUFFER = 90.0

# Normal range for the starting agent SoC values
NORMAL_SOC_RANGE = (0.8, 1.0)

# Stress range for the starting agent SoC values
STRESS_SOC_RANGE = (0.5, 0.7)


class ScenarioSocRange(StrEnum):
    """
    Starting-SoC-range kind for a scenario. "normal" and "stress" are the two
    built-in presets (NORMAL_SOC_RANGE/STRESS_SOC_RANGE); "custom" is any other
    caller-supplied [lb, ub] (SocRangeSpec.custom()). "both" is CLI selection
    sugar only (see benchmarks/_cli_common.py's resolve_cli_soc()) -- it never
    appears as a SocRangeSpec.kind
    """

    both = "both"
    normal = "normal"
    stress = "stress"
    custom = "custom"


@dataclass(slots=True)
class SocRangeSpec:
    """
    The starting-agent-SoC range: a ScenarioSocRange kind with [lb, ub] bounds.
    Bounds are implied by kind for normal()/stress(); custom() carries caller-supplied
    bounds, validated to 0 < lb <= ub <= 1.0. Use.suffix for scenario/point naming and
    .to_dict()/.from_dict() for serialization/persistence
    """

    kind: ScenarioSocRange
    lb: float
    ub: float

    @classmethod
    def normal(cls) -> SocRangeSpec:
        return cls(ScenarioSocRange.normal, *NORMAL_SOC_RANGE)

    @classmethod
    def stress(cls) -> SocRangeSpec:
        return cls(ScenarioSocRange.stress, *STRESS_SOC_RANGE)

    @classmethod
    def custom(cls, lb: float, ub: float) -> SocRangeSpec:
        if not 0 < lb <= ub <= 1.0:
            raise ValueError("custom SoC range must satisfy 0 < lb <= ub <= 1.0")
        return cls(ScenarioSocRange.custom, lb, ub)

    @property
    def bounds(self) -> tuple[float, float]:
        """(lb, ub), ready for generate_random_scenario(soc_lb=, soc_ub=)"""
        return (self.lb, self.ub)

    @property
    def suffix(self) -> str:
        """Scenario/point name suffix, e.g. "_soc80-100" """
        return f"_soc{int(self.lb * 100)}-{int(self.ub * 100)}"

    @property
    def label(self) -> str:
        """Display name: "normal"/"stress"/"custom" """
        return str(self.kind)

    def to_dict(self) -> dict:
        """
        {"kind": "normal"}/{"kind": "stress"} for the two presets, since their bounds
        are implied by kind; {"kind": "custom", "lb": lb, "ub": ub} for a custom range,
        the only case where bounds are not recoverable from kind alone
        """
        if self.kind == ScenarioSocRange.custom:
            return {"kind": str(self.kind), "lb": self.lb, "ub": self.ub}
        return {"kind": str(self.kind)}

    @classmethod
    def from_dict(cls, data: dict) -> SocRangeSpec:
        kind = ScenarioSocRange(data["kind"])
        if kind == ScenarioSocRange.normal:
            return cls.normal()
        if kind == ScenarioSocRange.stress:
            return cls.stress()
        return cls.custom(data["lb"], data["ub"])


class ModemsScenarioGenerator:
    """
    MODEMS scenario generator. Can generate random agents, requests, and test instances
    """

    rng: np.random.Generator
    network: RoadNetwork

    def __init__(
        self,
        seed: int | None = None,
        random_network: bool = False,
        nr_hubs: int = 3,
        nr_stations: int = 15,
        network: RoadNetwork | None = None,
    ) -> None:
        """
        Initialize the class. Generate a random road network or default to defined one

        Args:
            seed: Optional seed for the random number generator. Defaults to None
            random_network: Whether to generate a random road network. Defaults to False
            nr_hubs: Optional, number of hubs in the (generated) network
            nr_stations: Optional, number of stations in the (generated) network
            network: Optional existing network to use. Mutually exclusive with
                random_network; useful when generating demand for a workday
                simulation on a caller-provided network
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        if network is not None and random_network:
            raise ValueError("network and random_network=True are mutually exclusive")
        if network is not None:
            self.network = copy.deepcopy(network)
        elif random_network:
            self.network = RoadNetwork(
                nr_hubs=nr_hubs, nr_stations=nr_stations, rng=self.rng
            )
        else:
            locations = np.array(
                [
                    [42.3, 40.5],
                    [58.4, 64.0],
                    [64.2, 48.9],
                    [52.8, 53.3],
                    [46.0, 50.4],
                    [57.1, 61.2],
                    [63.0, 56.6],
                    [42.5, 57.1],
                    [47.0, 60.3],
                    [53.3, 63.7],
                    [43.6, 48.2],
                    [51.4, 58.0],
                    [62.7, 63.8],
                    [42.3, 45.0],
                    [64.2, 53.4],
                    [44.75, 61.2],
                    [49.4, 49.35],
                    [57.9, 54.95],
                ]
            )
            nr_hubs = 3
            nr_stations = 15
            self.network = RoadNetwork(
                nr_hubs=nr_hubs,
                nr_stations=nr_stations,
                locations=locations,
                rng=self.rng,
            )

    # ----------------------------------------------------------------------------------
    # Spatial and Temporal clustering helpers
    # ----------------------------------------------------------------------------------

    def _station_locations(self) -> np.ndarray:
        """return (x,y) locations of service stations"""
        return self.network.locations[self.network.nr_hubs :]

    def sample_cluster_centers(self, nr_clusters: int) -> np.ndarray:
        """
        Sample nr_clusters (x, y) points within the stations' bounding box, used as
        the geographic centers requests get drawn around in CLUSTERED/MIXED mode
        (centers do not necessarily coincide with an actual station, rather are snapped)
        """
        station_locs = self._station_locations()
        x_lo, x_hi = station_locs[:, 0].min(), station_locs[:, 0].max()
        y_lo, y_hi = station_locs[:, 1].min(), station_locs[:, 1].max()
        return np.column_stack(
            [
                self.rng.uniform(x_lo, x_hi, size=nr_clusters),
                self.rng.uniform(y_lo, y_hi, size=nr_clusters),
            ]
        )

    def _nearest_station_index(self, point: np.ndarray) -> int:
        """0-based station-space index of the station nearest to (x,y) point"""
        station_locs = self._station_locations()
        dists = np.linalg.norm(station_locs - point, axis=1)
        return int(np.argmin(dists))

    def _sample_station_index(
        self,
        use_clustered: bool,
        cluster_centers: np.ndarray | None,
        cluster_spread: float,
    ) -> int:
        """
        Sample a single 0-based station-space index: either uniformly at random, or
        if use_clustered, from a truncated-normal spread around a randomly chosen
        cluster center, which is snapped to the nearest actual station
        """
        if use_clustered and cluster_centers is not None and len(cluster_centers) > 0:
            center = cluster_centers[self.rng.integers(0, len(cluster_centers))]
            point = center + self.rng.normal(0.0, cluster_spread, size=2)
            return self._nearest_station_index(point)
        return int(self.rng.integers(0, self.network.nr_stations))

    def sample_surge_moments(
        self, nr_surges: int, time_lb: float, time_ub: float
    ) -> np.ndarray:
        """
        Sample nr_surges moments within [time_lb, time_ub] for ScenarioTiming.tight
        earliest-pickup times
        """
        return self.rng.uniform(time_lb, time_ub, size=nr_surges)

    def _sample_earliest_pickup(
        self,
        use_tight: bool,
        surge_moments: np.ndarray | None,
        temporal_spread: float,
        time_lb: float,
        time_ub: float,
    ) -> float:
        """sample an earliest pickup time within surge_moments if use_tight"""
        if use_tight and surge_moments is not None and len(surge_moments) > 0:
            moment = surge_moments[self.rng.integers(0, len(surge_moments))]
            t_pickup = self.rng.normal(moment, temporal_spread)
            return round(float(np.clip(t_pickup, time_lb, time_ub)), 1)
        return round(self.rng.uniform(time_lb, time_ub), 1)

    # ----------------------------------------------------------------------------------
    # Agent and request generation
    # ----------------------------------------------------------------------------------

    def generate_random_agent(
        self,
        node_lb: int,
        node_ub: int,
        time_lb: float,
        time_ub: float,
        soc_lb: float,
        soc_ub: float,
    ) -> ModemsAgent:
        """
        Generate a random agent starting at a hub or station

        Args:
            node_lb: Lower bound for node (hub or station) indices (0-based)
            node_ub: Upper bound for node (hub or station) indices (0-based)
            time_lb: Lower bound for initial time
            time_ub: Upper bound for initial time
            soc_lb: Lower bound for initial SOC
            soc_ub: Upper bound for initial SOC

        Returns:
            ModemsAgent object
        """
        node_lb = max(node_lb, 0)
        node_ub = min(node_ub, self.network.nr_nodes)
        node_idx = int(
            self.rng.integers(node_lb, node_ub)
        )  # 0-based raw array position
        if node_idx >= self.network.nr_hubs:
            node_type = NetworkNodeType.station
            node_idx -= self.network.nr_hubs
        else:
            node_type = NetworkNodeType.hub
        load_max = int(self.rng.choice([5, 6, 8]))  # three agent sizes
        time_initial = round(self.rng.uniform(time_lb, time_ub), 1)
        soc_initial = self.rng.uniform(soc_lb, soc_ub)
        return ModemsAgent(
            node_type=node_type,
            node_index=node_idx + 1,  # stored index is 1-based
            load_max=load_max,
            time_initial=time_initial,
            soc_initial=soc_initial,
            agent_id=self._make_unique_id(),
        )

    def _make_unique_id(self) -> str:
        """
        Deterministic, unique persistent request ID: sha256 hash of a {nonce} sampled
        using this generator seeded RNG stream. Reproducible given the same seed
        (same draw sequence -> same IDs); unique in practice (64-bit random space)
        even without depending on any process-global counter
        """
        nonce = int(self.rng.integers(0, 2**63))
        return hashlib.sha256(str(nonce).encode()).hexdigest()[:16]

    def generate_random_request(
        self,
        load_lb: int,
        load_ub: int,
        time_lb: float,
        time_ub: float,
        tw_length: float,
        status: RequestStatus | str = RequestStatus.new,
        cluster_centers: np.ndarray | None = None,
        cluster_spread: float = 6.0,
        use_clustered_pickup: bool = False,
        use_clustered_delivery: bool = False,
        surge_moments: np.ndarray | None = None,
        temporal_spread: float = 3.0,
        use_tight_timing: bool = False,
    ) -> ModemsRequest:
        """
        Generate a random request

        Args:
            load_lb: Lower bound for load
            load_ub: Upper bound for load
            time_lb: Lower bound for earliest pickup time
            time_ub: Upper bound for earliest pickup time
            tw_length: Fixed pickup time-window length (omega)
            status: "new" (R_n) or "scheduled" (R_s)
            cluster_centers: Optional (x, y) centers for spatial clustering
            cluster_spread: std-dev (in coordinate units) of the spread around a center
            use_clustered_pickup: sample the pickup station using cluster_centers
            use_clustered_delivery: sample the delivery station using cluster_centers
            surge_moments: optional "surge" times for temporal clustering
            temporal_spread: std-dev (in minutes) of the spread around a surge moment
            use_tight_timing: sample earliest_pickup TIGHTly around a surge moment

        Returns:
            ModemsRequest object
        """
        pickup_idx = self._sample_station_index(
            use_clustered_pickup, cluster_centers, cluster_spread
        )
        delivery_idx = self._sample_station_index(
            use_clustered_delivery, cluster_centers, cluster_spread
        )
        nr_attempts = 0  # avoid infinite loops
        while delivery_idx == pickup_idx and nr_attempts < 20:
            delivery_idx = self._sample_station_index(
                use_clustered_delivery, cluster_centers, cluster_spread
            )
            nr_attempts += 1
        if delivery_idx == pickup_idx:
            # extremely bad luck, fall back to a guaranteed-distinct pick
            delivery_idx = (pickup_idx + 1) % self.network.nr_stations

        load = int(self.rng.integers(load_lb, load_ub))
        service_time = min(1.0, max(0.2, load / 10))  # [0.2-1.0] min, load-dependent
        earliest_pickup = self._sample_earliest_pickup(
            use_tight_timing, surge_moments, temporal_spread, time_lb, time_ub
        )
        return ModemsRequest(
            node_pickup_index=pickup_idx + 1,  # stored index is 1-based
            node_delivery_index=delivery_idx + 1,
            load=load,
            service_time=service_time,
            earliest_pickup=earliest_pickup,
            tw_length=tw_length,
            status=status,
            request_id=self._make_unique_id(),
        )

    def generate_random_scenario(
        self,
        nr_agents: int = 3,
        nr_requests: int = 10,
        scenario_type: ScenarioType | str = ScenarioType.random,
        scenario_timing: ScenarioTiming | str = ScenarioTiming.loose,
        tw_length: float = DEFAULT_TIME_WINDOW,
        nr_scheduled: int = 0,
        nr_clusters: int = 3,
        cluster_spread: float = 6.0,
        mixed_probability: float = 0.5,
        nr_surges: int = 2,
        temporal_spread: float = 3.0,
        soc_lb: float = 0.8,
        soc_ub: float = 1.0,
    ) -> ModemsScenario:
        """
        Generate a complete random scenario with road network, agents, and requests

        Args:
            nr_agents: Number of agents. Defaults to 3
            nr_requests: Number of requests. Defaults to 10
            scenario_type: Spatial distribution of pickup/delivery stations
                (random/clustered/mixed). Defaults to ScenarioType.random
            scenario_timing: Temporal distribution of earliest-pickup times
                (loose/tight). Defaults to ScenarioTiming.loose
            tw_length: Fixed pickup time-window length (omega). Defaults to 5.0 minutes
            nr_scheduled: Number of requests (out of nr_requests) marked as
                scheduled (R_s) rather than new (R_n). Defaults to 0
            nr_clusters: Number of geographic cluster centers for clustered/mixed
                scenario_type. Defaults to 3
            cluster_spread: Std-dev (coordinate units) of the spread around a cluster
                center. Defaults to 6.0
            mixed_probability: Per-endpoint probability of using the clustered draw
                when scenario_type is mixed. Defaults to 0.5
            nr_surges: Number of "surge" moments for TIGHT timings. Defaults to 2
            temporal_spread: Std-dev (minutes) of the spread around a surge moment.
                Defaults to 3.0
            soc_lb: Lower bound for each agent's initial SoC. Defaults to 0.8
            soc_ub: Upper bound for each agent's initial SoC. Defaults to 1.0

        Returns:
            ModemsScenario: Generated scenario
        """
        scenario_type = ScenarioType(scenario_type)
        scenario_timing = ScenarioTiming(scenario_timing)

        agents = [
            self.generate_random_agent(
                node_lb=0,
                node_ub=self.network.nr_nodes,
                time_lb=0.0,
                time_ub=10.0,
                soc_lb=soc_lb,
                soc_ub=soc_ub,
            )
            for _ in range(nr_agents)
        ]

        cluster_centers = (
            self.sample_cluster_centers(nr_clusters)
            if scenario_type in (ScenarioType.clustered, ScenarioType.mixed)
            else None
        )
        # e^r must lie within [15, 60] in the future relative to submission time (t=0)
        time_lb, time_ub = 15.0, 60.0
        surge_moments = (
            self.sample_surge_moments(nr_surges, time_lb, time_ub)
            if scenario_timing == ScenarioTiming.tight
            else None
        )

        requests = []
        for i in range(nr_requests):
            if scenario_type == ScenarioType.random:
                use_clustered_pickup = use_clustered_delivery = False
            elif scenario_type == ScenarioType.clustered:
                use_clustered_pickup = use_clustered_delivery = True
            else:  # mixed: each endpoint independently clustered with mixed_probability
                use_clustered_pickup = self.rng.random() < mixed_probability
                use_clustered_delivery = self.rng.random() < mixed_probability

            requests.append(
                self.generate_random_request(
                    load_lb=1,
                    load_ub=5,
                    time_lb=time_lb,
                    time_ub=time_ub,
                    tw_length=tw_length,
                    status=(
                        RequestStatus.scheduled
                        if i < nr_scheduled
                        else RequestStatus.new
                    ),
                    cluster_centers=cluster_centers,
                    cluster_spread=cluster_spread,
                    use_clustered_pickup=use_clustered_pickup,
                    use_clustered_delivery=use_clustered_delivery,
                    surge_moments=surge_moments,
                    temporal_spread=temporal_spread,
                    use_tight_timing=(scenario_timing == ScenarioTiming.tight),
                )
            )
        return ModemsScenario(
            agents=agents,
            requests=requests,
            network=self.network,
            type=scenario_type,
            timing=scenario_timing,
        )

    def generate_workday_requests(
        self,
        workday_length: float = DEFAULT_WORKDAY_LENGTH,
        base_rate_per_hour: float = DEFAULT_BASE_RATE_PER_HOUR,
        nr_surges: int = 3,
        surge_size_range: tuple[int, int] = (6, 10),
        surge_spread_min: float = 5.0,
        scenario_type: ScenarioType | str = ScenarioType.random,
        nr_clusters: int = 3,
        cluster_spread: float = 6.0,
        mixed_probability: float = 0.5,
        tw_length: float = DEFAULT_TIME_WINDOW,
        min_lead: float = 15.0,
        max_lead: float = 45.0,
    ) -> list[tuple[float, ModemsRequest]]:
        """
        Generate a full workday request-arrival stream: a steady baseline Poisson
        process plus a handful of "surge" bursts (e.g., end-of-lecture), spread across
        [0, workday_length]. Returns a list of (time_submit, ModemsRequest) sorted by
        time_submit; each request earliest_pickup is time_submit + a random lead
        in [min_lead, max_lead], matching the >=15min submission-to-pickup
        """
        scenario_type = ScenarioType(scenario_type)
        cluster_centers = (
            self.sample_cluster_centers(nr_clusters)
            if scenario_type in (ScenarioType.clustered, ScenarioType.mixed)
            else None
        )

        def make_one_request(t_submit: float) -> ModemsRequest:
            if scenario_type == ScenarioType.random:
                use_p = use_d = False
            elif scenario_type == ScenarioType.clustered:
                use_p = use_d = True
            else:
                use_p = self.rng.random() < mixed_probability
                use_d = self.rng.random() < mixed_probability
            t_lead = self.rng.uniform(min_lead, max_lead)
            pickup_time = t_submit + t_lead
            return self.generate_random_request(
                load_lb=1,
                load_ub=6,
                time_lb=pickup_time,
                time_ub=pickup_time,
                tw_length=tw_length,
                status=RequestStatus.new,
                cluster_centers=cluster_centers,
                cluster_spread=cluster_spread,
                use_clustered_pickup=use_p,
                use_clustered_delivery=use_d,
            )

        request_submissions: list[tuple[float, ModemsRequest]] = []
        rate_per_min = base_rate_per_hour / 60.0
        if rate_per_min > 0:
            t_submit = 0.0
            while True:
                t_submit += self.rng.exponential(1.0 / rate_per_min)
                if t_submit >= workday_length:
                    break
                request_submissions.append((t_submit, make_one_request(t_submit)))

        surge_moments = (
            self.rng.uniform(0.0, workday_length, size=nr_surges)
            if nr_surges > 0
            else []
        )
        for moment in surge_moments:
            size = int(self.rng.integers(surge_size_range[0], surge_size_range[1] + 1))
            for _ in range(size):
                t_submit = float(
                    np.clip(
                        self.rng.normal(moment, surge_spread_min),
                        0.0,
                        workday_length - 1e-3,
                    )
                )
                request_submissions.append((t_submit, make_one_request(t_submit)))

        request_submissions.sort(key=lambda x: x[0])
        return request_submissions
