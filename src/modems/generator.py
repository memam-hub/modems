from __future__ import annotations

import copy
import hashlib
import itertools
import math
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

# default request arrival base rate per hour (an integer number of requests)
DEFAULT_BASE_RATE_PER_HOUR = 5

# Duration of a full simulated workday (minutes)
DEFAULT_WORKDAY_LENGTH = 480.0

# Normal range for the starting agent SoC values
NORMAL_SOC_RANGE = (0.8, 1.0)

# Stress range for the starting agent SoC values
STRESS_SOC_RANGE = (0.5, 0.7)

# Lead time (minutes) between a request submission and its earliest pickup: every
# earliest pickup lies in [DEFAULT_MIN_LEAD, horizon end], its lead is drawn uniformly
# from [DEFAULT_MIN_LEAD, min(DEFAULT_MAX_LEAD, earliest pickup)]
DEFAULT_MIN_LEAD = 15.0
DEFAULT_MAX_LEAD = 45.0

# Static scenario horizon (minutes): earliest pickups lie in [DEFAULT_MIN_LEAD,
# STATIC_HORIZON], all requests are submitted at t=0
STATIC_HORIZON = 60.0

# ScenarioTiming.peaks shape: PEAKS_FLOOR_SHARE of the demand spread uniformly, the
# rest within two plateaus, each PEAKS_DURATION (fraction of the horizon) wide, centered
# at PEAKS_CENTERS (fractions of the horizon)
PEAKS_FLOOR_SHARE = 0.5
PEAKS_CENTERS = (1.0 / 3.0, 2.0 / 3.0)
PEAKS_DURATION = 1.0 / 6.0

# Workday surge window length (minutes); a surge adds one hour worth of base-rate
# demand within its window, i.e., (60 / SURGE_DURATION) x the base rate on top
SURGE_DURATION = 30.0

# Minimum gap (minutes) a surge window keeps from any other surge window and from the
# peaks plateaus, per timing
SURGE_GAP_BY_TIMING: dict[ScenarioTiming, float] = {
    ScenarioTiming.uniform: 30.0,
    ScenarioTiming.peaks: 15.0,
}


def _peaks_plateaus(
    horizon_start: float, horizon_end: float, t_lb: float
) -> list[tuple[float, float]]:
    """ScenarioTiming.peaks plateaus of [horizon_start, horizon_end], clipped at t_lb"""
    span = horizon_end - horizon_start
    plateaus = []
    for center in PEAKS_CENTERS:
        mid = horizon_start + center * span
        lb = max(t_lb, mid - PEAKS_DURATION * span / 2.0)
        ub = min(horizon_end, mid + PEAKS_DURATION * span / 2.0)
        if ub > lb:
            plateaus.append((lb, ub))
    return plateaus


def timing_profile(
    timing: ScenarioTiming | str,
    horizon_start: float,
    horizon_end: float,
    t_lb: float | None = None,
) -> list[tuple[float, float, float]]:
    """
    Earliest-pickup probability density of the timing over [t_lb, horizon_end] as
    (start, end, density) pieces that overlap (additively) and integrate to 1. Peaks
    plateaus are placed relative to the full horizon; t_lb defaults to horizon_start
    """
    t_lb = horizon_start if t_lb is None else t_lb
    if not horizon_start <= t_lb < horizon_end:
        raise ValueError("expected horizon_start <= t_lb < horizon_end")
    if ScenarioTiming(timing) == ScenarioTiming.uniform:
        return [(t_lb, horizon_end, 1.0 / (horizon_end - t_lb))]
    plateaus = _peaks_plateaus(horizon_start, horizon_end, t_lb)
    p_span = sum(ub - lb for lb, ub in plateaus)
    floor_share = PEAKS_FLOOR_SHARE if p_span > 0 else 1.0
    pieces = [(t_lb, horizon_end, floor_share / (horizon_end - t_lb))]
    pieces += [(lb, ub, (1.0 - floor_share) / p_span) for lb, ub in plateaus]
    return pieces


def surge_spans(
    timing: ScenarioTiming | str,
    workday_length: float,
    min_lead: float = DEFAULT_MIN_LEAD,
) -> list[tuple[float, float]]:
    """
    Time spans within [min_lead, workday_length] where a surge window may lie, kept
    SURGE_GAP_BY_TIMING away from the peaks plateaus; the gap between surges themselves
    is enforced by _sample_surge_starts
    """
    timing = ScenarioTiming(timing)
    gap = SURGE_GAP_BY_TIMING[timing]
    plateaus = (
        _peaks_plateaus(0.0, workday_length, min_lead)
        if timing == ScenarioTiming.peaks
        else []
    )
    spans, t_start, after_plateau = [], min_lead, False
    for lb, ub in sorted(plateaus):
        spans.append((t_start + (gap if after_plateau else 0.0), lb - gap))
        t_start, after_plateau = ub, True
    spans.append((t_start + (gap if after_plateau else 0.0), workday_length))
    return [(lb, ub) for lb, ub in spans if ub - lb >= SURGE_DURATION]


def max_surges(
    timing: ScenarioTiming | str,
    workday_length: float,
    min_lead: float = DEFAULT_MIN_LEAD,
) -> int:
    """Maximum number of non-overlapping, gap-separated surges a workday can hold"""
    gap = SURGE_GAP_BY_TIMING[ScenarioTiming(timing)]
    return sum(
        int((ub - lb + gap) // (SURGE_DURATION + gap))
        for lb, ub in surge_spans(timing, workday_length, min_lead)
    )


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
        """Scenario/point name suffix, e.g., "80_100" """
        return f"{round(self.lb * 100)}_{round(self.ub * 100)}"

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

    def _sample_from_pieces(
        self, pieces: list[tuple[float, float, float]], size: int
    ) -> np.ndarray:
        """
        Sample size points from piecewise-constant (lb, ub, density) pieces that
        overlap additively: pick a piece by its mass, then a uniform point within it
        """
        masses = np.array([(ub - lb) * density for lb, ub, density in pieces])
        idx = self.rng.choice(len(pieces), size=size, p=masses / masses.sum())
        starts = np.array([pieces[i][0] for i in idx])
        ends = np.array([pieces[i][1] for i in idx])
        return starts + (ends - starts) * self.rng.uniform(0.0, 1.0, size=size)

    def _sample_surge_starts(
        self,
        nr_surges: int,
        spans: list[tuple[float, float]],
        duration: float,
        gap: float,
    ) -> list[float]:
        """
        Sample nr_surges non-overlapping surge window starts within spans, each pair
        gap apart, uniformly over all valid placements: pick how many surges each span
        holds (weighted by the volume of its placements), then spread each span's
        slack uniformly. Raises ValueError if they do not fit
        """
        if nr_surges == 0:
            return []

        def slack(span_length: float, n: int) -> float:
            return span_length - n * duration - max(0, n - 1) * gap

        span_lengths = [ub - lb for lb, ub in spans]
        capacities = [
            int((span_length + gap) // (duration + gap)) for span_length in span_lengths
        ]
        allocations = [
            counts
            for counts in itertools.product(*(range(c + 1) for c in capacities))
            if sum(counts) == nr_surges
        ]
        if not allocations:
            raise ValueError(f"{nr_surges} surges do not fit in the workday")
        weights = np.array(
            [
                math.prod(
                    slack(span_length, count) ** count / math.factorial(count)
                    for span_length, count in zip(span_lengths, counts)
                )
                for counts in allocations
            ]
        )
        if weights.sum() <= 0.0:  # every allocation is packed exactly (zero slack)
            weights = np.ones(len(allocations))
        counts = allocations[
            self.rng.choice(len(allocations), p=weights / weights.sum())
        ]
        starts = []
        for (lb, _), span_length, count in zip(spans, span_lengths, counts):
            offsets = np.sort(
                self.rng.uniform(0.0, slack(span_length, count), size=count)
            )
            starts += [
                lb + offset + j * (duration + gap) for j, offset in enumerate(offsets)
            ]
        return sorted(float(t) for t in starts)

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
        earliest_pickup: float,
        tw_length: float,
        status: RequestStatus | str = RequestStatus.new,
        cluster_centers: np.ndarray | None = None,
        cluster_spread: float = 6.0,
        use_clustered_pickup: bool = False,
        use_clustered_delivery: bool = False,
    ) -> ModemsRequest:
        """
        Generate a random request

        Args:
            load_lb: Lower bound for load
            load_ub: Upper bound for load
            earliest_pickup: Earliest pickup time (sampled by the caller)
            tw_length: Fixed pickup time-window length (omega)
            status: "new" (R_n) or "scheduled" (R_s)
            cluster_centers: Optional (x, y) centers for spatial clustering
            cluster_spread: std-dev (in coordinate units) of the spread around a center
            use_clustered_pickup: sample the pickup station using cluster_centers
            use_clustered_delivery: sample the delivery station using cluster_centers

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
        scenario_timing: ScenarioTiming | str = ScenarioTiming.uniform,
        tw_length: float = DEFAULT_TIME_WINDOW,
        nr_scheduled: int = 0,
        nr_clusters: int = 3,
        cluster_spread: float = 6.0,
        mixed_probability: float = 0.5,
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
            scenario_timing: Shape of the earliest-pickup times distribution over
                [DEFAULT_MIN_LEAD, STATIC_HORIZON] (uniform/peaks, check
                timing_profile). Defaults to ScenarioTiming.uniform
            tw_length: Fixed pickup time-window length (omega). Defaults to 5.0 minutes
            nr_scheduled: Number of requests (out of nr_requests) marked as
                scheduled (R_s) rather than new (R_n). Defaults to 0
            nr_clusters: Number of geographic cluster centers for clustered/mixed
                scenario_type. Defaults to 3
            cluster_spread: Std-dev (coordinate units) of the spread around a cluster
                center. Defaults to 6.0
            mixed_probability: Per-endpoint probability of using the clustered draw
                when scenario_type is mixed. Defaults to 0.5
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
        # e^r lies within [15, 60] in the future relative to submission time (t=0), a
        # fixed number of requests follows the timing shape (a Poisson process given
        # its count), each rounded to 0.1 minutes
        earliest_pickups = self._sample_from_pieces(
            timing_profile(scenario_timing, DEFAULT_MIN_LEAD, STATIC_HORIZON),
            nr_requests,
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
                    earliest_pickup=round(float(earliest_pickups[i]), 1),
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
        base_rate_per_hour: int = DEFAULT_BASE_RATE_PER_HOUR,
        nr_surges: int = 0,
        scenario_timing: ScenarioTiming | str = ScenarioTiming.uniform,
        scenario_type: ScenarioType | str = ScenarioType.random,
        nr_clusters: int = 3,
        cluster_spread: float = 6.0,
        mixed_probability: float = 0.5,
        tw_length: float = DEFAULT_TIME_WINDOW,
        min_lead: float = DEFAULT_MIN_LEAD,
        max_lead: float = DEFAULT_MAX_LEAD,
    ) -> list[tuple[float, ModemsRequest]]:
        """
        Generate a workday request stream as a (non-homogeneous) Poisson process over
        the earliest-pickup times in [min_lead, workday_length], with the rate:
          - base_rate_per_hour x the scenario_timing shape (check timing_profile, the
            peaks plateaus are placed relative to [0, workday_length]), so the expected
            number of requests is base_rate_per_hour x (workday_length - min_lead)/60
          - plus nr_surges surge windows of SURGE_DURATION minutes, each adding one hour
            worth of base-rate demand, placed uniformly over all valid placements: never
            overlapping each other or the peaks plateaus, and at least the timing's
            SURGE_GAP_BY_TIMING apart from both (check max_surges)
        Each request is submitted at earliest_pickup - lead, with the lead drawn
        uniformly from [min_lead, min(max_lead, earliest_pickup)], so submissions lie
        in [0, workday_length - min_lead]. Returns a list of (time_submit,
        ModemsRequest), sorted by time_submit. Raises ValueError for a base rate below
        1, a negative nr_surges, or more surges than max_surges allows
        """
        scenario_timing = ScenarioTiming(scenario_timing)
        scenario_type = ScenarioType(scenario_type)
        if base_rate_per_hour < 1:
            raise ValueError("base_rate_per_hour must be a positive integer")
        if not 0.0 <= min_lead <= max_lead:
            raise ValueError("expected 0 <= min_lead <= max_lead")
        if workday_length <= min_lead:
            raise ValueError("workday_length must exceed min_lead")
        if nr_surges < 0:
            raise ValueError("nr_surges must be non-negative")
        capacity = max_surges(scenario_timing, workday_length, min_lead)
        if nr_surges > capacity:
            raise ValueError(
                f"{nr_surges} surges exceed the maximum of {capacity} for a "
                f"{workday_length}-minute {scenario_timing} workday"
            )

        cluster_centers = (
            self.sample_cluster_centers(nr_clusters)
            if scenario_type in (ScenarioType.clustered, ScenarioType.mixed)
            else None
        )

        # rate pieces (requests per minute): base shape plus surge windows
        rate_per_min = base_rate_per_hour / 60.0
        pieces = [
            (lb, ub, density * rate_per_min * (workday_length - min_lead))
            for lb, ub, density in timing_profile(
                scenario_timing, 0.0, workday_length, min_lead
            )
        ]
        surge_starts = self._sample_surge_starts(
            nr_surges,
            surge_spans(scenario_timing, workday_length, min_lead),
            SURGE_DURATION,
            SURGE_GAP_BY_TIMING[scenario_timing],
        )
        pieces += [
            (t, t + SURGE_DURATION, rate_per_min * 60.0 / SURGE_DURATION)
            for t in surge_starts
        ]
        nr_requests = int(self.rng.poisson(sum((b - a) * h for a, b, h in pieces)))
        earliest_pickups = np.sort(self._sample_from_pieces(pieces, nr_requests))

        request_submissions: list[tuple[float, ModemsRequest]] = []
        for earliest_pickup in earliest_pickups:
            if scenario_type == ScenarioType.random:
                use_p = use_d = False
            elif scenario_type == ScenarioType.clustered:
                use_p = use_d = True
            else:
                use_p = self.rng.random() < mixed_probability
                use_d = self.rng.random() < mixed_probability
            lead = self.rng.uniform(min_lead, min(max_lead, earliest_pickup))
            request = self.generate_random_request(
                load_lb=1,
                load_ub=6,
                earliest_pickup=float(earliest_pickup),
                tw_length=tw_length,
                status=RequestStatus.new,
                cluster_centers=cluster_centers,
                cluster_spread=cluster_spread,
                use_clustered_pickup=use_p,
                use_clustered_delivery=use_d,
            )
            request_submissions.append((float(earliest_pickup - lead), request))

        request_submissions.sort(key=lambda x: x[0])
        return request_submissions
