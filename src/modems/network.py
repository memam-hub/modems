from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from .core import ModemsAgent, ModemsRequest

# Distance/speed calibration for converting (x, y) coordinate distances into travel
# times. Coordinate units are for plotting only, the actual travel times are computed
# from the Euclidean distances between the nodes, rescaled to real meters with
# METERS_PER_UNIT, converted to minutes at AVERAGE_SPEED_MS, and multiplied by a
# random variability factor between 1.1 and 1.25 (network-level variability)
METERS_PER_UNIT = 37.2
AVERAGE_SPEED_MS = 3.0


class NetworkNodeType(StrEnum):
    """Road network node type enumeration class"""

    # agent node prefix
    agent = "a"
    # request node prefix
    request = "r"
    # pickup station node
    pickup = "p"
    # delivery station node
    delivery = "d"
    # generic station node
    station = "s"
    # depot/hub node
    hub = "h"

    @staticmethod
    def is_pickup(node_type: str) -> bool:
        """Check if the node type is a pickup station"""
        return node_type == NetworkNodeType.pickup

    @staticmethod
    def is_delivery(node_type: str) -> bool:
        """Check if the node type is a delivery station"""
        return node_type == NetworkNodeType.delivery

    @staticmethod
    def is_station(node_type: str) -> bool:
        """Check if the node type is a station"""
        return node_type in (
            NetworkNodeType.station,
            NetworkNodeType.pickup,
            NetworkNodeType.delivery,
        )

    @staticmethod
    def is_hub(node_type: str) -> bool:
        """Check if the node type is a final depot/hub"""
        return node_type == NetworkNodeType.hub


class NetworkNodeName:
    """
    Road network node name handler class, all indices are 1-based. Format follows:
        - agent inital:     a_<agent_index>_<node_type>_<node_index>, e.g., a_1_h_1
        - request pickup:   r_<request_index>_p_<node_index>, e.g., r_1_p_1
        - request delivery: r_<request_index>_d_<node_index>, e.g., r_1_d_1
        - final depot/hub:  h_<hub_index>, e.g., h_1
    """

    @staticmethod
    def make_agent_node_name(index: int, node_type: str, node_index: int) -> str:
        """
        Create a node name from the agent index, its initial node type and index

        Args:
            index: 1-based index of the agent
            node_type: Type of the initial node
            node_index: 1-based index of the initial node
        """
        node_type = (
            NetworkNodeType.hub
            if NetworkNodeType.is_hub(node_type)
            else NetworkNodeType.station
        )
        return f"{NetworkNodeType.agent}_{index}_{node_type}_{node_index}"

    @staticmethod
    def make_request_pickup_node_name(index: int, node_index: int) -> str:
        """
        Create a node name from the request index and its pickup node index

        Args:
            index: 1-based index of the request
            node_index: 1-based index of the pickup node
        """
        node_name = (
            f"{NetworkNodeType.request}_{index}_{NetworkNodeType.pickup}_{node_index}"
        )
        return node_name

    @staticmethod
    def make_request_delivery_node_name(index: int, node_index: int) -> str:
        """
        Create a node name from the request index and its delivery node index

        Args:
            index: 1-based index of the request
            node_index: 1-based index of the delivery node
        """
        node_name = (
            f"{NetworkNodeType.request}_{index}_{NetworkNodeType.delivery}_{node_index}"
        )
        return node_name

    @staticmethod
    def make_hub_name(index: int) -> str:
        """
        Create a node name from the final depot index

        Args:
            index: 1-based index of the final hub
        """
        return f"{NetworkNodeType.hub}_{index}"

    @staticmethod
    def make_station_name(index: int) -> str:
        """
        Create a node name from the service station index

        Args:
            index: 1-based index of the service station
        """
        return f"{NetworkNodeType.station}_{index}"

    @staticmethod
    def get_node_index(node_name: str) -> int:
        """
        Read the (1-based) node index from the node name

        Args:
            node_name: Node name
        """
        return int(node_name.split("_")[-1])

    @staticmethod
    def get_node_type(node_name: str) -> str:
        """
        Read the node type from the node name

        Args:
            node_name: Node name
        """
        return node_name.split("_")[-2]

    @staticmethod
    def get_request_index(node_name: str) -> int:
        """
        Read the (1-based) request index from the node name

        Args:
            node_name: Node name
        """
        return int(node_name.split("_")[1])


class RoadNetwork:
    """Road Network class representing the directed graph of nodes and travel times"""

    def __init__(
        self,
        nr_hubs: int,
        nr_stations: int,
        locations: np.ndarray | None = None,
        travel_times: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Initialize the road network with hubs and stations.
        If data is provided, use it; otherwise, generate random data.

        Args:
            nr_hubs: Number of hubs in the network
            nr_stations: Number of stations in the network
            locations: Optional, predefined (x, y) node positions
            travel_times: Optional, predefined travel times matrix
            rng: Optional, seeded random generator used for random node positions
                and/or travel-time factors. Defaults to an unseeded generator (NOT
                REPRODUCIBLE!) -- always pass one explicitly for reproducible generation
        """
        if not isinstance(nr_hubs, int) or nr_hubs < 1:
            raise ValueError("nr_hubs must be a positive integer")
        if not isinstance(nr_stations, int) or nr_stations < 1:
            raise ValueError("nr_stations must be a positive integer")
        self.nr_hubs = nr_hubs
        self.nr_stations = nr_stations
        self.nr_nodes = nr_hubs + nr_stations
        self.rng = rng if rng is not None else np.random.default_rng()
        generate_nodes = locations is None
        generate_times = travel_times is None
        if locations is not None:
            locations = np.asarray(locations, dtype=float)
            if locations.shape != (self.nr_nodes, 2):
                raise ValueError("locations must have shape (nr_hubs + nr_stations, 2)")
            if not np.all(np.isfinite(locations)):
                raise ValueError("locations must contain only finite values")
            if (
                generate_times
                and np.unique(locations, axis=0).shape[0] != self.nr_nodes
            ):
                raise ValueError("physical network nodes must have unique locations")
            self.locations = locations.copy()
        if travel_times is not None:
            travel_times = np.asarray(travel_times, dtype=float)
            if travel_times.shape != (self.nr_nodes, self.nr_nodes):
                raise ValueError(
                    "travel_times must be a square matrix of size nr_hubs + nr_stations"
                )
            if not np.all(np.isfinite(travel_times)):
                raise ValueError("travel_times must contain only finite values")
            if np.any(travel_times < 0):
                raise ValueError("travel_times must be non-negative")
            if not np.allclose(np.diag(travel_times), 0.0, atol=1e-9):
                raise ValueError("travel_times must have a zero diagonal")
            off_diagonal = ~np.eye(self.nr_nodes, dtype=bool)
            if np.any(travel_times[off_diagonal] <= 0):
                raise ValueError("off-diagonal travel_times must be positive")
            # Algorithm 3's monotone pruning is valid only for shortest-path
            # travel times satisfying triangle inequality
            for intermediate in range(self.nr_nodes):
                via = (
                    travel_times[:, intermediate, None]
                    + travel_times[None, intermediate, :]
                )
                if np.any(travel_times > via + 1e-9):
                    raise ValueError(
                        "travel_times must satisfy the directed triangle inequality"
                    )
            self.travel_times = travel_times.copy()

        self._generate_random_network(generate_nodes, generate_times)

    def _generate_random_network(
        self, generate_nodes: bool = True, generate_times: bool = True
    ) -> None:
        """
        Generate a random road network for the problem.
        Positions are random in a 100x100 area between (40, 40) and (65, 65).
        Travel times are computed from Euclidean coordinate distances, rescaled
        to real meters via METERS_PER_UNIT, converted to minutes at AVERAGE_SPEED_MS,
        and multiplied by a random variability factor between 1.1 and 1.25 (constant
        on the network-level for all c_ij relationships)

        Args:
            generate_nodes: Whether to generate random node positions
            generate_times: Whether to generate random travel times
        """
        if generate_nodes:
            # generate random node positions
            self.locations = np.zeros((self.nr_nodes, 2))
            used_locations: set[tuple[float, float]] = set()
            for i in range(self.nr_nodes):
                while True:
                    x = round(self.rng.uniform(40, 65), 1)
                    y = round(self.rng.uniform(40, 65), 1)
                    if (x, y) not in used_locations:
                        used_locations.add((x, y))
                        break
                self.locations[i, :] = np.array([x, y])
        if generate_times:
            # initialize with zeros
            self.travel_times = np.zeros((self.nr_nodes, self.nr_nodes))
            # random factor between 1.1 and 1.25
            factor = self.rng.uniform(1.1, 1.25)
            # symmetrical travel times, only compute upper triangle and mirror it
            for idx in range(self.nr_nodes):
                for jdx in range(idx + 1, self.nr_nodes):
                    # Euclidean distance between nodes, in coordinate units
                    distance_units = np.linalg.norm(
                        self.locations[idx] - self.locations[jdx]
                    )
                    distance_m = distance_units * METERS_PER_UNIT
                    # Keep full precision.  Independently rounding each arc
                    # can break the triangle inequality that Algorithm 3 uses
                    # for safe monotone pruning.
                    travel_time = (distance_m / AVERAGE_SPEED_MS) / 60.0 * factor
                    self.travel_times[idx, jdx] = travel_time
                    self.travel_times[jdx, idx] = travel_time

    def get_travel_time_by_index(
        self,
        i_type: NetworkNodeType | str,
        i_idx: int,
        j_type: NetworkNodeType | str,
        j_idx: int,
    ) -> float:
        """
        Get travel time between two nodes based on their types and indices

        Args:
            i_type: Type of the first node
            i_idx: Index of the first node
            j_type: Type of the second node
            j_idx: Index of the second node
        """
        if NetworkNodeType.is_hub(i_type):
            i_index = i_idx - 1
        else:
            i_index = (i_idx - 1) + self.nr_hubs
        if NetworkNodeType.is_hub(j_type):
            j_index = j_idx - 1
        else:
            j_index = (j_idx - 1) + self.nr_hubs
        return self.travel_times[i_index, j_index]

    def get_travel_time_by_name(self, i_name: str, j_name: str) -> float:
        """
        Get travel time between two nodes based on their names

        Args:
            i_name: Name of the first node
            j_name: Name of the second node
        """
        i_type = NetworkNodeName.get_node_type(i_name)
        i_idx = NetworkNodeName.get_node_index(i_name)
        j_type = NetworkNodeName.get_node_type(j_name)
        j_idx = NetworkNodeName.get_node_index(j_name)
        return self.get_travel_time_by_index(i_type, i_idx, j_type, j_idx)

    def get_travel_times_dict(
        self,
        agents: list[ModemsAgent],
        requests: list[ModemsRequest],
    ) -> dict[tuple[str, str], float]:
        """
        Create travel times data dictionary for all node pairs

        Args:
            agents (list): List of agents in the problem
            requests (list): List of requests in the problem
        """
        # construct node names
        node_names_H0 = [a.make_node_name(i) for i, a in enumerate(agents, start=1)]
        node_names_Sp = [
            r.make_pickup_node_name(i) for i, r in enumerate(requests, start=1)
        ]
        node_names_Sd = [
            r.make_delivery_node_name(i) for i, r in enumerate(requests, start=1)
        ]
        node_names_Hf = [
            NetworkNodeName.make_hub_name(i + 1) for i in range(self.nr_hubs)
        ]
        # construct all node names
        all_node_names = node_names_H0 + node_names_Sp + node_names_Sd + node_names_Hf
        # construct travel times data dictionary for all node pairs
        travel_times_data = {}
        for i_name in all_node_names:
            for j_name in all_node_names:
                # The physical road network is directed.  Do not mirror this
                # value into (j, i): callers may provide an asymmetric travel-
                # time matrix, and the nested loop will populate that reverse
                # arc from its own physical-network entry.
                travel_times_data[(i_name, j_name)] = self.get_travel_time_by_name(
                    i_name, j_name
                )
        return travel_times_data

    def __str__(self) -> str:
        """Return a string summary of the Road Network object"""
        return (
            "RoadNetwork:\n"
            f"  nr_hubs: {self.nr_hubs}\n"
            f"  nr_stations: {self.nr_stations}\n"
        )

    @staticmethod
    def _setup_fig_axis(
        xlabel: str, ylabel: str, alpha: bool = True
    ) -> tuple[Any, Any]:
        """Create and setup a figure/axis combo"""
        font = {"family": "serif", "size": 24}
        plt.rc("font", **font)
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5, forward=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if alpha:
            ax.grid(True, alpha=0.5, zorder=-2)
        return fig, ax

    @staticmethod
    def _agent_color(idx: int) -> str:
        """Persistent colors for plotting the agent lines"""
        a_colors = [
            "k",
            "tab:red",
            "tab:purple",
            "tab:orange",
            "tab:darkgreen",
            "tab:darkblue",
        ]
        return a_colors[idx % len(a_colors)]

    @staticmethod
    def _node_color(node: str | NetworkNodeType) -> str:
        """
        Persistent colors for plotting the visited nodes. The given node is either
        the node type (no handling needed), or the node name (must be converted)
        """
        if not isinstance(node, NetworkNodeType):
            node = NetworkNodeType(NetworkNodeName.get_node_type(node))
        node_colors = {
            NetworkNodeType.agent: "maroon",
            NetworkNodeType.pickup: "tab:green",
            NetworkNodeType.delivery: "tab:blue",
            NetworkNodeType.hub: "tab:olive",
        }
        return node_colors[node]

    def plot_network(
        self,
        title: str = "",
        legend: bool = True,
        show: bool = True,
        outfile: str = "",
    ) -> None:
        """
        Plot the network nodes in different colors

        Args:
            title: Diagram title. Default is empty
            legend: Whether to create a legend for the plot. Default is True
            show: Whether to show the plot. Default is True
            outfile: File name to save the plot figure. Defaults to not saving
        """
        fig, ax = self._setup_fig_axis("x (m)", "y (m)")
        ax.set_xlim(38.0, 67.0)
        ax.set_ylim(38.0, 67.0)
        if title:
            ax.set_title(title, pad=12)
        # set plot parameters
        plot_marker_params = {
            NetworkNodeType.hub: ("darkred", "D", 220, "H"),  # hub
            NetworkNodeType.station: ("teal", "o", 360, "S"),  # station
        }
        # plot all nodes
        for i, pos in enumerate(self.locations):
            if i < self.nr_hubs:
                color, shape, size, _ = plot_marker_params[NetworkNodeType.hub]
            else:
                color, shape, size, _ = plot_marker_params[NetworkNodeType.station]
            ax.scatter(
                pos[0],
                pos[1],
                s=size,
                alpha=1.0,
                edgecolors="k",
                c=color,
                marker=shape,
            )
            # write node index
            fig_txt = ax.text(
                pos[0],
                pos[1] - 1.0,
                str(i + 1) if i < self.nr_hubs else str(i - self.nr_hubs + 1),
                bbox={"facecolor": "none", "alpha": 0},
                ha="center",
                va="center",
                color="w",
                fontweight="normal",
            )
            fig_txt.set_path_effects(
                [
                    path_effects.Stroke(linewidth=3, foreground="black"),
                    path_effects.Normal(),
                ]
            )
        if legend:
            # add legend
            legend_elements: list[Line2D] = [
                Line2D(
                    [0],
                    [1],
                    color="w",
                    markersize=16 if shape == "D" else 20,
                    markeredgecolor="k",
                    alpha=1.0,
                    label=name,
                    marker=shape,
                    markerfacecolor=color,
                )
                for color, shape, _, name in plot_marker_params.values()
            ]
            ax.legend(
                handletextpad=0.1,
                handles=legend_elements,
                loc="best",
                framealpha=0.7,
                edgecolor="k",
                facecolor="w",
            )
        if outfile:
            plt.savefig(outfile, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        plt.close(fig)

    def plot_routes(
        self,
        routes: dict[str, list[str]],
        title: str = "",
        legend: bool = True,
        show: bool = True,
        outfile: str = "",
    ) -> None:
        """
        Plot the network nodes and the agent routes between them in different colors

        Args:
            routes: agent routes in the form of {agent_name: [node_names]}
            title: Diagram title. Default is empty
            legend: Whether to create a legend for the plot. Default is False
            show: Whether to show the plot. Default is True.
            outfile: File name to save the plot figure. Defaults to not saving
        """

        def _get_0_based_index(node_name: str) -> int:
            # node name indices are 1-based, convert to 0-based
            node_index = NetworkNodeName.get_node_index(node_name) - 1
            if not NetworkNodeType.is_hub(NetworkNodeName.get_node_type(node_name)):
                node_index += self.nr_hubs
            return node_index

        fig, ax = self._setup_fig_axis("x (m)", "y (m)")
        fig_xlim, fig_ylim = [38.0, 67.0], [38.0, 67.0]
        ax.set_xlim(fig_xlim[0], fig_xlim[1])
        ax.set_ylim(fig_ylim[0], fig_ylim[1])
        if title:
            ax.set_title(title, pad=12)
        # get visited nodes and their type (agent start, pickup, delivery, final hub)
        visited_nodes: list[list[NetworkNodeType]] = [[] for _ in range(self.nr_nodes)]
        for route in routes.values():
            if not route:
                continue
            # first node is the agent start node
            node_idx = _get_0_based_index(route[0])
            if NetworkNodeType.agent not in visited_nodes[node_idx]:
                visited_nodes[node_idx].append(NetworkNodeType.agent)
            # check the rest of the nodes
            for node_name in route[1:]:
                node_idx = _get_0_based_index(node_name)
                node_type = NetworkNodeName.get_node_type(node_name)
                if node_type not in visited_nodes[node_idx]:
                    visited_nodes[node_idx].append(NetworkNodeType(node_type))

        # plot hub markers
        for idx, node_type in enumerate(visited_nodes[: self.nr_hubs]):
            m_zorder = 1
            # special case: both agent start and final hub
            if len(node_type) == 2:
                v_sorted = sorted(node_type)
                ax.plot(
                    self.locations[idx][0],
                    self.locations[idx][1],
                    fillstyle="right",
                    marker="D",
                    mec="k",
                    mfc=self._node_color(v_sorted[0]),
                    mfcalt=self._node_color(v_sorted[1]),
                    markersize=16,
                    alpha=1.0,
                    zorder=m_zorder,
                )
                continue
            # single visit or no visit
            if len(node_type) == 1:
                m_color = self._node_color(node_type[0])
            else:
                m_color = "dimgray"
                m_zorder = -1
            ax.scatter(
                self.locations[idx][0],
                self.locations[idx][1],
                edgecolors="k",
                alpha=1.0,
                color=m_color,
                zorder=m_zorder,
                s=220,
                marker="D",
            )

        # plot station markers
        for idx, node_type in enumerate(visited_nodes[self.nr_hubs :]):
            node_idx = idx + self.nr_hubs
            m_zorder = 1
            # special case: three types visited
            if len(node_type) == 3:
                # draw pie chart as marker
                colors = [
                    self._node_color(c)
                    for c in (
                        NetworkNodeType.agent,
                        NetworkNodeType.pickup,
                        NetworkNodeType.delivery,
                    )
                ]

                def _get_pie_position(pos, size=0.034):
                    x_norm = (pos[0] - fig_xlim[0]) / (fig_xlim[1] - fig_xlim[0])
                    y_norm = (pos[1] - fig_ylim[0]) / (fig_ylim[1] - fig_ylim[0])
                    ax_pos = ax.get_position()
                    x_norm = ax_pos.x0 + x_norm * (ax_pos.x1 - ax_pos.x0)
                    y_norm = ax_pos.y0 + y_norm * (ax_pos.y1 - ax_pos.y0)
                    left = x_norm - size / 2
                    bottom = y_norm - size / 2
                    return [left, bottom, size, size]  # left, bottom, width, height

                ax_pie = fig.add_axes(_get_pie_position(self.locations[node_idx]))
                ax_pie.pie(
                    [1, 1, 1],
                    colors=colors,
                    startangle=90,
                    wedgeprops={"edgecolor": "k"},
                )
                continue

            # special case: two types visited
            if len(node_type) == 2:
                v_sorted = sorted(node_type)
                ax.plot(
                    self.locations[node_idx][0],
                    self.locations[node_idx][1],
                    fillstyle="right",
                    marker="o",
                    mec="k",
                    mfc=self._node_color(v_sorted[0]),
                    mfcalt=self._node_color(v_sorted[1]),
                    markersize=20,
                    alpha=1.0,
                    zorder=m_zorder,
                )
                continue
            # single visit or no visit
            if len(node_type) == 1:
                m_color = self._node_color(node_type[0])
            else:
                m_color = "lightgray"
                m_zorder = -1
            ax.scatter(
                self.locations[node_idx][0],
                self.locations[node_idx][1],
                edgecolors="k",
                alpha=1.0,
                color=m_color,
                zorder=m_zorder,
                s=360,
                marker="o",
            )
        # write node indices
        for idx in range(self.nr_nodes):
            fig_txt = ax.text(
                self.locations[idx][0],
                self.locations[idx][1] - 1.0,
                str(idx + 1) if idx < self.nr_hubs else str(idx - self.nr_hubs + 1),
                bbox={"facecolor": "none", "alpha": 0},
                ha="center",
                va="center",
                color="w",
                fontweight="normal",
                zorder=2,
            )
            fig_txt.set_path_effects(
                [
                    path_effects.Stroke(linewidth=3, foreground="black"),
                    path_effects.Normal(),
                ]
            )
        # plot route as lines between nodes
        for idx, route in enumerate(routes.values()):
            a_color = self._agent_color(idx)
            for node_idx in range(len(route) - 1):
                x1, y1 = self.locations[_get_0_based_index(route[node_idx])]
                x2, y2 = self.locations[_get_0_based_index(route[node_idx + 1])]
                ax.plot([x1, x2], [y1, y2], c=a_color, linewidth=2.0, zorder=-1)
                ax.arrow(
                    x1,
                    y1,
                    0.4 * (x2 - x1),
                    0.4 * (y2 - y1),
                    head_width=0.2,
                    head_length=0.4,
                    ec=a_color,
                    fc=a_color,
                    zorder=0,
                )

        # set plot parameters
        plot_marker_params = {
            "hubs": ("white", "k", "D", "H"),  # hub
            "stations": ("white", "k", "o", "S"),  # station
            "H_0": (self._node_color(NetworkNodeType.agent), "none", "s", r"$H_0$"),
            "S_p": (self._node_color(NetworkNodeType.pickup), "none", "s", r"$S^p$"),
            "S_d": (self._node_color(NetworkNodeType.delivery), "none", "s", r"$S^d$"),
            "H_f": (self._node_color(NetworkNodeType.hub), "none", "s", r"$H_f$"),
        }
        if legend:
            # add legend
            legend_elements: list[Line2D] = [
                Line2D(
                    [0],
                    [1],
                    color="w",
                    markersize=16 if shape == "D" else 20,
                    markerfacecolor=face_color,
                    markeredgecolor=edge_color,
                    marker=shape,
                    label=name,
                    alpha=1.0,
                )
                for face_color, edge_color, shape, name in plot_marker_params.values()
            ]
            ax.legend(
                handletextpad=0.1,
                handles=legend_elements,
                loc="best",
                framealpha=0.7,
                edgecolor="k",
                facecolor="w",
            )
        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        plt.close(fig)
