from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

import numpy as np


class NodeState(str, Enum):
    """MAC/PHY states used by the discrete-event simulator."""

    IDLE = "IDLE"
    AIFS = "AIFS"
    BACKOFF = "BACKOFF"
    BACKOFF_FROZEN = "BACKOFF FROZEN"
    DL_TX = "DL TX"
    COBF_TX = "CO-BF TX"
    COORDINATION = "COORDINATION"
    ACK = "ACK"
    RECEIVING = "RECEIVING"


@dataclass
class Node:
    id: int
    x: float
    y: float
    is_ap: bool
    array_orientation_rad: float = 0.0
    cw: int = 15
    retry_count: int = 0
    backoff_slots: int = 0
    state: NodeState = NodeState.IDLE
    state_start_time_us: float = 0.0
    backoff_countdown_start_us: float = 0.0
    pending_event_id: int = 0
    # Virtual carrier sensing (Network Allocation Vector).
    nav_end_time_us: float = 0.0
    nav_event_id: int = 0


@dataclass
class AP(Node):
    associated_stas: List[int] = field(default_factory=list)


@dataclass
class STA(Node):
    serving_ap: int = -1


@dataclass
class ActiveTransmission:
    """One active downlink spatial stream."""

    tx_id: int
    txop_id: int
    tx_node_id: int
    rx_node_id: int
    is_cobf: bool
    cluster_id: int
    start_time_us: float
    channels: Dict[int, np.ndarray]
    beamforming_vector: np.ndarray
    tx_power_mw: float
    selected_mcs: int = 0
    phy_rate_mbps: float = 0.0
    required_sinr_db: float = 0.0
    initial_sinr_db: float = 0.0
    min_sinr_linear: float = 1e9


@dataclass
class CoordinationReservation:
    """CCA-only reservation footprint during Co-BF coordination."""

    cluster_id: int
    tx_node_id: int
    channels: Dict[int, np.ndarray]
    tx_power_mw: float


@dataclass
class TransmissionRecord:
    """Completed downlink stream used for metrics and debugging."""

    tx_id: int
    txop_id: int
    tx_id_node: int
    rx_id_node: int
    is_cobf: bool
    cluster_id: int
    start_time_us: float
    end_time_us: float
    selected_mcs: int
    phy_rate_mbps: float
    required_sinr_db: float
    initial_sinr_db: float
    min_sinr_linear: float
    tx_power_mw: float
    payload_bytes: int
    success: bool


@dataclass
class SnapshotStream:
    """Immutable copy of one active stream for post-run visualization."""

    tx_id: int
    txop_id: int
    tx_node_id: int
    rx_node_id: int
    is_cobf: bool
    cluster_id: int
    tx_power_mw: float
    beamforming_vector: np.ndarray
    initial_sinr_db: float
    snapshot_sinr_db: float


@dataclass
class TxopSnapshot:
    """Representative instantaneous set of active streams."""

    mode: str
    time_us: float
    focus_txop_id: int
    focus_cluster_id: int
    streams: List[SnapshotStream]
    precoder_scheme: str = "zf"
    max_ici_penalty_db: float = 0.0
    avg_ici_penalty_db: float = 0.0


@dataclass
class TimingEvent:
    node_id: int
    state: str
    start_ms: float
    end_ms: float


class Topology:
    """Random AP/STA topology with nearest-eligible-AP association.

    Array orientations are global azimuths in radians. By default all ULAs are
    parallel to the global x axis. Optional random orientations are useful for
    sensitivity studies but are disabled by default to make beam plots easier
    to interpret.
    """

    def __init__(
        self,
        num_aps: int = 10,
        num_stas: int = 30,
        area_width: float = 100.0,
        area_height: float = 100.0,
        association_radius: float = 30.0,
        seed: int = 12345,
        randomize_array_orientations: bool = False,
    ) -> None:
        self.num_aps = int(num_aps)
        self.num_stas = int(num_stas)
        self.area_width = float(area_width)
        self.area_height = float(area_height)
        self.association_radius = float(association_radius)
        self.randomize_array_orientations = bool(randomize_array_orientations)
        self.seed = int(seed)

        self.aps: List[AP] = []
        self.stas: List[STA] = []
        self.nodes: Dict[int, Node] = {}
        self.distances = np.zeros(
            (self.num_aps + self.num_stas, self.num_aps + self.num_stas),
            dtype=float,
        )

        self._generate_valid_topology()

    def _generate_valid_topology(self) -> None:
        """Generate a topology in which every STA has an eligible serving AP."""
        rng = np.random.default_rng(self.seed)

        for _ in range(1000):
            ap_x = rng.uniform(0.0, self.area_width, self.num_aps)
            ap_y = rng.uniform(0.0, self.area_height, self.num_aps)
            sta_x = rng.uniform(0.0, self.area_width, self.num_stas)
            sta_y = rng.uniform(0.0, self.area_height, self.num_stas)

            sta_associations: List[int] = []
            valid = True

            for sta_index in range(self.num_stas):
                distances = np.hypot(
                    ap_x - sta_x[sta_index],
                    ap_y - sta_y[sta_index],
                )
                eligible = np.where(distances <= self.association_radius)[0]
                if len(eligible) == 0:
                    valid = False
                    break

                # Nearest eligible AP is a clearer baseline than random
                # association and avoids avoidable weak serving links.
                nearest_local_index = int(np.argmin(distances[eligible]))
                sta_associations.append(int(eligible[nearest_local_index]))

            if not valid:
                continue

            if self.randomize_array_orientations:
                ap_orientation = rng.uniform(-np.pi, np.pi, self.num_aps)
                sta_orientation = rng.uniform(-np.pi, np.pi, self.num_stas)
            else:
                ap_orientation = np.zeros(self.num_aps, dtype=float)
                sta_orientation = np.zeros(self.num_stas, dtype=float)

            self.aps = []
            self.stas = []
            self.nodes = {}

            for ap_id in range(self.num_aps):
                ap = AP(
                    id=ap_id,
                    x=float(ap_x[ap_id]),
                    y=float(ap_y[ap_id]),
                    is_ap=True,
                    array_orientation_rad=float(ap_orientation[ap_id]),
                )
                self.aps.append(ap)
                self.nodes[ap_id] = ap

            for sta_index in range(self.num_stas):
                sta_id = self.num_aps + sta_index
                serving_ap = sta_associations[sta_index]
                sta = STA(
                    id=sta_id,
                    x=float(sta_x[sta_index]),
                    y=float(sta_y[sta_index]),
                    is_ap=False,
                    array_orientation_rad=float(sta_orientation[sta_index]),
                    serving_ap=serving_ap,
                )
                self.stas.append(sta)
                self.nodes[sta_id] = sta
                self.aps[serving_ap].associated_stas.append(sta_id)

            all_x = np.concatenate([ap_x, sta_x])
            all_y = np.concatenate([ap_y, sta_y])
            self.distances = np.hypot(
                all_x[:, None] - all_x[None, :],
                all_y[:, None] - all_y[None, :],
            )
            return

        raise RuntimeError(
            "Could not generate a valid topology in 1000 attempts. "
            "Increase association_radius or adjust node counts/area size."
        )

    def get_distance(self, node_i: int, node_j: int) -> float:
        return float(self.distances[node_i, node_j])
