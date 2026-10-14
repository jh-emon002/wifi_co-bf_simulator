from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from wifi_sim.beamforming import (
    calculate_cluster_precoders,
    InfeasiblePrecoderError,
    calculate_mrt_beamformer,
    calculate_sinr_dl,
    MIN_MCS_SINR_DB,
    select_mcs,
)
from wifi_sim.channel import WirelessChannel
from wifi_sim.config import Config
from wifi_sim.metrics import SimulationMetrics, compute_jain_fairness
from wifi_sim.models import (
    AP,
    ActiveTransmission,
    CoordinationReservation,
    Node,
    NodeState,
    SnapshotStream,
    TimingEvent,
    Topology,
    TransmissionRecord,
    TxopSnapshot,
)


class SimulationMode(str):
    MODE_A = "A"
    MODE_B = "B"
    MODE_C = "C"


MAX_COBF_USERS_PER_AP = 2


class Simulator:
    """Discrete-event, downlink-only Wi-Fi multi-AP simulator.

    Mode A
        AP-only CSMA/CA. The winning AP serves one randomly selected STA with MRT.

    Mode B
        The winning AP recruits the nearest eligible free AP. Each AP serves up
        to two randomly selected STAs. All scheduled streams use cluster-wide ZF.
        Interference from other simultaneously active clusters is included.

    Mode C
        Same MAC, CCA, clustering, power and beamforming as Mode B. Only the
        receiver SINR model is idealized: a clustered STA ignores AP streams
        outside its serving cluster.
    """

    def __init__(
        self,
        topology: Topology,
        channel: WirelessChannel,
        config: Config,
        mode: str = SimulationMode.MODE_A,
        seed: int = 12345,
    ) -> None:
        if mode not in {
            SimulationMode.MODE_A,
            SimulationMode.MODE_B,
            SimulationMode.MODE_C,
        }:
            raise ValueError(f"Unknown simulation mode: {mode}")

        self.topology = topology
        self.channel = channel
        self.config = config
        self.mode = mode
        self.rng = np.random.default_rng(seed)

        self.current_time_us = 0.0
        self.sim_time_us = self.config.simulation.sim_time_s * 1e6
        self.timing_window_us = self.config.simulation.timing_window_ms * 1e3

        # Heap entries: (time_us, priority, unique_id, handler_name, payload)
        self.event_queue: List[Tuple[float, int, int, str, dict]] = []
        self._event_counter = 0

        self.active_transmissions: Dict[int, ActiveTransmission] = {}
        self.active_clusters: Dict[int, Set[int]] = {}

        # CCA-only control/reservation footprints used during the Co-BF
        # coordination interval. Keyed by AP ID because one AP can belong to at
        # most one active cluster.
        self.active_coordination_reservations: Dict[int, CoordinationReservation] = {}

        self.tx_id_counter = 0
        self.txop_id_counter = 0
        self.cluster_id_counter = 0

        self.completed_transmissions: List[TransmissionRecord] = []
        self.timing_events: List[TimingEvent] = []

        # One representative instantaneous data-TX snapshot is kept for
        # optional post-run beam/SINR visualization. It never affects the
        # event engine or any reported metric.
        self.visualization_snapshot: Optional[TxopSnapshot] = None
        self._visualization_snapshot_score: Optional[Tuple[float, float, int, int]] = None
        self.two_cluster_snapshot_candidates = 0
        self.precoder_infeasible_txops = 0
        self.reduced_stream_txops = 0

        # Time integrals for time-weighted average concurrency metrics.
        self.last_integral_update_us = 0.0
        self.simultaneous_ap_time_integral = 0.0
        self.active_cluster_time_integral = 0.0

    # ------------------------------------------------------------------
    # Event/timing helpers
    # ------------------------------------------------------------------
    def schedule_event(
        self,
        time_us: float,
        priority: int,
        handler_name: str,
        payload: dict,
    ) -> None:
        self._event_counter += 1
        heapq.heappush(
            self.event_queue,
            (time_us, priority, self._event_counter, handler_name, payload),
        )

    def _log_timing_event(
        self,
        node_id: int,
        state: NodeState,
        start_us: float,
        end_us: float,
    ) -> None:
        if start_us >= self.timing_window_us:
            return
        clamped_end = min(end_us, self.timing_window_us)
        if clamped_end <= start_us:
            return
        self.timing_events.append(
            TimingEvent(
                node_id=node_id,
                state=state.value,
                start_ms=start_us / 1000.0,
                end_ms=clamped_end / 1000.0,
            )
        )

    def _set_node_state(self, node: Node, new_state: NodeState) -> None:
        now = self.current_time_us
        if node.state_start_time_us < now:
            self._log_timing_event(
                node.id,
                node.state,
                node.state_start_time_us,
                now,
            )

        node.state = new_state
        node.state_start_time_us = now
        # Existing AIFS/backoff timers become stale after any state transition.
        node.pending_event_id += 1

    def _update_integrals(self) -> None:
        dt = self.current_time_us - self.last_integral_update_us
        if dt <= 0.0:
            return

        active_ap_ids = {
            tx.tx_node_id for tx in self.active_transmissions.values()
        }
        self.simultaneous_ap_time_integral += len(active_ap_ids) * dt
        self.active_cluster_time_integral += len(self.active_clusters) * dt
        self.last_integral_update_us = self.current_time_us

    # ------------------------------------------------------------------
    # Channel / CCA / SINR helpers
    # ------------------------------------------------------------------
    def _get_tx_channel(
        self,
        tx: ActiveTransmission,
        rx_id: int,
    ) -> np.ndarray:
        """Get the same fixed LOS AP-to-receiver channel for every stream."""
        if rx_id not in tx.channels:
            tx.channels[rx_id] = self.channel.sample_channel_matrix(
                rx_id,
                tx.tx_node_id,
            )
        return tx.channels[rx_id]

    def _get_reservation_channel(
        self,
        reservation: CoordinationReservation,
        rx_id: int,
    ) -> np.ndarray:
        """Lazily sample the AP->observer channel for a coordination reservation."""
        if rx_id not in reservation.channels:
            reservation.channels[rx_id] = self.channel.sample_channel_matrix(
                rx_id,
                reservation.tx_node_id,
            )
        return reservation.channels[rx_id]

    def _activate_coordination_reservations(
        self,
        cluster_id: int,
        aps: Tuple[AP, AP],
    ) -> None:
        """Reserve the sensed medium for the whole Co-BF coordination interval.

        Coordination happens before the data ZF beams exist. Therefore its CCA
        footprint is intentionally non-beamformed. The received control energy
        at an observer is modeled as ``P_AP / N_t * ||H||_F^2``.
        """
        for ap in aps:
            self.active_coordination_reservations[ap.id] = CoordinationReservation(
                cluster_id=cluster_id,
                tx_node_id=ap.id,
                channels={},
                tx_power_mw=self.channel.ap_tx_power_mw,
            )

    def _clear_coordination_reservations(self, cluster_id: int) -> None:
        """Remove CCA reservations for a cluster immediately before data starts."""
        for ap_id in [
            ap_id
            for ap_id, reservation in self.active_coordination_reservations.items()
            if reservation.cluster_id == cluster_id
        ]:
            self.active_coordination_reservations.pop(ap_id, None)

    # ------------------------------------------------------------------
    # Virtual carrier sensing (NAV)
    # ------------------------------------------------------------------
    def is_nav_active(self, node_id: int) -> bool:
        """Return True while an AP's Network Allocation Vector is active."""
        if not self.config.mac.nav_enabled:
            return False
        node = self.topology.nodes[node_id]
        return self.current_time_us < node.nav_end_time_us

    def _set_nav(self, ap: AP, duration_us: float) -> None:
        """Set or extend NAV; a later Duration field can never shorten it."""
        if not self.config.mac.nav_enabled or duration_us <= 0.0:
            return

        new_end_time_us = self.current_time_us + float(duration_us)
        if new_end_time_us <= ap.nav_end_time_us + 1e-12:
            return

        ap.nav_end_time_us = new_end_time_us
        ap.nav_event_id += 1
        nav_event_id = ap.nav_event_id
        self.schedule_event(
            new_end_time_us,
            0,
            "_on_nav_expired",
            {"node_id": ap.id, "nav_event_id": nav_event_id},
        )

    def _on_nav_expired(self, payload: dict) -> None:
        """Release virtual carrier sensing when the latest NAV expires."""
        ap = self.topology.aps[payload["node_id"]]
        if payload["nav_event_id"] != ap.nav_event_id:
            return
        if self.current_time_us + 1e-12 < ap.nav_end_time_us:
            return
        ap.nav_end_time_us = 0.0
        self._update_all_nodes_cca()

    @staticmethod
    def _ap_can_decode_nav(ap: AP, excluded_ap_ids: Set[int]) -> bool:
        """Half-duplex guard for APs eligible to decode a NAV control frame."""
        if ap.id in excluded_ap_ids:
            return False
        return ap.state not in {
            NodeState.DL_TX,
            NodeState.COBF_TX,
            NodeState.COORDINATION,
            NodeState.ACK,
        }

    def _omnidirectional_nav_rx_power_mw(
        self,
        sender_ap_id: int,
        observer_ap_id: int,
    ) -> float:
        """Received power of an omnidirectional NAV/control frame.

        No MRT/ZF beam exists for NAV. The total AP power is distributed across
        the transmit antennas, giving P_AP/N_t * ||H||_F^2. Here H is the same
        geometry-based AP-to-AP multipath MIMO channel used everywhere else in
        Version 2, so NAV reception is angle/spatially consistent too.
        """
        channel = self.channel.sample_channel_matrix(observer_ap_id, sender_ap_id)
        n_tx = channel.shape[1]
        return (
            self.channel.ap_tx_power_mw
            / n_tx
            * float(np.linalg.norm(channel, "fro") ** 2)
        )

    def _nav_external_interference_mw(
        self,
        observer_ap_id: int,
        excluded_ap_ids: Set[int],
    ) -> float:
        """External RF interference while an observer AP decodes NAV."""
        interference_mw = 0.0

        for tx in self.active_transmissions.values():
            if tx.tx_node_id in excluded_ap_ids:
                continue
            channel = self._get_tx_channel(tx, observer_ap_id)
            beamformed_signal = channel @ tx.beamforming_vector
            interference_mw += tx.tx_power_mw * float(
                np.real(np.vdot(beamformed_signal, beamformed_signal))
            )

        for reservation in self.active_coordination_reservations.values():
            if reservation.tx_node_id in excluded_ap_ids:
                continue
            channel = self._get_reservation_channel(reservation, observer_ap_id)
            n_tx = channel.shape[1]
            interference_mw += (
                reservation.tx_power_mw
                / n_tx
                * float(np.linalg.norm(channel, "fro") ** 2)
            )

        return interference_mw

    def _calculate_nav_sinr_db(
        self,
        sender_ap_id: int,
        observer_ap_id: int,
        excluded_ap_ids: Set[int],
    ) -> float:
        """SINR of one omnidirectional NAV advertisement at an observer AP."""
        signal_mw = self._omnidirectional_nav_rx_power_mw(
            sender_ap_id,
            observer_ap_id,
        )
        interference_mw = self._nav_external_interference_mw(
            observer_ap_id,
            excluded_ap_ids,
        )
        # The signal and interference expressions sum receive-antenna energy,
        # so sum thermal noise over the observer AP's receive antennas too.
        noise_mw = self.channel.noise_mw * self.config.rf.ap_antennas
        sinr_linear = signal_mw / max(noise_mw + interference_mw, 1e-18)
        return 10.0 * math.log10(max(sinr_linear, 1e-12))

    def _advertise_omnidirectional_nav(
        self,
        advertising_ap_ids: Set[int],
        duration_us: float,
    ) -> None:
        """Advertise NAV omnidirectionally; decode using the minimum MCS SINR.

        For a two-AP Co-BF TXOP the control advertisements are treated as
        sequential frames, so the two cooperating APs do not interfere with one
        another while the Duration field is decoded. A neighboring AP sets NAV
        when at least one advertisement has SINR >= MIN_MCS_SINR_DB.
        """
        if not self.config.mac.nav_enabled or duration_us <= 0.0:
            return

        for observer in self.topology.aps:
            if not observer.associated_stas:
                continue
            if not self._ap_can_decode_nav(observer, advertising_ap_ids):
                continue

            for sender_ap_id in sorted(advertising_ap_ids):
                nav_sinr_db = self._calculate_nav_sinr_db(
                    sender_ap_id=sender_ap_id,
                    observer_ap_id=observer.id,
                    excluded_ap_ids=advertising_ap_ids,
                )
                if nav_sinr_db >= MIN_MCS_SINR_DB:
                    self._set_nav(observer, duration_us)
                    break

    def get_aggregate_rx_power_mw(self, node_id: int) -> float:
        """Aggregate RF power used for carrier sensing.

        Data streams contribute ``P_stream * ||H w||^2`` using their actual
        complex MIMO channel and beamformer. Coordination occurs before ZF data
        beams exist, so each coordination reservation contributes non-beamformed
        control energy ``P_AP / N_t * ||H||_F^2``. Carrier sensing is identical
        in Modes A, B and C.
        """
        total_power_mw = 0.0

        for tx in self.active_transmissions.values():
            if tx.tx_node_id == node_id:
                continue

            channel = self._get_tx_channel(tx, node_id)
            beamformed_signal = channel @ tx.beamforming_vector
            total_power_mw += tx.tx_power_mw * float(
                np.real(np.vdot(beamformed_signal, beamformed_signal))
            )

        for reservation in self.active_coordination_reservations.values():
            if reservation.tx_node_id == node_id:
                continue

            channel = self._get_reservation_channel(reservation, node_id)
            n_tx = channel.shape[1]
            total_power_mw += (
                reservation.tx_power_mw
                / n_tx
                * float(np.linalg.norm(channel, "fro") ** 2)
            )

        return total_power_mw

    def is_channel_busy(self, node_id: int) -> bool:
        # Virtual carrier sensing supplements, rather than replaces, physical
        # beamforming-aware CCA. Mode C never changes CCA or NAV behavior.
        if self.is_nav_active(node_id):
            return True
        received_mw = self.get_aggregate_rx_power_mw(node_id)
        if received_mw <= 0.0:
            return False
        received_dbm = 10.0 * math.log10(received_mw)
        return received_dbm >= self.config.mac.cca_threshold_dbm

    def _recalculate_ongoing_sinrs(self) -> None:
        active_list = list(self.active_transmissions.values())
        for tx in active_list:
            sinr_linear, _ = calculate_sinr_dl(
                desired_tx_id=tx.tx_id,
                rx_sta_id=tx.rx_node_id,
                get_channel_fn=self._get_tx_channel,
                active_txs=active_list,
                mode=self.mode,
                serving_cluster_id=tx.cluster_id,
                p_noise_mw=self.channel.noise_mw,
            )
            tx.min_sinr_linear = min(tx.min_sinr_linear, sinr_linear)

    def _consider_visualization_snapshot(self) -> None:
        """Save the real active-data state only when two Co-BF clusters overlap.

        Never fabricate concurrent transmissions or defeat NAV/CCA to obtain
        an aesthetically pleasing plot. Score each qualifying focus cluster by
        the peak and mean instantaneous *counterfactual SINR loss* from the
        concurrently active external streams. Both clusters must each have
        two actually transmitting APs. Mode A keeps a normal AP-only snapshot.
        """
        active = list(self.active_transmissions.values())
        if not active:
            return

        by_cluster: Dict[int, List[ActiveTransmission]] = {}
        by_txop: Dict[int, List[ActiveTransmission]] = {}
        for tx in active:
            by_txop.setdefault(tx.txop_id, []).append(tx)
            if tx.is_cobf and tx.cluster_id >= 0:
                by_cluster.setdefault(tx.cluster_id, []).append(tx)

        truly_active_clusters = {
            cluster_id: txs for cluster_id, txs in by_cluster.items()
            if len({tx.tx_node_id for tx in txs}) == 2
        }
        if self.mode in ('B', 'C') and self.config.visualization.require_two_cobf_clusters:
            if len(truly_active_clusters) < 2:
                return
        if self.mode in ('B', 'C') and len(truly_active_clusters) >= 2:
            self.two_cluster_snapshot_candidates += 1

        for focus_txop_id, focus_streams in by_txop.items():
            if self.mode in ('B', 'C') and len(truly_active_clusters) >= 2:
                if focus_streams[0].cluster_id not in truly_active_clusters:
                    continue
                ext = [t for t in active if t.cluster_id != focus_streams[0].cluster_id]
                if not ext:
                    continue
            else:
                ext = [t for t in active if t.txop_id != focus_txop_id]

            penalties = []
            for stream in focus_streams:
                h_des = self._get_tx_channel(stream, stream.rx_node_id)
                signal = stream.tx_power_mw * float(
                    abs((h_des @ stream.beamforming_vector).item()) ** 2)
                intra = 0.0
                external = 0.0
                for interferer in active:
                    if interferer.tx_id == stream.tx_id:
                        continue
                    hi = self._get_tx_channel(interferer, stream.rx_node_id)
                    power = interferer.tx_power_mw * float(
                        abs((hi @ interferer.beamforming_vector).item()) ** 2)
                    if interferer.cluster_id == stream.cluster_id:
                        intra += power
                    else:
                        external += power
                # Exactly the SINR-dB degradation due to the extra ICI term.
                penalty = 10.0 * math.log10(
                    (self.channel.noise_mw + intra + external)
                    / max(self.channel.noise_mw + intra, 1e-18)
                )
                penalties.append(penalty)

            peak = max(penalties) if penalties else 0.0
            mean = sum(penalties) / len(penalties) if penalties else 0.0
            score = (peak, mean, len(focus_streams), len(ext))
            if (self._visualization_snapshot_score is not None
                    and score <= self._visualization_snapshot_score):
                continue

            snapshot_sinr_db: Dict[int, float] = {}
            for tx in active:
                _, actual_sinr = calculate_sinr_dl(
                    desired_tx_id=tx.tx_id,
                    rx_sta_id=tx.rx_node_id,
                    get_channel_fn=self._get_tx_channel,
                    active_txs=active,
                    mode=self.mode,
                    serving_cluster_id=tx.cluster_id,
                    p_noise_mw=self.channel.noise_mw,
                )
                snapshot_sinr_db[tx.tx_id] = actual_sinr
            snapshot_streams = [
                SnapshotStream(
                    tx_id=tx.tx_id,
                    txop_id=tx.txop_id,
                    tx_node_id=tx.tx_node_id,
                    rx_node_id=tx.rx_node_id,
                    is_cobf=tx.is_cobf,
                    cluster_id=tx.cluster_id,
                    tx_power_mw=tx.tx_power_mw,
                    beamforming_vector=np.array(tx.beamforming_vector, copy=True),
                    initial_sinr_db=tx.initial_sinr_db,
                    snapshot_sinr_db=snapshot_sinr_db[tx.tx_id],
                ) for tx in active
            ]
            self.visualization_snapshot = TxopSnapshot(
                mode=self.mode,
                time_us=self.current_time_us,
                focus_txop_id=focus_txop_id,
                focus_cluster_id=focus_streams[0].cluster_id,
                streams=snapshot_streams,
                precoder_scheme=self.config.beamforming.scheme,
                max_ici_penalty_db=peak,
                avg_ici_penalty_db=mean,
            )
            self._visualization_snapshot_score = score

    # ------------------------------------------------------------------
    # CSMA/CA
    # ------------------------------------------------------------------
    def _update_all_nodes_cca(self) -> None:
        """Freeze/resume AP backoff when the sensed channel changes."""
        for ap in self.topology.aps:
            # APs without associated users do not contend.
            if not ap.associated_stas:
                continue

            if ap.state in {
                NodeState.DL_TX,
                NodeState.COBF_TX,
                NodeState.COORDINATION,
                NodeState.ACK,
            }:
                continue

            busy = self.is_channel_busy(ap.id)

            if busy:
                if ap.state == NodeState.BACKOFF:
                    elapsed_us = self.current_time_us - ap.backoff_countdown_start_us
                    completed_slots = int(elapsed_us // self.config.mac.slot_time_us)
                    ap.backoff_slots = max(0, ap.backoff_slots - completed_slots)
                    self._set_node_state(ap, NodeState.BACKOFF_FROZEN)
                elif ap.state in {NodeState.AIFS, NodeState.IDLE}:
                    self._set_node_state(ap, NodeState.BACKOFF_FROZEN)
                continue

            if ap.state in {NodeState.BACKOFF_FROZEN, NodeState.IDLE}:
                self._set_node_state(ap, NodeState.AIFS)
                event_id = ap.pending_event_id
                self.schedule_event(
                    self.current_time_us + self.config.mac.aifs_us,
                    2,
                    "_on_aifs_expired",
                    {"node_id": ap.id, "event_id": event_id},
                )

    def _on_aifs_expired(self, payload: dict) -> None:
        ap = self.topology.aps[payload["node_id"]]
        if payload["event_id"] != ap.pending_event_id or ap.state != NodeState.AIFS:
            return

        if self.is_channel_busy(ap.id):
            self._set_node_state(ap, NodeState.BACKOFF_FROZEN)
            return

        if ap.backoff_slots == 0:
            self._attempt_transmission(ap)
            return

        self._set_node_state(ap, NodeState.BACKOFF)
        ap.backoff_countdown_start_us = self.current_time_us
        event_id = ap.pending_event_id
        self.schedule_event(
            self.current_time_us + ap.backoff_slots * self.config.mac.slot_time_us,
            3,
            "_on_backoff_expired",
            {"node_id": ap.id, "event_id": event_id},
        )

    def _on_backoff_expired(self, payload: dict) -> None:
        ap = self.topology.aps[payload["node_id"]]
        if payload["event_id"] != ap.pending_event_id or ap.state != NodeState.BACKOFF:
            return

        if self.is_channel_busy(ap.id):
            self._set_node_state(ap, NodeState.BACKOFF_FROZEN)
            return

        ap.backoff_slots = 0
        self._attempt_transmission(ap)

    def _reset_node_backoff(self, ap: AP, success: bool) -> None:
        if success:
            ap.cw = self.config.mac.cw_min
            ap.retry_count = 0
        else:
            ap.retry_count += 1
            if ap.retry_count > self.config.mac.retry_limit:
                ap.retry_count = 0
                ap.cw = self.config.mac.cw_min
            else:
                ap.cw = min(
                    2 * (ap.cw + 1) - 1,
                    self.config.mac.cw_max,
                )

        ap.backoff_slots = int(self.rng.integers(0, ap.cw + 1))
        self._set_node_state(ap, NodeState.IDLE)

    # ------------------------------------------------------------------
    # Scheduling / beamforming
    # ------------------------------------------------------------------
    def _select_random_stas(self, ap: AP, max_users: int) -> List[int]:
        if not ap.associated_stas or max_users <= 0:
            return []
        count = min(max_users, len(ap.associated_stas))
        selected = self.rng.choice(ap.associated_stas, size=count, replace=False)
        return [int(sta_id) for sta_id in np.atleast_1d(selected)]

    def _attempt_transmission(self, ap: AP) -> None:
        if self.mode == SimulationMode.MODE_A:
            self._start_mode_a(ap)
        else:
            self._start_mode_bc(ap)

    def _new_txop_id(self) -> int:
        self.txop_id_counter += 1
        return self.txop_id_counter

    def _new_tx_id(self) -> int:
        self.tx_id_counter += 1
        return self.tx_id_counter

    def _initialize_stream_mcs(self, tx: ActiveTransmission) -> None:
        active_list = list(self.active_transmissions.values())
        sinr_linear, sinr_db = calculate_sinr_dl(
            desired_tx_id=tx.tx_id,
            rx_sta_id=tx.rx_node_id,
            get_channel_fn=self._get_tx_channel,
            active_txs=active_list,
            mode=self.mode,
            serving_cluster_id=tx.cluster_id,
            p_noise_mw=self.channel.noise_mw,
        )
        mcs, rate_mbps, required_sinr_db = select_mcs(sinr_db)
        tx.selected_mcs = mcs
        tx.phy_rate_mbps = rate_mbps
        tx.required_sinr_db = required_sinr_db
        tx.initial_sinr_db = sinr_db
        tx.min_sinr_linear = sinr_linear

    # --------------------------- Mode A -------------------------------
    def _start_mode_a(self, ap: AP) -> None:
        """One randomly selected STA, one MRT stream, full AP power."""
        selected_stas = self._select_random_stas(ap, max_users=1)
        if not selected_stas:
            self._reset_node_backoff(ap, success=True)
            self._update_all_nodes_cca()
            return

        sta_id = selected_stas[0]
        sta = self.topology.nodes[sta_id]

        self._update_integrals()
        self._set_node_state(ap, NodeState.DL_TX)
        self._set_node_state(sta, NodeState.RECEIVING)

        txop_id = self._new_txop_id()
        tx_id = self._new_tx_id()

        channel = self.channel.sample_channel_matrix(sta_id, ap.id)
        shared_channels = {sta_id: channel}
        tx = ActiveTransmission(
            tx_id=tx_id,
            txop_id=txop_id,
            tx_node_id=ap.id,
            rx_node_id=sta_id,
            is_cobf=False,
            cluster_id=-1,
            start_time_us=self.current_time_us,
            channels=shared_channels,
            beamforming_vector=calculate_mrt_beamformer(channel),
            tx_power_mw=self.channel.ap_tx_power_mw,
        )
        self.active_transmissions[tx_id] = tx

        self._initialize_stream_mcs(tx)
        self._recalculate_ongoing_sinrs()
        self._consider_visualization_snapshot()

        # Reserve DATA + SIFS + ACK using an omnidirectional Duration-carrying
        # control advertisement. NAV decoding is SINR based.
        self._advertise_omnidirectional_nav(
            advertising_ap_ids={ap.id},
            duration_us=(
                self.config.mac.ppdu_duration_us
                + self.config.mac.sifs_us
                + self.config.mac.ack_duration_us
            ),
        )
        self._update_all_nodes_cca()

        self.schedule_event(
            self.current_time_us + self.config.mac.ppdu_duration_us,
            1,
            "_on_ppdu_end",
            {"tx_ids": [tx_id], "cluster_id": -1},
        )

    # -------------------------- Modes B/C -----------------------------
    def _find_nearest_free_ap(self, initiator: AP) -> Optional[AP]:
        """Nearest eligible AP within the coordination-distance limit."""
        best_ap: Optional[AP] = None
        best_distance = float("inf")

        for candidate in self.topology.aps:
            if candidate.id == initiator.id or not candidate.associated_stas:
                continue
            if candidate.state in {
                NodeState.DL_TX,
                NodeState.COBF_TX,
                NodeState.COORDINATION,
                NodeState.ACK,
            }:
                continue
            if self.is_nav_active(candidate.id) or self.is_channel_busy(candidate.id):
                continue
            if any(candidate.id in members for members in self.active_clusters.values()):
                continue

            distance = self.topology.get_distance(initiator.id, candidate.id)
            if distance > self.config.network.max_coordination_distance:
                continue

            if distance < best_distance:
                best_distance = distance
                best_ap = candidate
            elif math.isclose(distance, best_distance, rel_tol=1e-7, abs_tol=1e-7):
                if best_ap is None or candidate.id < best_ap.id:
                    best_ap = candidate

        return best_ap

    def _start_mode_bc(self, initiator: AP) -> None:
        """Form a two-AP Co-BF cluster or fall back to Mode-A MRT."""
        partner = self._find_nearest_free_ap(initiator)
        if partner is None:
            self._start_mode_a(initiator)
            return

        initiator_stas = self._select_random_stas(
            initiator,
            MAX_COBF_USERS_PER_AP,
        )
        partner_stas = self._select_random_stas(
            partner,
            MAX_COBF_USERS_PER_AP,
        )

        # Both eligible APs have at least one associated STA, but keep this guard
        # so future topology/scheduler changes fail safely.
        if not initiator_stas or not partner_stas:
            self._start_mode_a(initiator)
            return

        self._update_integrals()
        self._set_node_state(initiator, NodeState.COORDINATION)
        self._set_node_state(partner, NodeState.COORDINATION)

        self.cluster_id_counter += 1
        cluster_id = self.cluster_id_counter
        txop_id = self._new_txop_id()
        self.active_clusters[cluster_id] = {initiator.id, partner.id}

        # The coordination exchange belongs to the TXOP. Represent its control
        # traffic as a CCA-only reservation so neighboring APs freeze their
        # backoff instead of treating these 200 us as an idle gap.
        self._activate_coordination_reservations(
            cluster_id,
            (initiator, partner),
        )

        # Reserve coordination + DATA + SIFS + ACK from the beginning of the
        # coordinated TXOP. Both APs advertise omnidirectionally.
        self._advertise_omnidirectional_nav(
            advertising_ap_ids={initiator.id, partner.id},
            duration_us=(
                self.config.mac.coordination_overhead_us
                + self.config.mac.ppdu_duration_us
                + self.config.mac.sifs_us
                + self.config.mac.ack_duration_us
            ),
        )
        self._update_all_nodes_cca()

        self.schedule_event(
            self.current_time_us + self.config.mac.coordination_overhead_us,
            1,
            "_on_coordination_end",
            {
                "cluster_id": cluster_id,
                "txop_id": txop_id,
                "ap1_id": initiator.id,
                "ap2_id": partner.id,
                "ap1_sta_ids": initiator_stas,
                "ap2_sta_ids": partner_stas,
            },
        )

    def _on_coordination_end(self, payload: dict) -> None:
        cluster_id = payload["cluster_id"]
        txop_id = payload["txop_id"]
        ap1 = self.topology.aps[payload["ap1_id"]]
        ap2 = self.topology.aps[payload["ap2_id"]]

        ap_to_stas = {
            ap1.id: list(payload["ap1_sta_ids"]),
            ap2.id: list(payload["ap2_sta_ids"]),
        }
        assignments = [
            (ap_id, sta_id)
            for ap_id, sta_ids in ap_to_stas.items()
            for sta_id in sta_ids
        ]
        scheduled_stas = [sta_id for _, sta_id in assignments]

        # Build all intra-cluster channels before starting data. Every AP
        # transmits only to its OWN scheduled STAs, but uses CSI to all STAs
        # in this cluster for ZF/RZF victim protection.
        channels_by_ap: Dict[int, Dict[int, np.ndarray]] = {ap1.id: {}, ap2.id: {}}
        for ap in (ap1, ap2):
            for sta_id in scheduled_stas:
                channels_by_ap[ap.id][sta_id] = self.channel.sample_channel_matrix(sta_id, ap.id)

        # Geometry can make four requested LOS steering directions linearly
        # indistinguishable. Reduce loading rather than inventing a zero-power
        # desired beam, and never silently substitute MRT for a ZF stream.
        original_count = len(assignments)
        beams = None
        while True:
            try:
                beams = calculate_cluster_precoders(
                    channels_by_ap=channels_by_ap,
                    stream_assignments=assignments,
                    scheme=self.config.beamforming.scheme,
                    mmse_alpha=self.config.beamforming.mmse_alpha,
                    zf_rank_tolerance=self.config.beamforming.zf_rank_tolerance,
                    zf_min_desired_projection=self.config.beamforming.zf_min_desired_projection,
                )
                break
            except InfeasiblePrecoderError:
                removable = [a for a in (ap1.id, ap2.id)
                             if sum(ap_id == a for ap_id, _ in assignments) > 1]
                if not removable:
                    break
                ap_reduce = max(removable,
                                key=lambda a: sum(ap_id == a for ap_id, _ in assignments))
                removed = next((i for i in range(len(assignments) - 1, -1, -1)
                                if assignments[i][0] == ap_reduce), None)
                assignments.pop(removed)

        if beams is None:
            # Cannot obey requested physical ZF while delivering any data.
            # This is an aborted reservation, not a pretend successful TXOP.
            self.precoder_infeasible_txops += 1
            self._clear_coordination_reservations(cluster_id)
            self.active_clusters.pop(cluster_id, None)
            for ap in (ap1, ap2):
                self._reset_node_backoff(ap, success=False)
            self._update_all_nodes_cca()
            return
        if len(assignments) < original_count:
            self.reduced_stream_txops += 1

        # Switch from omnidirectional coordination reservation to beamformed
        # data without an intervening CCA update.
        self._clear_coordination_reservations(cluster_id)
        self._update_integrals()
        self._set_node_state(ap1, NodeState.COBF_TX)
        self._set_node_state(ap2, NodeState.COBF_TX)
        for _, sta_id in assignments:
            self._set_node_state(self.topology.nodes[sta_id], NodeState.RECEIVING)

        tx_ids: List[int] = []
        for ap_id, sta_id in assignments:
            tx_id = self._new_tx_id()
            tx_ids.append(tx_id)
            num_streams_from_ap = sum(a == ap_id for a, _ in assignments)
            tx = ActiveTransmission(
                tx_id=tx_id,
                txop_id=txop_id,
                tx_node_id=ap_id,
                rx_node_id=sta_id,
                is_cobf=True,
                cluster_id=cluster_id,
                start_time_us=self.current_time_us,
                channels=channels_by_ap[ap_id],
                beamforming_vector=beams[(ap_id, sta_id)],
                tx_power_mw=self.channel.ap_tx_power_mw / num_streams_from_ap,
            )
            self.active_transmissions[tx_id] = tx

        # Select each stream's MCS only after all streams are active, so the initial
        # SINR includes every same-cluster and external concurrent stream.
        for tx_id in tx_ids:
            self._initialize_stream_mcs(self.active_transmissions[tx_id])

        self._recalculate_ongoing_sinrs()
        self._consider_visualization_snapshot()

        # Refresh the remaining Duration at data start. A hidden AP that missed
        # the coordination exchange can still set NAV if it decodes this frame.
        self._advertise_omnidirectional_nav(
            advertising_ap_ids={ap1.id, ap2.id},
            duration_us=(
                self.config.mac.ppdu_duration_us
                + self.config.mac.sifs_us
                + self.config.mac.ack_duration_us
            ),
        )
        self._update_all_nodes_cca()

        self.schedule_event(
            self.current_time_us + self.config.mac.ppdu_duration_us,
            1,
            "_on_ppdu_end",
            {"tx_ids": tx_ids, "cluster_id": cluster_id},
        )

    # ------------------------------------------------------------------
    # PPDU completion / metrics records
    # ------------------------------------------------------------------
    def _on_ppdu_end(self, payload: dict) -> None:
        self._update_integrals()
        tx_ids: List[int] = payload["tx_ids"]
        cluster_id = payload.get("cluster_id", -1)

        records: List[TransmissionRecord] = []
        transmitting_aps: Dict[int, AP] = {}

        for tx_id in tx_ids:
            tx = self.active_transmissions.pop(tx_id)
            ap = self.topology.aps[tx.tx_node_id]
            sta = self.topology.nodes[tx.rx_node_id]
            transmitting_aps[ap.id] = ap

            min_sinr_db = 10.0 * math.log10(max(tx.min_sinr_linear, 1e-12))
            success = min_sinr_db >= tx.required_sinr_db

            if success:
                available_bits = int(
                    tx.phy_rate_mbps
                    * 1e6
                    * (self.config.mac.ppdu_duration_us * 1e-6)
                )
                packet_bits = self.config.traffic.payload_size_bytes * 8
                complete_packets = available_bits // packet_bits
                payload_bytes = complete_packets * self.config.traffic.payload_size_bytes
            else:
                payload_bytes = 0

            record = TransmissionRecord(
                tx_id=tx.tx_id,
                txop_id=tx.txop_id,
                tx_id_node=tx.tx_node_id,
                rx_id_node=tx.rx_node_id,
                is_cobf=tx.is_cobf,
                cluster_id=tx.cluster_id,
                start_time_us=tx.start_time_us,
                end_time_us=self.current_time_us,
                selected_mcs=tx.selected_mcs,
                phy_rate_mbps=tx.phy_rate_mbps,
                required_sinr_db=tx.required_sinr_db,
                initial_sinr_db=tx.initial_sinr_db,
                min_sinr_linear=tx.min_sinr_linear,
                tx_power_mw=tx.tx_power_mw,
                payload_bytes=payload_bytes,
                success=success,
            )
            self.completed_transmissions.append(record)
            records.append(record)
            self._set_node_state(sta, NodeState.ACK)

        for ap in transmitting_aps.values():
            self._set_node_state(ap, NodeState.ACK)

        if cluster_id != -1:
            self.active_clusters.pop(cluster_id, None)

        # Remaining transmissions immediately experience their new (usually higher)
        # SINR after these AP streams leave the channel.
        self._recalculate_ongoing_sinrs()
        self._update_all_nodes_cca()

        self.schedule_event(
            self.current_time_us
            + self.config.mac.sifs_us
            + self.config.mac.ack_duration_us,
            1,
            "_on_txop_end",
            {"records": records},
        )

    def _on_txop_end(self, payload: dict) -> None:
        self._update_integrals()
        records: List[TransmissionRecord] = payload["records"]

        for record in records:
            self._set_node_state(
                self.topology.nodes[record.rx_id_node],
                NodeState.IDLE,
            )

        records_by_ap: Dict[int, List[TransmissionRecord]] = {}
        for record in records:
            records_by_ap.setdefault(record.tx_id_node, []).append(record)

        for ap_id, ap_records in records_by_ap.items():
            # MU PPDU outcome for contention: partial user success is not treated
            # as a collision. Reset CW when at least one scheduled stream from
            # this AP succeeds; individual failed streams still deliver zero bits
            # and remain visible in stream_failure_rate.
            ap_success = any(record.success for record in ap_records)
            self._reset_node_backoff(self.topology.aps[ap_id], ap_success)

        self._update_all_nodes_cca()

    # ------------------------------------------------------------------
    # Main loop and metrics
    # ------------------------------------------------------------------
    def run(self) -> SimulationMetrics:
        """Run until the configured simulation time and return summary metrics."""
        for ap in self.topology.aps:
            if not ap.associated_stas:
                continue
            ap.cw = self.config.mac.cw_min
            ap.retry_count = 0
            ap.nav_end_time_us = 0.0
            ap.nav_event_id = 0
            ap.backoff_slots = int(self.rng.integers(0, ap.cw + 1))
            self._set_node_state(ap, NodeState.AIFS)
            event_id = ap.pending_event_id
            self.schedule_event(
                self.config.mac.aifs_us,
                2,
                "_on_aifs_expired",
                {"node_id": ap.id, "event_id": event_id},
            )

        while self.event_queue:
            time_us, _, _, handler_name, payload = heapq.heappop(self.event_queue)
            if time_us > self.sim_time_us:
                self.current_time_us = self.sim_time_us
                break

            self.current_time_us = time_us
            getattr(self, handler_name)(payload)

        self._update_integrals()
        return self._compute_metrics()

    def _compute_metrics(self) -> SimulationMetrics:
        sim_duration_s = self.sim_time_us * 1e-6
        records = self.completed_transmissions

        total_payload_bits = sum(record.payload_bytes * 8 for record in records)
        total_throughput_mbps = total_payload_bits / sim_duration_s / 1e6

        # A Co-BF AP with two users sends one MU PPDU carrying two stream records.
        ppdus: Dict[Tuple[int, int], List[TransmissionRecord]] = {}
        for record in records:
            ppdus.setdefault((record.txop_id, record.tx_id_node), []).append(record)

        total_ppdus = len(ppdus)
        # AP-level MU PPDU success uses the same rule as contention-window
        # feedback: the AP PPDU is a complete failure only when every scheduled
        # user stream fails. Partial user failures are captured separately below.
        successful_ppdus = sum(
            1 for streams in ppdus.values() if any(stream.success for stream in streams)
        )
        failed_ppdus = total_ppdus - successful_ppdus
        ppdu_success_rate = successful_ppdus / total_ppdus if total_ppdus else 0.0
        ppdu_failure_rate = failed_ppdus / total_ppdus if total_ppdus else 0.0

        total_streams = len(records)
        successful_streams = sum(record.success for record in records)
        failed_streams = total_streams - successful_streams
        stream_failure_rate = failed_streams / total_streams if total_streams else 0.0

        mean_sinr_db = (
            float(np.mean([record.initial_sinr_db for record in records]))
            if records
            else 0.0
        )

        per_ap_bits = {ap.id: 0 for ap in self.topology.aps}
        for record in records:
            per_ap_bits[record.tx_id_node] += record.payload_bytes * 8
        per_ap_total_mbps = {
            ap_id: bits / sim_duration_s / 1e6
            for ap_id, bits in per_ap_bits.items()
        }

        return SimulationMetrics(
            mode=self.mode,
            total_throughput_mbps=total_throughput_mbps,
            total_ppdus=total_ppdus,
            successful_ppdus=successful_ppdus,
            failed_ppdus=failed_ppdus,
            ppdu_success_rate=ppdu_success_rate,
            ppdu_failure_rate=ppdu_failure_rate,
            total_streams=total_streams,
            successful_streams=successful_streams,
            failed_streams=failed_streams,
            stream_failure_rate=stream_failure_rate,
            mean_sinr_db=mean_sinr_db,
            jain_fairness=compute_jain_fairness(list(per_ap_total_mbps.values())),
            avg_simultaneous_tx=(
                self.simultaneous_ap_time_integral / self.sim_time_us
            ),
            avg_active_clusters=(
                self.active_cluster_time_integral / self.sim_time_us
            ),
            per_ap_total_mbps=per_ap_total_mbps,
        )
