"""Narrowband geometry-based LOS channel with four-element AP ULAs.

Each STA has one omnidirectional receive antenna. There are no scatterers and
no receive combining. The same complex h is used by the event engine, beam
pattern verification, CCA/NAV, and spatial interference calculations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

import numpy as np

from wifi_sim.config import ChannelConfig, RfConfig
from wifi_sim.models import Topology


@dataclass(frozen=True)
class PathComponent:
    kind: str
    length_m: float
    aod_rad: float
    aoa_rad: float
    gain_linear: float
    phase_rad: float


class GeometryBasedChannel:
    SPEED_OF_LIGHT_MPS = 299_792_458.0

    def __init__(self, topology: Topology, channel_cfg: ChannelConfig,
                 rf_cfg: RfConfig, seed: int = 12345) -> None:
        self.topology = topology
        self.channel_cfg = channel_cfg
        self.rf_cfg = rf_cfg
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.wavelength_m = self.SPEED_OF_LIGHT_MPS / channel_cfg.center_frequency_hz
        self.pl_d0_db = 20.0 * math.log10(
            4.0 * math.pi * channel_cfg.reference_distance_m / self.wavelength_m
        )
        self.noise_dbm = (-174.0 + 10.0 * math.log10(channel_cfg.bandwidth_hz)
                          + channel_cfg.noise_figure_db)
        self.noise_mw = 10.0 ** (self.noise_dbm / 10.0)
        self.ap_tx_power_mw = 10.0 ** (rf_cfg.ap_tx_power_dbm / 10.0)

        # Optional large-scale shadowing field; zero by default.
        count = channel_cfg.shadowing_components
        corr = max(channel_cfg.shadowing_correlation_distance_m, 1e-9)
        phases = rng.uniform(0.0, 2.0 * math.pi, count)
        orientations = rng.uniform(0.0, 2.0 * math.pi, count)
        magnitudes = 2.0 * math.pi / corr * rng.uniform(0.55, 1.45, count)
        self._shadow_phase = phases
        self._shadow_kx = magnitudes * np.cos(orientations)
        self._shadow_ky = magnitudes * np.sin(orientations)
        self._node_channel_cache: Dict[Tuple[int, int], np.ndarray] = {}

        # Compatibility data for the existing summary/diagnostic routines.
        self.scatterers: List = []  # intentionally ALWAYS empty
        n_nodes = len(topology.nodes)
        self.path_loss_db = np.zeros((n_nodes, n_nodes), dtype=float)
        self.path_gain_linear = np.ones((n_nodes, n_nodes), dtype=float)
        self._initialize_direct_path_tables()

    def steering_vector(self, azimuth_rad: float, num_elements: int,
                        orientation_rad: float = 0.0) -> np.ndarray:
        """Unit-norm ULA response, array axis at global azimuth orientation_rad.

        a(theta)[n] = exp(j*2*pi*(d/lambda)*n*cos(theta-psi)) / sqrt(N).
        For N=1, a(theta)=1. The channel uses a_tx^H; transmitted gain is
        N*|a_tx(theta)^H w|^2 for unit-norm beam w.
        """
        n = np.arange(int(num_elements), dtype=float)
        phase = (2.0 * math.pi * self.channel_cfg.element_spacing_wavelengths
                 * n * math.cos(float(azimuth_rad) - float(orientation_rad)))
        return np.exp(1j * phase) / math.sqrt(len(n))

    @staticmethod
    def bearing(x0: float, y0: float, x1: float, y1: float) -> float:
        return math.atan2(y1 - y0, x1 - x0)

    def shadowing_db(self, tx_x: float, tx_y: float,
                     rx_x: float, rx_y: float) -> float:
        sigma = self.channel_cfg.shadowing_std_db
        if sigma == 0:
            return 0.0
        mx = (tx_x + rx_x) * 0.5
        my = (tx_y + rx_y) * 0.5
        phase = self._shadow_kx * mx + self._shadow_ky * my + self._shadow_phase
        return float(sigma * np.sum(np.cos(phase)) / math.sqrt(len(phase) / 2.0))

    def path_gain_to_position(self, ap_id: int, rx_x: float, rx_y: float) -> float:
        ap = self.topology.nodes[ap_id]
        r = max(math.hypot(rx_x - ap.x, rx_y - ap.y),
                self.channel_cfg.minimum_distance_m,
                self.channel_cfg.reference_distance_m)
        loss = (self.pl_d0_db + 10.0 * self.channel_cfg.path_loss_exponent
                * math.log10(r / self.channel_cfg.reference_distance_m)
                + self.shadowing_db(ap.x, ap.y, rx_x, rx_y))
        return 10.0 ** (-loss / 10.0)

    def path_components_to_position(self, ap_id: int, rx_x: float,
                                    rx_y: float) -> List[PathComponent]:
        ap = self.topology.nodes[ap_id]
        if not ap.is_ap:
            raise ValueError('Transmitter must be an AP')
        r = math.hypot(rx_x - ap.x, rx_y - ap.y)
        theta = self.bearing(ap.x, ap.y, rx_x, rx_y)
        return [PathComponent(
            kind='LOS', length_m=r, aod_rad=theta,
            aoa_rad=self.bearing(rx_x, rx_y, ap.x, ap.y),
            gain_linear=self.path_gain_to_position(ap_id, rx_x, rx_y),
            phase_rad=-2.0 * math.pi * r / self.wavelength_m,
        )]

    def channel_matrix_to_position(self, ap_id: int, rx_x: float, rx_y: float,
                                   rx_antennas: int | None = None,
                                   rx_orientation_rad: float = 0.0) -> np.ndarray:
        ap = self.topology.nodes[ap_id]
        if not ap.is_ap:
            raise ValueError('Transmitter must be an AP')
        nt = self.rf_cfg.ap_antennas
        nr = self.rf_cfg.sta_antennas if rx_antennas is None else int(rx_antennas)
        if nr < 1:
            raise ValueError('rx_antennas must be positive')
        path = self.path_components_to_position(ap_id, rx_x, rx_y)[0]
        at = self.steering_vector(path.aod_rad, nt, ap.array_orientation_rad)
        ar = self.steering_vector(path.aoa_rad, nr, rx_orientation_rad)
        coef = (math.sqrt(path.gain_linear * nt * nr)
                * np.exp(1j * path.phase_rad))
        return coef * np.outer(ar, at.conj())

    def channel_matrix(self, rx_id: int, ap_id: int) -> np.ndarray:
        key = (int(rx_id), int(ap_id))
        if key not in self._node_channel_cache:
            node = self.topology.nodes[rx_id]
            self._node_channel_cache[key] = self.channel_matrix_to_position(
                ap_id=ap_id, rx_x=node.x, rx_y=node.y,
                rx_antennas=self.rf_cfg.ap_antennas if node.is_ap else 1,
                rx_orientation_rad=node.array_orientation_rad,
            )
        return self._node_channel_cache[key]

    def sample_channel_matrix(self, rx_id: int, ap_id: int) -> np.ndarray:
        """Quasi-static channel, not a new Rayleigh sample."""
        return self.channel_matrix(rx_id, ap_id)

    def _initialize_direct_path_tables(self) -> None:
        for rx_id, rx in self.topology.nodes.items():
            for tx_id, tx in self.topology.nodes.items():
                if tx_id == rx_id:
                    continue
                r = max(math.hypot(rx.x - tx.x, rx.y - tx.y),
                        self.channel_cfg.minimum_distance_m,
                        self.channel_cfg.reference_distance_m)
                pl = (self.pl_d0_db + self.channel_cfg.path_loss_exponent
                      * 10.0 * math.log10(r / self.channel_cfg.reference_distance_m)
                      + self.shadowing_db(tx.x, tx.y, rx.x, rx.y))
                self.path_loss_db[rx_id, tx_id] = pl
                self.path_gain_linear[rx_id, tx_id] = 10.0 ** (-pl / 10.0)

    def get_tx_power_mw(self, tx_id: int) -> float:
        if not self.topology.nodes[tx_id].is_ap:
            raise ValueError('Only APs transmit')
        return self.ap_tx_power_mw


WirelessChannel = GeometryBasedChannel
