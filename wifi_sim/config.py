"""Configuration for the LOS-only, single-antenna-STA Co-BF simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class NetworkConfig:
    area_width: float = 100.0
    area_height: float = 100.0
    num_aps: int = 10
    num_stas: int = 30
    association_radius: float = 30.0
    max_coordination_distance: float = 40.0
    randomize_array_orientations: bool = False
    seed: int = 12345

    def __post_init__(self) -> None:
        if self.num_aps < 2 or self.num_stas < 2:
            raise ValueError('Co-BF requires at least 2 APs and 2 STAs; two clusters need >=4 APs')
        if self.area_width <= 0 or self.area_height <= 0:
            raise ValueError('Deployment dimensions must be positive')
        if self.association_radius <= 0 or self.max_coordination_distance <= 0:
            raise ValueError('Association and coordination radii must be positive')


@dataclass
class ChannelConfig:
    """Spatially consistent, narrowband, single-LOS-path geometry channel.

    Explicit scatterers and receive combining have intentionally been removed.
    Shadowing defaults to zero for explainable verification, but can be enabled
    for large-scale sensitivity experiments without introducing angular paths.
    """
    center_frequency_hz: float = 5.18e9
    bandwidth_hz: float = 80e6
    path_loss_exponent: float = 3.0
    reference_distance_m: float = 1.0
    shadowing_std_db: float = 0.0
    shadowing_correlation_distance_m: float = 12.0
    shadowing_components: int = 6
    noise_figure_db: float = 7.0
    element_spacing_wavelengths: float = 0.5
    minimum_distance_m: float = 1.0

    def __post_init__(self) -> None:
        for key in ('center_frequency_hz', 'bandwidth_hz', 'path_loss_exponent',
                    'reference_distance_m', 'shadowing_std_db',
                    'shadowing_correlation_distance_m', 'noise_figure_db',
                    'element_spacing_wavelengths', 'minimum_distance_m'):
            setattr(self, key, float(getattr(self, key)))
        self.shadowing_components = int(self.shadowing_components)
        if self.center_frequency_hz <= 0 or self.bandwidth_hz <= 0:
            raise ValueError('Carrier frequency and bandwidth must be positive')
        if self.path_loss_exponent <= 0 or self.reference_distance_m <= 0:
            raise ValueError('Path loss parameters must be positive')
        if self.element_spacing_wavelengths <= 0 or self.minimum_distance_m <= 0:
            raise ValueError('Element spacing / minimum distance must be positive')
        if self.shadowing_std_db < 0:
            raise ValueError('shadowing_std_db must be nonnegative')
        if self.shadowing_components < 1:
            raise ValueError('shadowing_components must be >=1')


@dataclass
class MacConfig:
    slot_time_us: float = 9.0
    sifs_us: float = 16.0
    aifsn: int = 3
    cw_min: int = 15
    cw_max: int = 1023
    retry_limit: int = 7
    cca_threshold_dbm: float = -78.0
    nav_enabled: bool = True
    ppdu_duration_us: float = 1000.0
    ack_duration_us: float = 32.0
    coordination_overhead_us: float = 200.0

    @property
    def aifs_us(self) -> float:
        return self.sifs_us + self.aifsn * self.slot_time_us


@dataclass
class RfConfig:
    ap_tx_power_dbm: float = 20.0
    ap_antennas: int = 4
    sta_antennas: int = 1

    def __post_init__(self) -> None:
        if self.ap_antennas != 4:
            raise ValueError('This explainable baseline fixes 4 antennas per AP')
        if self.sta_antennas != 1:
            raise ValueError('This baseline fixes one receive antenna per STA')


@dataclass
class BeamformingConfig:
    """ZF or regularized ZF (MMSE-style) on coordinated, AP-local beams.

    mmse_alpha is dimensionless. For each AP's full intra-cluster channel
    matrix H, the absolute Tikhonov term is alpha*trace(H H^H)/K. This is
    intentionally tunable, not advertised as a universally optimal MMSE value.
    """
    scheme: str = 'zf'
    mmse_alpha: float = 0.1
    zf_rank_tolerance: float = 1e-9
    zf_min_desired_projection: float = 1e-8

    def __post_init__(self) -> None:
        self.mmse_alpha = float(self.mmse_alpha)
        self.zf_rank_tolerance = float(self.zf_rank_tolerance)
        self.zf_min_desired_projection = float(self.zf_min_desired_projection)
        self.scheme = self.scheme.lower()
        if self.scheme not in {'zf', 'mmse'}:
            raise ValueError('beamforming.scheme must be zf or mmse')
        if self.mmse_alpha <= 0 or not self.mmse_alpha < float('inf'):
            raise ValueError('mmse_alpha must be positive and finite')
        if self.zf_rank_tolerance <= 0 or self.zf_min_desired_projection < 0:
            raise ValueError('ZF tolerances must be nonnegative')


@dataclass
class VisualizationConfig:
    require_two_cobf_clusters: bool = True
    grid_resolution: int = 100
    beam_angular_resolution_deg: float = 0.25
    plot_all_active_aps: bool = True
    show_inactive_neighbor_stas: bool = True
    inactive_neighbor_radius_m: float = 40.0


@dataclass
class TrafficConfig:
    payload_size_bytes: int = 1500


@dataclass
class SimConfig:
    sim_time_s: float = 3.0
    timing_window_ms: float = 20.0


@dataclass
class MonteCarloConfig:
    num_runs: int = 20
    dev_runs: int = 3


@dataclass
class Config:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    mac: MacConfig = field(default_factory=MacConfig)
    rf: RfConfig = field(default_factory=RfConfig)
    beamforming: BeamformingConfig = field(default_factory=BeamformingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    simulation: SimConfig = field(default_factory=SimConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)


def load_config(config_path: Optional[str | Path] = None) -> Config:
    """Load project-root config.yaml, or use validated dataclass defaults."""
    if config_path is None:
        default_path = Path(__file__).resolve().parent.parent / 'config.yaml'
        config_path = default_path if default_path.exists() else None
    if config_path is None:
        return Config()
    with open(config_path, 'r', encoding='utf-8') as handle:
        data: Dict[str, Any] = yaml.safe_load(handle) or {}
    return Config(
        network=NetworkConfig(**data.get('network', {})),
        channel=ChannelConfig(**data.get('channel', {})),
        mac=MacConfig(**data.get('mac', {})),
        rf=RfConfig(**data.get('rf', {})),
        beamforming=BeamformingConfig(**data.get('beamforming', {})),
        visualization=VisualizationConfig(**data.get('visualization', {})),
        traffic=TrafficConfig(**data.get('traffic', {})),
        simulation=SimConfig(**data.get('simulation', {})),
        monte_carlo=MonteCarloConfig(**data.get('monte_carlo', {})),
    )
