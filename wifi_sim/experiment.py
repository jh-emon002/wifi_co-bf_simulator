from __future__ import annotations

from typing import Dict, Tuple

from wifi_sim.channel import WirelessChannel
from wifi_sim.config import Config
from wifi_sim.metrics import SimulationMetrics
from wifi_sim.models import Topology
from wifi_sim.simulator import Simulator, SimulationMode


MODES = (
    SimulationMode.MODE_A,
    SimulationMode.MODE_B,
    SimulationMode.MODE_C,
)


def build_topology(cfg: Config, seed: int) -> Topology:
    return Topology(
        num_aps=cfg.network.num_aps,
        num_stas=cfg.network.num_stas,
        area_width=cfg.network.area_width,
        area_height=cfg.network.area_height,
        association_radius=cfg.network.association_radius,
        randomize_array_orientations=cfg.network.randomize_array_orientations,
        seed=seed,
    )


def run_all_modes(
    cfg: Config,
    seed: int | None = None,
) -> Tuple[Topology, Dict[str, SimulationMetrics], Dict[str, Simulator]]:
    """Run A/B/C with fresh identically seeded topology/channel objects."""
    base_seed = cfg.network.seed if seed is None else int(seed)
    reference_topology = build_topology(cfg, base_seed)

    metrics: Dict[str, SimulationMetrics] = {}
    simulators: Dict[str, Simulator] = {}

    for mode in MODES:
        topology = build_topology(cfg, base_seed)
        channel = WirelessChannel(topology, cfg.channel, cfg.rf, seed=base_seed)
        simulator = Simulator(topology, channel, cfg, mode=mode, seed=base_seed)
        metrics[mode] = simulator.run()
        simulators[mode] = simulator

    return reference_topology, metrics, simulators
