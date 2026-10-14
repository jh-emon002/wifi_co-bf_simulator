from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wifi_sim.beam_visualization import (
    plot_snapshot_beams,
    plot_snapshot_sinr_heatmap,
)
from wifi_sim.config import load_config
from wifi_sim.experiment import run_all_modes
from wifi_sim.plotting import (
    plot_metric_bar,
    plot_per_ap_throughput,
    plot_throughput_comparison,
    plot_timing_diagram,
    plot_topology,
)


def main(sim_time_s: float | None = None, seed: int | None = None) -> None:
    cfg = load_config()
    if sim_time_s is not None:
        cfg.simulation.sim_time_s = sim_time_s
    if seed is not None:
        cfg.network.seed = seed

    topology, metrics, simulators = run_all_modes(cfg, seed=cfg.network.seed)
    figure_dir = PROJECT_ROOT / "results" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_topology(topology, figure_dir / "topology.png")
    plot_throughput_comparison(metrics, figure_dir / "throughput.png")
    plot_per_ap_throughput(topology, metrics, figure_dir / "per_ap_throughput.png")
    plot_metric_bar(metrics, "ppdu_failure_rate", "AP PPDU failure rate (%)", "AP PPDU Failure Rate (All Streams Failed)", figure_dir / "ppdu_failure_rate.png", as_percentage=True)
    plot_metric_bar(metrics, "stream_failure_rate", "Stream failure rate (%)", "Spatial-Stream Failure Rate", figure_dir / "stream_failure_rate.png", as_percentage=True)
    plot_metric_bar(metrics, "mean_sinr_db", "Mean initial SINR (dB)", "Mean Receiver SINR", figure_dir / "mean_sinr.png")
    plot_metric_bar(metrics, "jain_fairness", "Jain fairness index", "Per-AP Throughput Fairness", figure_dir / "jain_fairness.png")
    plot_metric_bar(metrics, "avg_simultaneous_tx", "Average simultaneous transmitting APs", "Spatial Reuse", figure_dir / "simultaneous_aps.png")
    plot_metric_bar(metrics, "avg_active_clusters", "Average active Co-BF clusters", "Active Co-BF Clusters", figure_dir / "active_clusters.png")

    for mode, simulator in simulators.items():
        plot_timing_diagram(
            simulator.timing_events,
            mode,
            figure_dir / f"timing_mode_{mode.lower()}.png",
            window_ms=cfg.simulation.timing_window_ms,
            num_aps=cfg.network.num_aps,
        )

        if simulator.visualization_snapshot is not None:
            plot_snapshot_beams(
                simulator,
                figure_dir / f"beam_snapshot_mode_{mode.lower()}.png",
            )
            plot_snapshot_sinr_heatmap(
                simulator,
                figure_dir / f"sinr_heatmap_mode_{mode.lower()}.png",
            )

    print(f"Generated figures under: {figure_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all comparison figures.")
    parser.add_argument("--sim-time", type=float, default=None, help="Simulation duration in seconds.")
    parser.add_argument("--seed", type=int, default=None, help="Override the configured random seed.")
    parser.add_argument("--quick", action="store_true", help="Use a short 0.5 s simulation.")
    args = parser.parse_args()

    main(sim_time_s=0.5 if args.quick else args.sim_time, seed=args.seed)
