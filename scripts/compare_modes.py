from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wifi_sim.config import load_config
from wifi_sim.experiment import run_all_modes
from wifi_sim.metrics import (
    format_per_ap_table,
    format_summary_table,
    save_mode_comparison_csv,
    save_per_ap_csv,
)
from wifi_sim.plotting import save_timing_events_csv


def main(sim_time_s: float | None = None, seed: int | None = None) -> None:
    cfg = load_config()
    if sim_time_s is not None:
        cfg.simulation.sim_time_s = sim_time_s
    if seed is not None:
        cfg.network.seed = seed

    topology, metrics, simulators = run_all_modes(cfg, seed=cfg.network.seed)

    print(format_summary_table(metrics))
    print()
    print(format_per_ap_table(topology, metrics))

    summary_dir = PROJECT_ROOT / "results" / "summaries"
    save_mode_comparison_csv(metrics, summary_dir / "mode_comparison.csv")
    save_per_ap_csv(topology, metrics, summary_dir / "per_ap_throughput.csv")

    for mode, simulator in simulators.items():
        save_timing_events_csv(
            simulator.timing_events,
            summary_dir / f"timing_events_mode_{mode.lower()}.csv",
        )

    print(f"\nSaved CSV summaries under: {summary_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Modes A, B and C.")
    parser.add_argument("--sim-time", type=float, default=None, help="Simulation duration in seconds.")
    parser.add_argument("--seed", type=int, default=None, help="Override the configured random seed.")
    parser.add_argument("--quick", action="store_true", help="Run a short 0.5 s comparison.")
    args = parser.parse_args()

    main(sim_time_s=0.5 if args.quick else args.sim_time, seed=args.seed)
