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


def main(
    sim_time_s: float | None = None,
    seed: int | None = None,
    grid_res: int = 50,
) -> None:
    cfg = load_config()
    if sim_time_s is not None:
        cfg.simulation.sim_time_s = float(sim_time_s)
    if seed is not None:
        cfg.network.seed = int(seed)

    _, _, simulators = run_all_modes(cfg, seed=cfg.network.seed)

    figure_dir = PROJECT_ROOT / "results" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for mode in ("A", "B", "C"):
        simulator = simulators[mode]
        if simulator.visualization_snapshot is None:
            print(f"Mode {mode}: no data snapshot was captured; skipping beam/SINR figures.")
            continue

        beam_path = figure_dir / f"beam_snapshot_mode_{mode.lower()}.png"
        heatmap_path = figure_dir / f"sinr_heatmap_mode_{mode.lower()}.png"

        plot_snapshot_beams(simulator, beam_path)
        plot_snapshot_sinr_heatmap(
            simulator,
            heatmap_path,
            grid_res=grid_res,
        )

        snapshot = simulator.visualization_snapshot
        focus_streams = [
            stream
            for stream in snapshot.streams
            if stream.txop_id == snapshot.focus_txop_id
        ]
        print(
            f"Mode {mode}: snapshot at {snapshot.time_us / 1000.0:.3f} ms, "
            f"focus streams={len(focus_streams)}, "
            f"active streams={len(snapshot.streams)}"
        )
        print(f"  saved {beam_path}")
        print(f"  saved {heatmap_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beam-weight and SINR spatial diagnostic figures."
    )
    parser.add_argument(
        "--sim-time",
        type=float,
        default=None,
        help="Simulation duration in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the configured random seed.",
    )
    parser.add_argument(
        "--grid-res",
        type=int,
        default=50,
        help="SINR heatmap grid resolution per axis.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a short 0.5 s simulation.",
    )
    args = parser.parse_args()

    main(
        sim_time_s=0.5 if args.quick else args.sim_time,
        seed=args.seed,
        grid_res=args.grid_res,
    )
