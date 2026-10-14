from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wifi_sim.config import load_config
from wifi_sim.experiment import MODES, run_all_modes
from wifi_sim.plotting import plot_monte_carlo_comparison


def main(num_runs: int | None = None, sim_time_s: float | None = None) -> None:
    cfg = load_config()
    if num_runs is None:
        num_runs = cfg.monte_carlo.num_runs
    if sim_time_s is not None:
        cfg.simulation.sim_time_s = sim_time_s

    raw_rows = []

    for run_index in range(num_runs):
        seed = cfg.network.seed + 100 * run_index
        _, metrics, _ = run_all_modes(cfg, seed=seed)

        print(f"Run {run_index + 1}/{num_runs} (seed={seed})")
        for mode in MODES:
            result = metrics[mode]
            print(f"  Mode {mode}: {result.total_throughput_mbps:.2f} Mbps")
            raw_rows.append(
                {
                    "Run": run_index + 1,
                    "Seed": seed,
                    "Mode": mode,
                    "Total_Mbps": result.total_throughput_mbps,
                    "PPDU_Failure_Rate": result.ppdu_failure_rate,
                    "Stream_Failure_Rate": result.stream_failure_rate,
                    "Jain_Fairness": result.jain_fairness,
                    "Mean_SINR_dB": result.mean_sinr_db,
                }
            )

    raw = pd.DataFrame(raw_rows)
    summary_rows = []

    for mode in MODES:
        values = raw.loc[raw["Mode"] == mode, "Total_Mbps"]
        n = len(values)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        if n > 1:
            t_critical = float(stats.t.ppf(0.975, df=n - 1))
            ci95 = t_critical * std / math.sqrt(n)
        else:
            ci95 = 0.0

        mode_rows = raw[raw["Mode"] == mode]
        summary_rows.append(
            {
                "Mode": mode,
                "Runs": n,
                "Mean_Total_Mbps": mean,
                "Std_Total_Mbps": std,
                "CI95_Total_Mbps": ci95,
                "Mean_PPDU_Failure_Rate": float(mode_rows["PPDU_Failure_Rate"].mean()),
                "Mean_Stream_Failure_Rate": float(mode_rows["Stream_Failure_Rate"].mean()),
                "Mean_Jain_Fairness": float(mode_rows["Jain_Fairness"].mean()),
                "Mean_SINR_dB": float(mode_rows["Mean_SINR_dB"].mean()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    print("\nMonte Carlo throughput summary (mean +/- 95% CI):")
    for row in summary_rows:
        print(
            f"  Mode {row['Mode']}: "
            f"{row['Mean_Total_Mbps']:.2f} +/- {row['CI95_Total_Mbps']:.2f} Mbps"
        )

    summary_dir = PROJECT_ROOT / "results" / "summaries"
    figure_dir = PROJECT_ROOT / "results" / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    raw.to_csv(summary_dir / "monte_carlo_raw.csv", index=False)
    summary.to_csv(summary_dir / "monte_carlo_summary.csv", index=False)
    plot_monte_carlo_comparison(summary, figure_dir / "monte_carlo_throughput.png")

    print(f"\nSaved Monte Carlo outputs under: {PROJECT_ROOT / 'results'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo comparison of Modes A/B/C.")
    parser.add_argument("--runs", type=int, default=None, help="Number of random realizations.")
    parser.add_argument("--sim-time", type=float, default=None, help="Simulation duration per realization.")
    parser.add_argument("--dev", action="store_true", help="Use configured dev-run count and 0.5 s simulations.")
    args = parser.parse_args()

    cfg = load_config()
    runs = cfg.monte_carlo.dev_runs if args.dev else args.runs
    sim_time = 0.5 if args.dev else args.sim_time
    main(num_runs=runs, sim_time_s=sim_time)
