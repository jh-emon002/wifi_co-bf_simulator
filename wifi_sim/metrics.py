from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd


MODE_LABELS = {
    "A": "CSMA/CA + MRT",
    "B": "Co-BF + ICI",
    "C": "Ideal Co-BF no ICI",
}


@dataclass
class SimulationMetrics:
    mode: str
    total_throughput_mbps: float
    total_ppdus: int
    successful_ppdus: int
    failed_ppdus: int
    ppdu_success_rate: float
    ppdu_failure_rate: float
    total_streams: int
    successful_streams: int
    failed_streams: int
    stream_failure_rate: float
    mean_sinr_db: float
    jain_fairness: float
    avg_simultaneous_tx: float
    avg_active_clusters: float
    per_ap_total_mbps: Dict[int, float]


def compute_jain_fairness(values: Sequence[float]) -> float:
    """Jain's fairness index: ``(sum x)^2 / (n * sum x^2)``."""
    if not values:
        return 1.0
    total = sum(values)
    sum_squares = sum(value * value for value in values)
    if sum_squares == 0.0:
        return 1.0
    return float((total * total) / (len(values) * sum_squares))


def format_summary_table(metrics: Dict[str, SimulationMetrics]) -> str:
    lines = [
        "=" * 72,
        "THREE-MODE DOWNLINK COMPARISON",
        "=" * 72,
        f"{'Mode':<22} {'Mbps':>10} {'AP-PPDU fail':>12} {'Stream fail':>13} {'SINR dB':>10}",
        "-" * 72,
    ]

    for mode in ("A", "B", "C"):
        if mode not in metrics:
            continue
        result = metrics[mode]
        lines.append(
            f"{MODE_LABELS[mode]:<22} "
            f"{result.total_throughput_mbps:>10.2f} "
            f"{100.0 * result.ppdu_failure_rate:>11.2f}% "
            f"{100.0 * result.stream_failure_rate:>12.2f}% "
            f"{result.mean_sinr_db:>10.2f}"
        )

    lines.extend(["", "Jain fairness / average simultaneously transmitting APs:"])
    for mode in ("A", "B", "C"):
        if mode in metrics:
            result = metrics[mode]
            lines.append(
                f"  Mode {mode}: fairness={result.jain_fairness:.4f}, "
                f"simultaneous APs={result.avg_simultaneous_tx:.2f}, "
                f"active clusters={result.avg_active_clusters:.2f}"
            )

    return "\n".join(lines)


def format_per_ap_table(topology, metrics: Dict[str, SimulationMetrics]) -> str:
    lines = [
        "=" * 66,
        "PER-AP DOWNLINK THROUGHPUT (Mbps)",
        "=" * 66,
        f"{'AP':<6} {'STAs':<8} {'Mode A':>14} {'Mode B':>14} {'Mode C':>14}",
        "-" * 66,
    ]

    for ap in topology.aps:
        values = [
            metrics[mode].per_ap_total_mbps.get(ap.id, 0.0) if mode in metrics else 0.0
            for mode in ("A", "B", "C")
        ]
        lines.append(
            f"{ap.id:<6} {len(ap.associated_stas):<8} "
            f"{values[0]:>14.2f} {values[1]:>14.2f} {values[2]:>14.2f}"
        )

    return "\n".join(lines)


def save_mode_comparison_csv(
    metrics: Dict[str, SimulationMetrics],
    filepath: Path,
) -> None:
    rows = []
    for mode in ("A", "B", "C"):
        if mode not in metrics:
            continue
        result = metrics[mode]
        rows.append(
            {
                "Mode": mode,
                "Description": MODE_LABELS[mode],
                "Total_Mbps": result.total_throughput_mbps,
                "Total_PPDUs": result.total_ppdus,
                "Successful_PPDUs": result.successful_ppdus,
                "Failed_PPDUs": result.failed_ppdus,
                "PPDU_Success_Rate": result.ppdu_success_rate,
                "PPDU_Failure_Rate": result.ppdu_failure_rate,
                "Total_Streams": result.total_streams,
                "Successful_Streams": result.successful_streams,
                "Failed_Streams": result.failed_streams,
                "Stream_Failure_Rate": result.stream_failure_rate,
                "Mean_SINR_dB": result.mean_sinr_db,
                "Jain_Fairness": result.jain_fairness,
                "Avg_Simultaneous_APs": result.avg_simultaneous_tx,
                "Avg_Active_CoBF_Clusters": result.avg_active_clusters,
            }
        )

    filepath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(filepath, index=False)


def save_per_ap_csv(topology, metrics: Dict[str, SimulationMetrics], filepath: Path) -> None:
    rows = []
    for ap in topology.aps:
        row = {
            "AP_ID": ap.id,
            "Associated_STAs": len(ap.associated_stas),
        }
        for mode in ("A", "B", "C"):
            if mode in metrics:
                row[f"Mode_{mode}_Mbps"] = metrics[mode].per_ap_total_mbps.get(
                    ap.id, 0.0
                )
        rows.append(row)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(filepath, index=False)
