from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wifi_sim.metrics import MODE_LABELS, SimulationMetrics
from wifi_sim.models import TimingEvent, Topology


MODE_COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
STATE_COLORS = {
    "AIFS": "#c7c7c7",
    "BACKOFF": "#aec7e8",
    "BACKOFF FROZEN": "#ffbb78",
    "DL TX": "#1f77b4",
    "CO-BF TX": "#9467bd",
    "COORDINATION": "#ff7f0e",
    "ACK": "#e377c2",
    "RECEIVING": "#98df8a",
    "IDLE": "#ffffff",
}


def _save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_topology(topology: Topology, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    for ap in topology.aps:
        for sta_id in ap.associated_stas:
            sta = topology.nodes[sta_id]
            ax.plot(
                [ap.x, sta.x],
                [ap.y, sta.y],
                color="gray",
                linestyle="--",
                alpha=0.45,
                linewidth=0.8,
                zorder=1,
            )

    ax.scatter(
        [sta.x for sta in topology.stas],
        [sta.y for sta in topology.stas],
        marker="o",
        s=55,
        edgecolors="black",
        label="STAs",
        zorder=2,
    )
    for sta in topology.stas:
        ax.annotate(f"S{sta.id}", (sta.x, sta.y), xytext=(3, 3), textcoords="offset points", fontsize=7)

    ax.scatter(
        [ap.x for ap in topology.aps],
        [ap.y for ap in topology.aps],
        marker="^",
        s=120,
        edgecolors="black",
        label="APs",
        zorder=3,
    )
    for ap in topology.aps:
        ax.annotate(f"AP{ap.id}", (ap.x, ap.y), xytext=(5, 5), textcoords="offset points", fontsize=9, fontweight="bold")

    ax.set_xlim(0, topology.area_width)
    ax.set_ylim(0, topology.area_height)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Network Topology: {topology.num_aps} APs, {topology.num_stas} STAs")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    ax.set_aspect("equal", "box")
    _save_figure(fig, output_path)


def plot_throughput_comparison(metrics: Dict[str, SimulationMetrics], output_path: Path) -> None:
    modes = [mode for mode in ("A", "B", "C") if mode in metrics]
    values = [metrics[mode].total_throughput_mbps for mode in modes]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        np.arange(len(modes)),
        values,
        color=[MODE_COLORS[mode] for mode in modes],
        edgecolor="black",
        width=0.58,
    )
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.1f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels([MODE_LABELS[mode] for mode in modes])
    ax.set_ylabel("Downlink throughput (Mbps)")
    ax.set_title("Network Downlink Throughput")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    if values:
        ax.set_ylim(0, max(values) * 1.18 if max(values) > 0 else 1.0)
    _save_figure(fig, output_path)


def plot_per_ap_throughput(
    topology: Topology,
    metrics: Dict[str, SimulationMetrics],
    output_path: Path,
) -> None:
    ap_ids = [ap.id for ap in topology.aps]
    x = np.arange(len(ap_ids))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(ap_ids)), 5))
    for mode, offset in zip(("A", "B", "C"), (-width, 0.0, width)):
        if mode not in metrics:
            continue
        values = [metrics[mode].per_ap_total_mbps.get(ap_id, 0.0) for ap_id in ap_ids]
        ax.bar(x + offset, values, width=width, label=MODE_LABELS[mode], color=MODE_COLORS[mode], edgecolor="black")

    ax.set_xlabel("AP ID")
    ax.set_ylabel("Downlink throughput (Mbps)")
    ax.set_title("Per-AP Throughput")
    ax.set_xticks(x)
    ax.set_xticklabels([str(ap_id) for ap_id in ap_ids])
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend()
    _save_figure(fig, output_path)


def plot_metric_bar(
    metrics: Dict[str, SimulationMetrics],
    attribute: str,
    y_label: str,
    title: str,
    output_path: Path,
    as_percentage: bool = False,
) -> None:
    modes = [mode for mode in ("A", "B", "C") if mode in metrics]
    raw_values = [getattr(metrics[mode], attribute) for mode in modes]
    values = [100.0 * value if as_percentage else value for value in raw_values]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        np.arange(len(modes)),
        values,
        color=[MODE_COLORS[mode] for mode in modes],
        edgecolor="black",
        width=0.58,
    )
    for bar, value in zip(bars, values):
        text = f"{value:.2f}%" if as_percentage else f"{value:.2f}"
        ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 4), textcoords="offset points", ha="center")

    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels([MODE_LABELS[mode] for mode in modes])
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    if values:
        ax.set_ylim(0, max(values) * 1.22 if max(values) > 0 else 1.0)
    _save_figure(fig, output_path)


def plot_timing_diagram(
    timing_events: List[TimingEvent],
    mode: str,
    output_path: Path,
    window_ms: float,
    num_aps: int,
    max_nodes_to_show: int = 16,
) -> None:
    events = [event for event in timing_events if event.start_ms < window_ms]
    if not events:
        return

    node_ids = sorted({event.node_id for event in events})
    if len(node_ids) > max_nodes_to_show:
        ap_ids = [node_id for node_id in node_ids if node_id < num_aps]
        sta_ids = [node_id for node_id in node_ids if node_id >= num_aps]
        node_ids = (ap_ids + sta_ids)[:max_nodes_to_show]
        events = [event for event in events if event.node_id in node_ids]

    y_map = {node_id: index for index, node_id in enumerate(node_ids)}
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.45 * len(node_ids))))

    for event in events:
        ax.barh(
            y_map[event.node_id],
            max(event.end_ms - event.start_ms, 0.01),
            left=event.start_ms,
            height=0.65,
            color=STATE_COLORS.get(event.state, "#999999"),
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlim(0, window_ms)
    ax.set_yticks(range(len(node_ids)))
    ax.set_yticklabels([
        f"AP {node_id}" if node_id < num_aps else f"STA {node_id}"
        for node_id in node_ids
    ])
    ax.set_xlabel("Simulation time (ms)")
    ax.set_title(f"Timing Diagram - Mode {mode}: {MODE_LABELS[mode]}")
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    used_states = {event.state for event in events}
    legend_handles = [
        mpatches.Patch(facecolor=STATE_COLORS[state], edgecolor="black", label=state)
        for state in STATE_COLORS
        if state in used_states and state != "IDLE"
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=min(5, len(legend_handles)), fontsize=8)

    _save_figure(fig, output_path)


def save_timing_events_csv(timing_events: List[TimingEvent], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "Node_ID": event.node_id,
            "State": event.state,
            "Start_Time_ms": round(event.start_ms, 4),
            "End_Time_ms": round(event.end_ms, 4),
            "Duration_ms": round(event.end_ms - event.start_ms, 4),
        }
        for event in timing_events
    ]).to_csv(output_path, index=False)


def plot_monte_carlo_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    modes = [mode for mode in ("A", "B", "C") if mode in set(summary["Mode"])]
    indexed = summary.set_index("Mode")
    means = [float(indexed.loc[mode, "Mean_Total_Mbps"]) for mode in modes]
    ci95 = [float(indexed.loc[mode, "CI95_Total_Mbps"]) for mode in modes]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        np.arange(len(modes)),
        means,
        yerr=ci95,
        capsize=6,
        color=[MODE_COLORS[mode] for mode in modes],
        edgecolor="black",
        width=0.58,
    )
    for bar, mean, ci in zip(bars, means, ci95):
        ax.annotate(f"{mean:.1f} +/- {ci:.1f}", (bar.get_x() + bar.get_width() / 2, mean + ci), xytext=(0, 5), textcoords="offset points", ha="center")

    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels([MODE_LABELS[mode] for mode in modes])
    ax.set_ylabel("Downlink throughput (Mbps)")
    ax.set_title("Monte Carlo Throughput: Mean and 95% CI")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    if means:
        ax.set_ylim(0, max(mean + ci for mean, ci in zip(means, ci95)) * 1.22)
    _save_figure(fig, output_path)
