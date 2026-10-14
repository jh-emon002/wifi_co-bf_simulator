"""Verified LOS beam, STA-bearing, SINR and cross-cluster ICI visualizations.

No independent plotting channel: equations here match GeometryBasedChannel
exactly. Each virtual receiver and actual STA has ONE antenna (no combiner).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from wifi_sim.models import SnapshotStream, TxopSnapshot

MODE_LABELS = {
    'A': 'CSMA/CA + MRT',
    'B': 'Co-BF + inter-cluster interference',
    'C': 'Ideal Co-BF without inter-cluster interference',
}
COLORS = {'desired': '#177844', 'intra': '#bd6900', 'external': '#bb2533'}


def _save(fig, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _snapshot(simulator) -> TxopSnapshot:
    snap = simulator.visualization_snapshot
    if snap is None:
        raise RuntimeError('No valid snapshot: Mode B/C require 2 genuinely simultaneous Co-BF clusters. '
                           'Try another topology seed or longer simulation; never force concurrency.')
    return snap


def _focus_streams(snapshot: TxopSnapshot) -> list[SnapshotStream]:
    return [s for s in snapshot.streams if s.txop_id == snapshot.focus_txop_id]


def array_gain_from_weights(simulator, ap_id: int, weights: np.ndarray,
                            azimuth_rad: np.ndarray | float) -> np.ndarray:
    """Ideal-element physical ULA array gain Nt*|a(theta)^H w|^2 (linear).

    Unit-norm beam weights are asserted. Pattern includes array gain, unlike
    normalized-to-own-peak polar snapshots. Gain can peak below Nt for ZF.
    """
    ap = simulator.topology.nodes[ap_id]
    weights = np.asarray(weights, dtype=complex).reshape(-1)
    if not np.isclose(np.linalg.norm(weights), 1.0, atol=1e-6):
        raise ValueError('Non-unit beam weight: array gain and EIRP are invalid')
    angular = np.asarray(azimuth_rad, dtype=float)
    flat = angular.reshape(-1)
    nt = len(weights)
    d = simulator.config.channel.element_spacing_wavelengths
    n = np.arange(nt, dtype=float)
    phase = 2.0 * np.pi * d * np.cos(flat[:, None] - ap.array_orientation_rad) * n
    steering = np.exp(1j * phase) / math.sqrt(nt)
    gain = nt * np.abs(steering.conj() @ weights) ** 2
    return gain.reshape(angular.shape)


def _categories_for_ap(snap: TxopSnapshot, ap_id: int) -> list[tuple[SnapshotStream, str]]:
    own = [s for s in snap.streams if s.tx_node_id == ap_id]
    if not own:
        return []
    cluster = own[0].cluster_id
    return [(s, ('desired' if s.tx_node_id == ap_id else
                 'intra' if (cluster >= 0 and s.cluster_id == cluster) else 'external'))
            for s in snap.streams]


def plot_ap_ula_pattern(simulator, ap_id: int, output_path: Path,
                        db_floor: float = -60.0) -> None:
    """Per-AP test_ula-style Cartesian/polar pattern with actual STA bearings.

    1/2 streams from the AP are plotted. Each STA marker is placed on the
    evaluated array-gain curve at that AP-to-STA physical direction.
    Orange = intra-cluster victim (ZF expected null, MMSE not guaranteed).
    Red = external-cluster active STA (no cross-cluster null prescribed).
    """
    snap = _snapshot(simulator)
    ap_streams = [s for s in snap.streams if s.tx_node_id == ap_id]
    if not ap_streams:
        raise ValueError(f'AP{ap_id} did not transmit at this snapshot')
    ap = simulator.topology.nodes[ap_id]
    angles_deg = np.arange(0.0, 360.0 + simulator.config.visualization.beam_angular_resolution_deg,
                           simulator.config.visualization.beam_angular_resolution_deg)
    angles_rad = np.deg2rad(angles_deg)
    fig = plt.figure(figsize=(14.5, 6.0))
    plot_grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
    cart = fig.add_subplot(plot_grid[0, 0])
    polar = fig.add_subplot(plot_grid[0, 1], projection='polar')
    mpl_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    external_count = 0
    for stream_index, stream in enumerate(ap_streams):
        linecolor = mpl_colors[stream_index % len(mpl_colors)]
        gain = array_gain_from_weights(simulator, ap_id, stream.beamforming_vector, angles_rad)
        gain_db = 10.0 * np.log10(np.maximum(gain, 1e-12))
        clipped_db = np.maximum(gain_db, db_floor)
        cart.plot(angles_deg, clipped_db, color=linecolor, lw=1.6,
                  label=f'Beam: AP{ap_id} → STA{stream.rx_node_id}')
        polar.plot(angles_rad, gain, color=linecolor, lw=1.6,
                   label=f'STA{stream.rx_node_id} beam')
        # Actual receiver positions (both neighboring same-cluster and external).
        for other in snap.streams:
            sta = simulator.topology.nodes[other.rx_node_id]
            theta = simulator.channel.bearing(ap.x, ap.y, sta.x, sta.y)
            degrees = math.degrees(theta) % 360.0
            bearing_gain = float(array_gain_from_weights(
                simulator, ap_id, stream.beamforming_vector, theta))
            if other.rx_node_id == stream.rx_node_id:
                kind, marker, size = 'desired', 'o', 55
            elif stream.cluster_id >= 0 and other.cluster_id == stream.cluster_id:
                kind, marker, size = 'intra', 'x', 52
            else:
                kind, marker, size = 'external', 's', 36
                external_count += 1
            color = COLORS[kind]
            # Show true value (flooring only for visible Cartesian markers).
            cart.scatter(degrees, max(10 * math.log10(max(bearing_gain, 1e-12)), db_floor),
                         color=color, marker=marker, s=size, zorder=5)
            polar.scatter(theta % (2.0 * math.pi), bearing_gain,
                          color=color, marker=marker, s=size * 0.7, zorder=5)
            # Put text on first stream only to reduce overlapping annotations.
            if stream_index == 0:
                cart.axvline(degrees, color=color, alpha=0.21, lw=1, ls='--')
                cart.annotate(f'STA{sta.id}',
                              (degrees, max(10 * math.log10(max(bearing_gain, 1e-12)), db_floor)),
                              xytext=(2, 5), textcoords='offset points',
                              fontsize=7, color=color, rotation=30)
                polar.annotate(f'S{sta.id}', (theta % (2 * np.pi), bearing_gain),
                               xytext=(4, 3), textcoords='offset points',
                               fontsize=7, color=color)

    # Grey dots are neighboring STA positions not sending/receiving in this
    # instantaneous TXOP; they must NEVER be counted as interfering streams.
    if simulator.config.visualization.show_inactive_neighbor_stas:
        active_stas = {s.rx_node_id for s in snap.streams}
        inactive = [sta for sta in simulator.topology.stas
                    if sta.id not in active_stas and
                    math.hypot(sta.x-ap.x, sta.y-ap.y) <=
                    simulator.config.visualization.inactive_neighbor_radius_m]
        inactive.sort(key=lambda sta: math.hypot(sta.x-ap.x, sta.y-ap.y))
        first_beam = ap_streams[0].beamforming_vector
        for i, sta in enumerate(inactive):
            theta = simulator.channel.bearing(ap.x, ap.y, sta.x, sta.y)
            gain_val = float(array_gain_from_weights(simulator, ap_id, first_beam, theta))
            gain_db = max(10 * math.log10(max(gain_val, 1e-12)), db_floor)
            cart.scatter(math.degrees(theta)%360, gain_db,
                         color='#888888', alpha=0.55, s=16, zorder=3,
                         label='inactive neighbor STA (first beam)' if i == 0 else None)
            polar.scatter(theta % (2*np.pi), gain_val, color='#888888',
                          alpha=0.55, s=14, zorder=3)
            if i < 5:
                cart.annotate(f'S{sta.id}', (math.degrees(theta)%360, gain_db),
                              xytext=(2, -9), textcoords='offset points',
                              fontsize=6, color='#888888')

    cart.set_xlabel('Global azimuth (degrees, 0–360)')
    cart.set_ylabel('Ideal-element array gain (dBi)')
    cart.set_ylim(db_floor, 10.0 * np.log10(simulator.config.rf.ap_antennas) + 2)
    cart.set_xlim(0, 360)
    cart.set_xticks(np.arange(0, 361, 45))
    cart.grid(alpha=0.3)
    cart.legend(fontsize=8, loc='lower left')
    polar.set_theta_zero_location('E')
    polar.set_theta_direction(1)
    polar.set_title('Actual linear array gain (NOT peak-normalized)', fontsize=10, pad=15)
    polar.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.2))
    fig.suptitle(
        f'Mode {snap.mode} / {snap.precoder_scheme.upper()} — AP{ap_id} '
        f'(cluster {ap_streams[0].cluster_id}, {len(ap_streams)} streams)\n'
        f'Green circles: beam target  |  orange crosses: intra-cluster users  |  '
        f'red squares: concurrently active external users\n'
        f't={snap.time_us / 1000:.3f} ms; gray: inactive nearby STA positions',
        fontsize=11,
    )
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.06, right=0.96, wspace=0.23)
    _save(fig, output_path)


def plot_ap_aggregate_eirp(simulator, ap_id: int, output_path: Path) -> None:
    """Total mean directional EIRP (dBm) from this AP's independent data streams.

    Unlike the per-beam *gain* plot, this includes actual stream powers and
    uses an INCOHERENT SUM (independent payload symbols per stream).
    """
    snap = _snapshot(simulator)
    streams = [s for s in snap.streams if s.tx_node_id == ap_id]
    if not streams:
        raise ValueError('This AP is not active')
    degrees = np.linspace(0, 360, 1441)
    rad = np.deg2rad(degrees)
    directional_mw = np.zeros_like(degrees)
    for s in streams:
        directional_mw += s.tx_power_mw * array_gain_from_weights(
            simulator, ap_id, s.beamforming_vector, rad)
    fig, ax = plt.subplots(figsize=(10, 4.1))
    ax.plot(degrees, 10.0 * np.log10(np.maximum(directional_mw, 1e-20)))
    ax.set_xlabel('Global azimuth (degrees)')
    ax.set_ylabel('Aggregate AP directional EIRP (dBm, ideal elements)')
    ax.set_title(f'AP{ap_id} aggregate radiated pattern, '
                 f'P_conducted={10*math.log10(sum(s.tx_power_mw for s in streams)):.1f} dBm, '
                 f'peak EIRP={10*math.log10(max(directional_mw)):.1f} dBm')
    ax.grid(alpha=0.3)
    _save(fig, output_path)


def plot_snapshot_beams(simulator, output_path: Path, db_floor: float = -60.0) -> None:
    """A single readable two-cluster overview with AP-level Cartesian gain."""
    snap = _snapshot(simulator)
    ap_ids = sorted({s.tx_node_id for s in snap.streams})
    if not simulator.config.visualization.plot_all_active_aps:
        ap_ids = sorted({s.tx_node_id for s in _focus_streams(snap)})
    n = len(ap_ids)
    fig, axs = plt.subplots(n, 1, figsize=(11.5, 3.0 * n), squeeze=False)
    degrees = np.linspace(0.0, 360.0, 1441)
    angles = np.deg2rad(degrees)
    for idx, ap_id in enumerate(ap_ids):
        ax = axs[idx, 0]
        own = [s for s in snap.streams if s.tx_node_id == ap_id]
        for s in own:
            gain = array_gain_from_weights(simulator, ap_id, s.beamforming_vector, angles)
            ax.plot(degrees, np.maximum(10 * np.log10(np.maximum(gain, 1e-12)), db_floor),
                    label=f'AP{ap_id}→STA{s.rx_node_id}')
        ap = simulator.topology.nodes[ap_id]
        for s in snap.streams:
            sta = simulator.topology.nodes[s.rx_node_id]
            angle = math.degrees(simulator.channel.bearing(ap.x, ap.y, sta.x, sta.y)) % 360
            kind = 'desired' if s.tx_node_id == ap_id else 'intra' if (own[0].cluster_id >= 0 and s.cluster_id == own[0].cluster_id) else 'external'
            ax.axvline(angle, alpha=0.5, lw=0.9, ls=':', color=COLORS[kind])
            ax.text(angle, -5 - 5 * (s.rx_node_id % 3), f'S{s.rx_node_id}',
                    rotation=90, fontsize=7, va='top', color=COLORS[kind])
        ax.set(xlim=(0, 360), ylim=(db_floor, 8), ylabel='Array gain (dBi)')
        ax.set_title(f'AP{ap_id}, cluster {own[0].cluster_id}')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.25)
    axs[-1, 0].set_xlabel('Global azimuth (degrees)')
    fig.suptitle(f'Mode {snap.mode}: real 2-cluster snapshot, {snap.precoder_scheme.upper()}, '
                 f't={snap.time_us/1000:.3f}ms; red = external users', fontsize=12)
    fig.subplots_adjust(hspace=0.50, top=0.94, bottom=0.07)
    _save(fig, output_path)


def plot_snapshot_topology(simulator, output_path: Path) -> None:
    snap = _snapshot(simulator)
    topology = simulator.topology
    active_aps = sorted({s.tx_node_id for s in snap.streams})
    active_stas = {s.rx_node_id for s in snap.streams}
    clusters = sorted({s.cluster_id for s in snap.streams})
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cluster_color = {c: palette[i % len(palette)] for i, c in enumerate(clusters)}
    fig, ax = plt.subplots(figsize=(9, 8))
    for ap in topology.aps:
        if ap.id not in active_aps:
            ax.scatter(ap.x, ap.y, c='lightgray', marker='^', s=58)
    for sta in topology.stas:
        if sta.id not in active_stas:
            ax.scatter(sta.x, sta.y, color='lightgray', s=12, alpha=0.55)
    for s in snap.streams:
        ap = topology.nodes[s.tx_node_id]
        sta = topology.nodes[s.rx_node_id]
        c = cluster_color[s.cluster_id]
        ax.scatter(ap.x, ap.y, color=c, marker='^', s=130, edgecolor='black', zorder=5)
        ax.scatter(sta.x, sta.y, color=c, marker='o', s=70,
                   edgecolor='black', zorder=5)
        ax.plot([ap.x, sta.x], [ap.y, sta.y], color=c, alpha=0.7, lw=1.6)
        ax.annotate(f'STA{sta.id}', (sta.x, sta.y), xytext=(3, 4),
                    textcoords='offset points', fontsize=7)
    for ap_id in active_aps:
        ap = topology.nodes[ap_id]
        ax.annotate(f'AP{ap.id}', (ap.x, ap.y), xytext=(5, 5),
                    textcoords='offset points', fontsize=9, weight='bold')
    ax.set(xlim=(0, topology.area_width), ylim=(0, topology.area_height),
           xlabel='X (m)', ylabel='Y (m)')
    ax.set_aspect('equal', 'box')
    ax.grid(alpha=0.25)
    ax.set_title(f'Actual concurrent data snapshot at {snap.time_us/1000:.3f} ms\n'
                 f'Co-BF clusters: {clusters}; focus cluster={snap.focus_cluster_id}; '
                 f'scheme={snap.precoder_scheme.upper()}')
    _save(fig, output_path)


def _rx_power_grid(simulator, stream: SnapshotStream,
                   xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Vectorized EXACT LOS one-receive-antenna |h(x,y)w|^2*P.

    Uses channel.py's AoD response, path loss, shadowing field, and antenna
    normalization (LOS phase cancels from receive *power*). This is the
    analytical vectorization of channel_matrix_to_position(..., nr=1).
    """
    ch = simulator.channel
    ap = simulator.topology.nodes[stream.tx_node_id]
    delta_x, delta_y = xx - ap.x, yy - ap.y
    azimuth = np.arctan2(delta_y, delta_x)
    distance = np.maximum(np.hypot(delta_x, delta_y),
                          max(ch.channel_cfg.minimum_distance_m,
                              ch.channel_cfg.reference_distance_m))
    d = ch.channel_cfg
    shadow = 0.0
    if d.shadowing_std_db > 0:
        mx, my = 0.5 * (ap.x + xx), 0.5 * (ap.y + yy)
        phase = (mx[..., None] * ch._shadow_kx
                 + my[..., None] * ch._shadow_ky + ch._shadow_phase)
        shadow = d.shadowing_std_db * np.sum(np.cos(phase), axis=-1) / math.sqrt(len(ch._shadow_phase)/2)
    pl_db = (ch.pl_d0_db + 10 * d.path_loss_exponent
             * np.log10(distance / d.reference_distance_m) + shadow)
    path_gain = 10.0 ** (-pl_db / 10.0)
    gain = array_gain_from_weights(simulator, stream.tx_node_id,
                                   stream.beamforming_vector, azimuth)
    return stream.tx_power_mw * path_gain * gain


def compute_mode_b_ici_components_grid(simulator, desired_stream: SnapshotStream,
                                       grid_res: int = 100) -> dict:
    """With/no ICI fields for the same fixed real snapshot, even in Mode C."""
    snap = _snapshot(simulator)
    x = np.linspace(0, simulator.topology.area_width, grid_res)
    y = np.linspace(0, simulator.topology.area_height, grid_res)
    xx, yy = np.meshgrid(x, y)
    signal_mw = _rx_power_grid(simulator, desired_stream, xx, yy)
    intra_mw = np.zeros_like(xx)
    external_mw = np.zeros_like(xx)
    for s in snap.streams:
        if s.tx_id == desired_stream.tx_id:
            continue
        power = _rx_power_grid(simulator, s, xx, yy)
        if desired_stream.cluster_id >= 0 and s.cluster_id == desired_stream.cluster_id:
            intra_mw += power
        else:
            external_mw += power
    n = simulator.channel.noise_mw
    with_ici_db = 10 * np.log10(np.maximum(signal_mw / np.maximum(n+intra_mw+external_mw, 1e-18), 1e-12))
    without_ici_db = 10 * np.log10(np.maximum(signal_mw / np.maximum(n+intra_mw, 1e-18), 1e-12))
    return {
        'xx': xx, 'yy': yy,
        'signal_mw': signal_mw, 'intra_mw': intra_mw,
        'external_ici_mw': external_mw,
        'sinr_with_ici_db': with_ici_db,
        'sinr_no_external_ici_db': without_ici_db,
        'ici_penalty_db': without_ici_db - with_ici_db,
    }


def compute_snapshot_sinr_grid(simulator, desired_stream: SnapshotStream,
                               grid_res: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fields = compute_mode_b_ici_components_grid(simulator, desired_stream, grid_res)
    mode = _snapshot(simulator).mode
    key = 'sinr_no_external_ici_db' if mode == 'C' else 'sinr_with_ici_db'
    return fields['xx'], fields['yy'], fields[key]


def _annotate_nodes(ax, sim, desired_stream: SnapshotStream) -> None:
    snap = _snapshot(sim)
    for ap_id in sorted({s.tx_node_id for s in snap.streams}):
        ap = sim.topology.nodes[ap_id]
        ax.scatter(ap.x, ap.y, marker='^', c='white', edgecolor='black', s=72)
        ax.annotate(f'AP{ap_id}', (ap.x, ap.y), fontsize=7)
    for s in snap.streams:
        sta = sim.topology.nodes[s.rx_node_id]
        category = 'desired' if s.tx_id == desired_stream.tx_id else 'intra' if (desired_stream.cluster_id >= 0 and s.cluster_id == desired_stream.cluster_id) else 'external'
        ax.scatter(sta.x, sta.y, marker='o', s=40,
                   c=COLORS[category], edgecolor='black', linewidth=0.5)
        ax.annotate(f'S{s.rx_node_id}', (sta.x, sta.y), fontsize=7, color='black')


def plot_snapshot_sinr_heatmap(simulator, output_path: Path, grid_res: int = 100,
                               sinr_min_db: float = -10, sinr_max_db: float = 45) -> None:
    snap = _snapshot(simulator)
    focus = _focus_streams(snap)
    ncols = min(2, len(focus))
    nrows = int(math.ceil(len(focus) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4*ncols, 5.5*nrows),
                             squeeze=False)
    for i, stream in enumerate(focus):
        ax = axes[i//ncols][i%ncols]
        xx, yy, data = compute_snapshot_sinr_grid(simulator, stream, grid_res)
        artist = ax.pcolormesh(xx, yy, data, cmap='RdYlGn',
                               vmin=sinr_min_db, vmax=sinr_max_db,
                               shading='auto')
        _annotate_nodes(ax, simulator, stream)
        fig.colorbar(artist, ax=ax, shrink=0.83, label='SINR (dB)')
        ax.set_title(f'AP{stream.tx_node_id} → STA{stream.rx_node_id} '
                     f'(actual STA {stream.snapshot_sinr_db:.1f} dB)')
        ax.set(xlabel='X (m)', ylabel='Y (m)')
        ax.set_aspect('equal', 'box')
    for idx in range(len(focus), nrows*ncols):
        axes[idx//ncols][idx%ncols].axis('off')
    fig.suptitle(f'Mode {snap.mode}: single-antenna LOS SINR at one true snapshot '
                 f'({snap.precoder_scheme.upper()}, {snap.time_us/1000:.3f} ms)')
    fig.subplots_adjust(wspace=0.23, hspace=0.32, top=0.91)
    _save(fig, output_path)


def _actual_sta_mode_b_breakdown(simulator, desired_stream: SnapshotStream) -> dict:
    """Physical scalar received powers at the ACTUAL STA, using stored beams."""
    snap = _snapshot(simulator)
    if snap.mode != 'B':
        raise ValueError('A Mode B snapshot is required')
    sta_id = desired_stream.rx_node_id
    ch = simulator.channel
    def received(s):
        h = ch.sample_channel_matrix(sta_id, s.tx_node_id)
        return s.tx_power_mw * float(abs((h @ s.beamforming_vector).item()) ** 2)
    signal = received(desired_stream)
    intra = 0.0
    external = 0.0
    for s in snap.streams:
        if s.tx_id == desired_stream.tx_id:
            continue
        if s.cluster_id == desired_stream.cluster_id:
            intra += received(s)
        else:
            external += received(s)
    n = ch.noise_mw
    with_ici = 10 * math.log10(max(signal / (n+intra+external), 1e-12))
    no_external = 10 * math.log10(max(signal / (n+intra), 1e-12))
    return {
        'signal_mw': signal, 'intra_cluster_interference_mw': intra,
        'external_ici_mw': external, 'noise_mw': n,
        'sinr_with_ici_db': with_ici,
        'sinr_no_external_ici_db': no_external,
        'ici_penalty_db': no_external - with_ici,
    }


def plot_mode_b_ici_verification(simulator, output_path: Path,
                                 grid_res: int = 100) -> None:
    snap = _snapshot(simulator)
    if snap.mode != 'B':
        raise ValueError('Mode B required')
    focus = _focus_streams(snap)
    most_affected = max(focus, key=lambda s: _actual_sta_mode_b_breakdown(
        simulator, s)['ici_penalty_db'])
    fields = compute_mode_b_ici_components_grid(simulator, most_affected, grid_res)
    panels = [
        ('signal_mw', 'Desired power (dBm)', 'viridis', True),
        ('external_ici_mw', 'External-cluster ICI (dBm)', 'magma', True),
        ('sinr_with_ici_db', 'SINR with ICI (dB)', 'RdYlGn', False),
        ('sinr_no_external_ici_db', 'Counterfactual SINR without ICI (dB)', 'RdYlGn', False),
        ('ici_penalty_db', 'SINR loss from external ICI (dB)', 'inferno', False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.8), squeeze=False)
    for ax, (field, title, cmap, dbm) in zip(axes.flat, panels):
        z = fields[field]
        if dbm:
            z = 10 * np.log10(np.maximum(z, 1e-18))
        image = ax.pcolormesh(fields['xx'], fields['yy'], z,
                              cmap=cmap, shading='auto')
        fig.colorbar(image, ax=ax, shrink=0.83)
        _annotate_nodes(ax, simulator, most_affected)
        ax.set(xlabel='X (m)', ylabel='Y (m)', title=title)
        ax.set_aspect('equal', 'box')
    axes[1, 2].axis('off')
    measured = _actual_sta_mode_b_breakdown(simulator, most_affected)
    fig.suptitle(f'Mode B ({snap.precoder_scheme.upper()}), most affected STA{most_affected.rx_node_id} '
                 f'at {snap.time_us/1000:.3f} ms — '
                 f'actual-STA ICI penalty = {measured["ici_penalty_db"]:.2f} dB', fontsize=12)
    fig.subplots_adjust(hspace=0.27, wspace=0.27, top=0.91)
    _save(fig, output_path)


def export_mode_b_ici_breakdown(simulator, output_csv: Path) -> None:
    snap = _snapshot(simulator)
    if snap.mode != 'B':
        raise ValueError('Mode B snapshot required')
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    def dbm(x):
        return 10 * math.log10(max(x, 1e-18))
    fields = ['ap_id', 'sta_id', 'cluster_id', 'snapshot_time_ms',
              'signal_dbm', 'intra_cluster_interference_dbm', 'external_ici_dbm',
              'noise_dbm', 'sinr_with_ici_db', 'sinr_no_external_ici_db',
              'ici_penalty_db']
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in _focus_streams(snap):
            b = _actual_sta_mode_b_breakdown(simulator, s)
            w.writerow({
                'ap_id': s.tx_node_id, 'sta_id': s.rx_node_id,
                'cluster_id': s.cluster_id,
                'snapshot_time_ms': snap.time_us/1000,
                'signal_dbm': dbm(b['signal_mw']),
                'intra_cluster_interference_dbm': dbm(b['intra_cluster_interference_mw']),
                'external_ici_dbm': dbm(b['external_ici_mw']),
                'noise_dbm': dbm(b['noise_mw']),
                'sinr_with_ici_db': b['sinr_with_ici_db'],
                'sinr_no_external_ici_db': b['sinr_no_external_ici_db'],
                'ici_penalty_db': b['ici_penalty_db'],
            })


def export_mode_b_zf_leakage_matrix(simulator, output_csv: Path) -> None:
    """|h_victim,ap @ w_serving|^2. ZF should null focus intra-cluster off-diagonals."""
    snap = _snapshot(simulator)
    if snap.mode != 'B':
        raise ValueError('Mode B snapshot required')
    focus = _focus_streams(snap)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['beam'] + [f'STA{s.rx_node_id}' for s in focus])
        for tx in focus:
            gains = []
            for victim in focus:
                h = simulator.channel.sample_channel_matrix(victim.rx_node_id, tx.tx_node_id)
                gains.append(f'{abs((h @ tx.beamforming_vector).item()) ** 2:.12e}')
            w.writerow([f'AP{tx.tx_node_id}->STA{tx.rx_node_id}'] + gains)


def export_mode_b_all_stream_interference(simulator, output_csv: Path) -> None:
    """Full directed matrix from every concurrent beam to every active STA.

    Actual ICI toward one STA = sum P_stream * this matrix for beams outside
    that STA's cluster. Includes both clusters, not just focus.
    """
    snap = _snapshot(simulator)
    rows = []
    for dest in snap.streams:
        sta_id = dest.rx_node_id
        for tx in snap.streams:
            h = simulator.channel.sample_channel_matrix(sta_id, tx.tx_node_id)
            received = tx.tx_power_mw * float(abs((h @ tx.beamforming_vector).item()) ** 2)
            rows.append(dict(receiver_sta=sta_id,
                             receiver_cluster=dest.cluster_id,
                             tx_ap=tx.tx_node_id,
                             tx_sta=tx.rx_node_id,
                             tx_cluster=tx.cluster_id,
                             component=('desired' if tx.tx_id == dest.tx_id else
                                        'intra' if tx.cluster_id == dest.cluster_id else
                                        'ICI'), received_power_mw=received,
                             received_power_dbm=10*math.log10(max(received, 1e-18))))
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
