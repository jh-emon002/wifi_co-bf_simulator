"""Generate an evidence-based two-cluster Mode B snapshot and its plots.

Run from project root: python scripts/generate_mode_b_ici_verification.py
The simulator must ACTUALLY generate overlapping Co-BF transmissions; the
script never turns off CCA/NAV or forces a fake concurrent cluster.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wifi_sim.beam_visualization import (
    export_mode_b_all_stream_interference,
    export_mode_b_ici_breakdown,
    export_mode_b_zf_leakage_matrix,
    plot_ap_aggregate_eirp,
    plot_ap_ula_pattern,
    plot_mode_b_ici_verification,
    plot_snapshot_beams,
    plot_snapshot_sinr_heatmap,
    plot_snapshot_topology,
)
from wifi_sim.channel import WirelessChannel
from wifi_sim.config import load_config
from wifi_sim.experiment import build_topology
from wifi_sim.simulator import SimulationMode, Simulator


def run_mode_b(cfg, seed: int):
    topo = build_topology(cfg, seed)
    ch = WirelessChannel(topo, cfg.channel, cfg.rf, seed=seed)
    sim = Simulator(topo, ch, cfg, mode=SimulationMode.MODE_B, seed=seed)
    result = sim.run()
    return sim, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    parser.add_argument('--sim-time', type=float, default=1.5)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--search-seeds', type=int, default=25)
    parser.add_argument('--target-penalty-db', type=float, default=3.0,
                        help='Stop early if peak ICI SINR loss at any focus STA reaches target')
    parser.add_argument('--grid-res', type=int, default=None)
    parser.add_argument('--scheme', choices=['zf', 'mmse'], default=None)
    parser.add_argument('--mmse-alpha', type=float, default=None)
    parser.add_argument('--out', type=Path, default=ROOT / 'results' / 'figures' / 'mode_b')
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.simulation.sim_time_s = args.sim_time
    if args.scheme is not None:
        cfg.beamforming.scheme = args.scheme
    if args.mmse_alpha is not None:
        if args.mmse_alpha <= 0:
            parser.error('--mmse-alpha must be > 0')
        cfg.beamforming.mmse_alpha = args.mmse_alpha
    grid = args.grid_res or cfg.visualization.grid_resolution
    seed0 = cfg.network.seed if args.seed is None else args.seed
    best = None
    best_seed = None
    best_result = None
    print(f'Searching real 2-cluster overlapping Mode-B snapshots using '
          f'{cfg.beamforming.scheme.upper()} (max {args.search_seeds} seeds).', flush=True)
    for seed in range(seed0, seed0 + max(args.search_seeds, 1)):
        sim, result = run_mode_b(cfg, seed)
        snap = sim.visualization_snapshot
        if snap is None:
            print(f'  seed {seed}: no qualifying two-cluster overlap', flush=True)
            continue
        if best is None or snap.max_ici_penalty_db > best.visualization_snapshot.max_ici_penalty_db:
            best, best_seed, best_result = sim, seed, result
        print(f'  seed {seed}: overlap at {snap.time_us/1000:.3f}ms, '
              f'peak ICI loss {snap.max_ici_penalty_db:.2f} dB', flush=True)
        if snap.max_ici_penalty_db >= args.target_penalty_db:
            break

    if best is None:
        raise SystemExit('No genuine two-Co-BF-cluster snapshot in searched random seeds. '
                         'Increase --search-seeds or --sim-time, or review NAV/CCA and deployment. '
                         'No fabricated snapshot or misleading plots created.')
    snap = best.visualization_snapshot
    cluster_aps = {}
    for stream in snap.streams:
        cluster_aps.setdefault(stream.cluster_id, set()).add(stream.tx_node_id)
    actual_cluster_aps = {cid: aps for cid, aps in cluster_aps.items() if len(aps) >= 2}
    assert len(actual_cluster_aps) >= 2, 'Snapshot must have 2 actual Co-BF clusters'

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    plot_snapshot_topology(best, out / 'two_cluster_topology.png')
    plot_snapshot_beams(best, out / 'beam_snapshot_all_aps.png')
    plot_snapshot_sinr_heatmap(best, out / 'focus_cluster_sinr_heatmaps.png', grid_res=grid)
    plot_mode_b_ici_verification(best, out / 'ici_verification.png', grid_res=grid)
    export_mode_b_ici_breakdown(best, out / 'focus_cluster_ici_breakdown.csv')
    export_mode_b_zf_leakage_matrix(best, out / 'intra_cluster_leakage_matrix.csv')
    export_mode_b_all_stream_interference(best, out / 'full_interference_matrix.csv')
    ap_ids = sorted({s.tx_node_id for s in snap.streams})
    for ap_id in ap_ids:
        plot_ap_ula_pattern(best, ap_id, out / f'ap{ap_id}_cartesian_polar.png')
        plot_ap_aggregate_eirp(best, ap_id, out / f'ap{ap_id}_total_eirp.png')

    info = {
        'seed': best_seed,
        'mode': snap.mode,
        'precoder': snap.precoder_scheme,
        'mmse_alpha': cfg.beamforming.mmse_alpha,
        'snapshot_time_ms': snap.time_us / 1000,
        'focus_cluster_id': snap.focus_cluster_id,
        'cluster_ap_ids': {str(k): sorted(v) for k,v in cluster_aps.items()},
        'max_focus_sta_ici_penalty_db': snap.max_ici_penalty_db,
        'avg_focus_sta_ici_penalty_db': snap.avg_ici_penalty_db,
        'sim_throughput_mbps': best_result.total_throughput_mbps,
        'channel': {
            'frequency_hz': cfg.channel.center_frequency_hz,
            'bandwidth_hz': cfg.channel.bandwidth_hz,
            'path_loss_exponent': cfg.channel.path_loss_exponent,
            'shadowing_std_db': cfg.channel.shadowing_std_db,
            'array_element_spacing_wavelengths': cfg.channel.element_spacing_wavelengths,
            'noise_dbm': best.channel.noise_dbm,
        },
        'ap_locations': [
            {'id': a.id, 'x_m': a.x, 'y_m': a.y,
             'orientation_rad': a.array_orientation_rad}
            for a in best.topology.aps
        ],
        'sta_locations': [
            {'id': sta.id, 'x_m': sta.x, 'y_m': sta.y,
             'associated_ap': sta.serving_ap}
            for sta in best.topology.stas
        ],
        'num_qualifying_events': best.two_cluster_snapshot_candidates,
        'reduced_loading_txops': best.reduced_stream_txops,
        'aborted_infeasible_txops': best.precoder_infeasible_txops,
        'active_streams': [
            {'tx_id': s.tx_id, 'ap_id': s.tx_node_id, 'sta_id': s.rx_node_id,
             'cluster_id': s.cluster_id, 'tx_power_mw': s.tx_power_mw,
             'sinr_at_sta_db': s.snapshot_sinr_db,
             'beam_vector_real_imag': [[float(z.real), float(z.imag)]
                                      for z in s.beamforming_vector]}
            for s in snap.streams
        ],
    }
    (out / 'snapshot_data.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'\nSELECTED SEED: {best_seed}; 2 genuine concurrent clusters: '
          f'{info["cluster_ap_ids"]}; max ICI loss: {snap.max_ici_penalty_db:.2f} dB')
    print(f'Figures and exact snapshot data: {out.resolve()}')


if __name__ == '__main__':
    main()
