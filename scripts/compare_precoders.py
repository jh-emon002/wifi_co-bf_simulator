"""Compare ZF and RZF on the same seeds and the same modes (separate trajectories)."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from wifi_sim.config import load_config
from wifi_sim.experiment import run_all_modes

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=None)
    p.add_argument('--runs',type=int,default=5)
    p.add_argument('--sim-time',type=float,default=1.0)
    p.add_argument('--mmse-alpha',type=float,default=None)
    p.add_argument('--out',type=Path,default=ROOT/'results'/'summaries'/'zf_mmse_comparison.csv')
    args=p.parse_args()
    cfg=load_config()
    cfg.simulation.sim_time_s=args.sim_time
    if args.mmse_alpha is not None:
        if args.mmse_alpha<=0:p.error('--mmse-alpha must be positive')
        cfg.beamforming.mmse_alpha=args.mmse_alpha
    base=cfg.network.seed if args.seed is None else args.seed
    rows=[]
    for seed in range(base,base+args.runs):
        for scheme in ('zf','mmse'):
            cfg.beamforming.scheme=scheme
            _,metrics,sims=run_all_modes(cfg,seed=seed)
            for mode, result in metrics.items():
                rows.append({'seed':seed,'precoder':scheme,'mmse_alpha':cfg.beamforming.mmse_alpha,
                             'mode':mode,'throughput_mbps':result.total_throughput_mbps,
                             'stream_failure_rate':result.stream_failure_rate,
                             'mean_initial_sinr_db':result.mean_sinr_db,
                             'avg_simultaneous_aps':result.avg_simultaneous_tx,
                             'has_true_two_cluster_snapshot': bool(sims[mode].visualization_snapshot)
                             if mode in ('B','C') else False,
                             'reduced_loading_txops': sims[mode].reduced_stream_txops,
                             'aborted_infeasible_txops': sims[mode].precoder_infeasible_txops})
            print(f'Seed {seed}, {scheme.upper()}: '
                  f'A={metrics["A"].total_throughput_mbps:.1f}, '
                  f'B={metrics["B"].total_throughput_mbps:.1f}, '
                  f'C={metrics["C"].total_throughput_mbps:.1f} Mbps',flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out,index=False)
    print(f'Exported: {args.out.resolve()}')

if __name__=='__main__':main()
