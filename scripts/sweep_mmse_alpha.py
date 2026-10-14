"""No-RL alpha sensitivity study for Mode B under the SAME random seed set.

Explores a tuned RZF regularizer without claiming a globally optimal value.
Outputs per-run CSV, per-alpha 95% t CI and a throughput sensitivity chart.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from wifi_sim.config import load_config
from wifi_sim.channel import WirelessChannel
from wifi_sim.experiment import build_topology
from wifi_sim.simulator import Simulator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alphas',type=float,nargs='+',default=[0.001,0.003,0.01,0.03,0.1])
    p.add_argument('--runs',type=int,default=10)
    p.add_argument('--sim-time',type=float,default=1.0)
    p.add_argument('--seed',type=int,default=None)
    p.add_argument('--out',type=Path,default=ROOT/'results'/'summaries')
    args=p.parse_args()
    if any(a<=0 for a in args.alphas):p.error('All alpha values must be positive')
    cfg=load_config()
    cfg.simulation.sim_time_s=args.sim_time
    base=cfg.network.seed if args.seed is None else args.seed
    rows=[]
    for i in range(args.runs):
        seed=base+i
        # One topology/channel pair per algorithm; random realizations are
        # identically seeded. MAC trajectories may differ *because of* precoding.
        for scheme, alpha in [('zf',0.0)] + [('mmse',a) for a in args.alphas]:
            cfg.beamforming.scheme=scheme
            if scheme=='mmse':cfg.beamforming.mmse_alpha=alpha
            topology=build_topology(cfg,seed)
            channel=WirelessChannel(topology,cfg.channel,cfg.rf,seed=seed)
            sim=Simulator(topology,channel,cfg,'B',seed=seed)
            m=sim.run()
            snap=sim.visualization_snapshot
            rows.append({'seed':seed,'scheme':scheme,'mmse_alpha':alpha,
                         'throughput_mbps':m.total_throughput_mbps,
                         'stream_failure_rate':m.stream_failure_rate,
                         'mean_initial_sinr_db':m.mean_sinr_db,
                         'mean_simultaneous_aps':m.avg_simultaneous_tx,
                         'two_cluster_snapshot':int(snap is not None),
                         'max_ici_penalty_db_if_snapshot':
                         snap.max_ici_penalty_db if snap is not None else np.nan,
                         'reduced_loading_txops':sim.reduced_stream_txops,
                         'aborted_infeasible_txops':sim.precoder_infeasible_txops})
        print(f'Seed {seed} finished',flush=True)
    raw=pd.DataFrame(rows)
    summary=[]
    for (scheme,alpha),group in raw.groupby(['scheme','mmse_alpha'],sort=True):
        values=group['throughput_mbps'].to_numpy()
        mean=float(values.mean());sd=float(values.std(ddof=1)) if len(values)>1 else 0.0
        half_ci=stats.t.ppf(.975,len(values)-1)*sd/np.sqrt(len(values)) if len(values)>1 else 0.0
        summary.append({'scheme':scheme,'mmse_alpha':alpha,'runs':len(values),
                        'mean_throughput_mbps':mean,'ci95_throughput_mbps':half_ci,
                        'avg_stream_failure_rate':group['stream_failure_rate'].mean(),
                        'avg_simultaneous_aps':group['mean_simultaneous_aps'].mean(),
                        'fraction_true_two_cluster_snapshots':group['two_cluster_snapshot'].mean()})
    args.out.mkdir(parents=True,exist_ok=True)
    raw.to_csv(args.out/'mmse_alpha_sweep_raw.csv',index=False)
    summary=pd.DataFrame(summary)
    summary.to_csv(args.out/'mmse_alpha_sweep_summary.csv',index=False)
    rz=summary.loc[summary.scheme=='mmse'].sort_values('mmse_alpha')
    z=summary.loc[summary.scheme=='zf'].iloc[0]
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    ax.errorbar(rz['mmse_alpha'],rz['mean_throughput_mbps'],
                yerr=rz['ci95_throughput_mbps'],marker='o',capsize=4,label='MMSE/RZF')
    ax.axhline(z['mean_throughput_mbps'],ls='--',label=f'ZF mean, {z["mean_throughput_mbps"]:.1f} Mbps')
    ax.set_xscale('log')
    ax.set_xlabel('User-set relative MMSE regularization alpha (log scale)')
    ax.set_ylabel('Mode B throughput (Mbps), mean ± 95% CI')
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(args.out/'mmse_alpha_sweep.png',dpi=200,bbox_inches='tight')
    plt.close(fig)
    print(summary.to_string(index=False))
    print('Saved:',args.out.resolve())

if __name__=='__main__':main()
