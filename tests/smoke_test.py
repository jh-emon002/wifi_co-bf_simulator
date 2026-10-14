"""Numerical/physical/MAC validation for single-STA-antenna Co-BF LOS simulator.

Run: python tests/smoke_test.py (no pytest required).
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from wifi_sim.config import Config, load_config
from wifi_sim.models import Topology
from wifi_sim.channel import WirelessChannel
from wifi_sim.beamforming import (
    MIN_MCS_SINR_DB, MCS_TABLE, calculate_cluster_precoders,
    calculate_mrt_beamformer, InfeasiblePrecoderError,
)
from wifi_sim.beam_visualization import (
    _rx_power_grid, _actual_sta_mode_b_breakdown,
    array_gain_from_weights, plot_ap_ula_pattern, compute_mode_b_ici_components_grid,
)
from wifi_sim.simulator import Simulator
from wifi_sim.experiment import build_topology, run_all_modes


def mk(seed=12345,scheme='zf',sim_time=.40,alpha=.01):
    cfg=load_config()
    cfg.beamforming.scheme=scheme
    cfg.beamforming.mmse_alpha=alpha
    cfg.simulation.sim_time_s=sim_time
    topology=build_topology(cfg,seed)
    channel=WirelessChannel(topology,cfg.channel,cfg.rf,seed)
    return Simulator(topology,channel,cfg,mode='B',seed=seed)


def check_los_mrt():
    sim=mk()
    assert sim.config.rf.sta_antennas==1 and sim.config.rf.ap_antennas==4
    assert sim.channel.scatterers==[]
    ap=next(a for a in sim.topology.aps if len(a.associated_stas))
    sta=sim.topology.nodes[ap.associated_stas[0]]
    h=sim.channel.sample_channel_matrix(sta.id,ap.id)
    w=calculate_mrt_beamformer(h)
    assert h.shape==(1,4)
    assert w.shape==(4,)
    assert not hasattr(w,'receive_combiner')
    theta=sim.channel.bearing(ap.x,ap.y,sta.x,sta.y)
    array_gain=float(array_gain_from_weights(sim,ap.id,w,theta))
    assert np.isclose(array_gain,4.0,atol=1e-10),array_gain
    G=sim.channel.path_gain_to_position(ap.id,sta.x,sta.y)
    assert np.isclose(abs((h@w).item())**2,G*array_gain,rtol=1e-11)
    assert math.isclose(float(array_gain_from_weights(sim,ap.id,w,theta)),
                        float(array_gain_from_weights(sim,ap.id,w,-theta)),
                        rel_tol=1e-12), 'ULA mirror ambiguity test'
    print('[PASS] scalar LOS channel, four-antenna array gain, phase convention, ULA symmetry')


def check_optional_shadowing_grid():
    cfg=load_config()
    cfg.channel.shadowing_std_db=2.0
    topology=build_topology(cfg,12345)
    channel=WirelessChannel(topology,cfg.channel,cfg.rf,seed=12345)
    ap=next(a for a in topology.aps if a.associated_stas)
    sta=topology.nodes[ap.associated_stas[0]]
    h=channel.sample_channel_matrix(sta.id,ap.id)
    w=calculate_mrt_beamformer(h)
    sim=Simulator(topology,channel,cfg,mode='B',seed=12345)
    from wifi_sim.models import SnapshotStream
    stream=SnapshotStream(tx_id=1,txop_id=1,tx_node_id=ap.id,rx_node_id=sta.id,
                          is_cobf=True,cluster_id=1,tx_power_mw=channel.ap_tx_power_mw,
                          beamforming_vector=w,initial_sinr_db=0,snapshot_sinr_db=0)
    grid=float(_rx_power_grid(sim,stream,np.array([[sta.x]]),np.array([[sta.y]])).item())
    physical=stream.tx_power_mw*float(abs((h@w).item())**2)
    assert math.isclose(grid,physical,rel_tol=1e-10)
    print('[PASS] optional deterministic shadowing gives exactly the same channel power on grid and STA')


def check_zf_mmse():
    sim=mk()
    t=sim.topology
    cooperating=[ap for ap in t.aps if len(ap.associated_stas)>=2][:2]
    assert len(cooperating)==2
    assignments=[(ap.id,st) for ap in cooperating for st in ap.associated_stas[:2]]
    stas=[st for _,st in assignments]
    H={a.id:{s:sim.channel.sample_channel_matrix(s,a.id) for s in stas}
       for a in cooperating}
    zf=calculate_cluster_precoders(H,assignments,'zf')
    mm=calculate_cluster_precoders(H,assignments,'mmse',mmse_alpha=.1)
    assert set(zf)==set(mm)==set(assignments)
    mm_nonzero_leakage=0
    for (ap,desired),beam in zf.items():
        assert np.isclose(np.linalg.norm(beam),1)
        for victim in stas:
            if victim==desired: continue
            power=abs((H[ap][victim] @ beam).item())**2
            assert power<1e-20,(ap,desired,victim,power)
    for (ap,desired),beam in mm.items():
        assert np.isclose(np.linalg.norm(beam),1)
        mm_nonzero_leakage += sum(abs((H[ap][victim] @ beam).item())**2
                                  for victim in stas if victim!=desired)
    assert mm_nonzero_leakage>1e-25
    assert math.isclose(MIN_MCS_SINR_DB,min(e[2] for e in MCS_TABLE))
    print('[PASS] true per-AP intra-cluster ZF zero leakage; RZF soft suppression; min-MCS NAV threshold')


def check_snapshot_numerics():
    sim=mk(seed=12345,sim_time=.40)
    metrics=sim.run()
    snap=sim.visualization_snapshot
    assert snap is not None, 'Seed 12345 should include two actual overlapping clusters'
    clustered={}
    for s in snap.streams:
        clustered.setdefault(s.cluster_id,set()).add(s.tx_node_id)
    assert sum(len(v)==2 for v in clustered.values())>=2,clustered
    assert sim.two_cluster_snapshot_candidates>0
    assert metrics.total_throughput_mbps>0
    assert math.isclose(metrics.total_throughput_mbps,
                        sum(metrics.per_ap_total_mbps.values()),rel_tol=1e-10)
    for stream in snap.streams:
        ap_streams=[s for s in snap.streams if s.tx_node_id==stream.tx_node_id]
        assert len(ap_streams)<=2
        assert sum(s.tx_power_mw for s in ap_streams)<=sim.channel.ap_tx_power_mw+1e-9
        assert np.isclose(np.linalg.norm(stream.beamforming_vector),1)
    for stream in [s for s in snap.streams if s.txop_id==snap.focus_txop_id]:
        b=_actual_sta_mode_b_breakdown(sim,stream)
        assert math.isclose(b['sinr_with_ici_db'],stream.snapshot_sinr_db,
                            abs_tol=1e-8), (b['sinr_with_ici_db'],stream.snapshot_sinr_db)
        assert b['ici_penalty_db']>=0
        sta=sim.topology.nodes[stream.rx_node_id]
        grid_power=_rx_power_grid(sim,stream,np.array([[sta.x]]),np.array([[sta.y]]))
        assert np.isclose(float(grid_power[0,0]),b['signal_mw'],rtol=1e-9)
        grid=compute_mode_b_ici_components_grid(sim,stream,grid_res=12)
        assert np.min(grid['ici_penalty_db'])>=-1e-8
        assert np.all(np.isfinite(grid['sinr_with_ici_db']))
    with TemporaryDirectory() as d:
        first_ap=snap.streams[0].tx_node_id
        fig=Path(d)/'beam.png'
        plot_ap_ula_pattern(sim,first_ap,fig)
        assert fig.stat().st_size>10_000
    print('[PASS] real two-cluster snapshot, exact actual-STA SINR, power budgets, grid physics, plotted figure')


def check_nav_and_mmse():
    sim=mk(seed=12358,scheme='mmse',sim_time=1.4,alpha=.1)
    candidates=[a for a in sim.topology.aps if a.associated_stas]
    a,b=candidates[:2]
    assert math.isfinite(sim._calculate_nav_sinr_db(a.id,b.id,{a.id}))
    assert sim._omnidirectional_nav_rx_power_mw(a.id,b.id)>0
    results=sim.run()
    assert math.isfinite(results.total_throughput_mbps)
    assert sim.visualization_snapshot is not None,'MMSE test seed must contain real overlap'
    for s in sim.visualization_snapshot.streams:
        assert not hasattr(s,'receive_combiner')
    print('[PASS] MMSE/RZF end-to-end plus existing omni NAV and real two-cluster overlap')


def check_three_modes():
    cfg=load_config()
    cfg.simulation.sim_time_s=.15
    _,data,_=run_all_modes(cfg,seed=12345)
    for mode in ['A','B','C']:
        assert math.isfinite(data[mode].total_throughput_mbps)
        assert data[mode].total_throughput_mbps>=0
    print('[PASS] A/B/C simulation comparison still operates')


def main():
    check_los_mrt()
    check_optional_shadowing_grid()
    check_zf_mmse()
    check_snapshot_numerics()
    check_nav_and_mmse()
    check_three_modes()
    print('ALL EXPLAINABLE SIMULATOR SMOKE TESTS PASSED')

if __name__=='__main__':main()
