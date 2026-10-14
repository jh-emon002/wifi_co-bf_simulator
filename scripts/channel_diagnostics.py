"""Diagnostics for the single-LOS-path, 1-antenna-STA channel."""
from __future__ import annotations
import math
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wifi_sim.channel import WirelessChannel
from wifi_sim.config import load_config
from wifi_sim.experiment import build_topology
from wifi_sim.beamforming import calculate_mrt_beamformer
from wifi_sim.beam_visualization import array_gain_from_weights

def main():
    cfg=load_config()
    topo=build_topology(cfg,cfg.network.seed)
    ch=WirelessChannel(topo,cfg.channel,cfg.rf,cfg.network.seed)
    ap=next(a for a in topo.aps if a.associated_stas)
    sta=topo.nodes[ap.associated_stas[0]]
    path=ch.path_components_to_position(ap.id,sta.x,sta.y)[0]
    h=ch.sample_channel_matrix(sta.id,ap.id)
    w=calculate_mrt_beamformer(h)
    direction=path.aod_rad
    gain=float(array_gain_from_weights(type('S',(),{'topology':topo,'config':cfg})(),
                                       ap.id,w,direction))
    direct_power=float(abs((h@w).item())**2)
    assert math.isclose(direct_power,path.gain_linear*gain,rel_tol=1e-9), 'Physical steering/gain inconsistency'
    assert h.shape==(1,4)
    print(f'LOS-only: AP{ap.id} -> STA{sta.id}, distance={path.length_m:.2f}m, '
          f'AoD={math.degrees(direction):.2f}deg')
    print(f'channel shape={h.shape}, path gain={10*math.log10(path.gain_linear):.2f}dB')
    print(f'MRT gain={10*math.log10(gain):.2f}dBi, noise={ch.noise_dbm:.2f}dBm')
    print('Analytical LOS path-gain * ULA-gain == measured |h @ w|^2: PASS')

if __name__=='__main__':
    main()
