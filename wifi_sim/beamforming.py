"""AP-local coordinated multi-user precoding: ZF and regularized ZF (MMSE-style).

Co-BF here means CSI sharing, NOT payload data sharing: each AP only sends
streams to its own STAs. Every AP nevertheless considers all scheduled intra-
cluster STAs when calculating its beam vectors. No receive combining exists.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

import numpy as np

from wifi_sim.models import ActiveTransmission

MCS_TABLE: List[Tuple[int, float, float]] = [
    (0, 36.0, 4.0), (1, 72.1, 7.0), (2, 108.1, 9.0),
    (3, 144.1, 12.0), (4, 216.2, 16.0), (5, 288.2, 20.0),
    (6, 324.3, 22.0), (7, 360.3, 24.0), (8, 432.4, 28.0),
    (9, 480.4, 30.0), (10, 540.4, 33.0), (11, 600.5, 35.0),
    (12, 648.5, 39.0), (13, 720.6, 42.0),
]
MIN_MCS_SINR_DB = min(t for _, _, t in MCS_TABLE)


def select_mcs(sinr_db: float) -> Tuple[int, float, float]:
    eligible = [e for e in MCS_TABLE if sinr_db >= e[2]]
    return eligible[-1] if eligible else MCS_TABLE[0]


def calculate_mrt_beamformer(channel: np.ndarray) -> np.ndarray:
    """Conjugate scalar-row channel: w = h^H / ||h|| (one STA antenna)."""
    h = np.asarray(channel, dtype=complex).reshape(-1)
    norm = np.linalg.norm(h)
    if norm <= 0:
        raise ValueError('MRT cannot operate on zero channel')
    return h.conj() / norm


class InfeasiblePrecoderError(RuntimeError):
    """Requested perfect intra-cluster ZF is geometrically unachievable."""


def _zf_beam_for_victim_rows(channel_rows: np.ndarray, desired_index: int,
                             rank_tolerance: float,
                             min_desired_projection: float) -> np.ndarray:
    """Unit-norm null-space projection onto the desired STA's channel.

    Normalize constraint rows before the SVD: interference-null feasibility
    depends on angular rank, not on received pathloss magnitude.
    """
    h = np.asarray(channel_rows, dtype=complex)
    magnitudes = np.linalg.norm(h, axis=1)
    if np.any(magnitudes <= 0):
        raise InfeasiblePrecoderError('Zero or invalid channel row')
    directions = h / magnitudes[:, None]
    mask = np.arange(len(h)) != desired_index
    victims = directions[mask]
    wanted = directions[desired_index].conj()
    if len(victims):
        _, sing, vh = np.linalg.svd(victims, full_matrices=True)
        rank = int(np.sum(sing > rank_tolerance * (sing[0] if len(sing) else 1.0)))
        basis = vh[rank:, :].conj().T
        if basis.shape[1] == 0:
            raise InfeasiblePrecoderError('No spatial degrees of freedom for ZF')
        projection = basis @ (basis.conj().T @ wanted)
    else:
        projection = wanted
    length = float(np.linalg.norm(projection))
    if length < min_desired_projection:
        raise InfeasiblePrecoderError(
            f'Desired spatial direction lies in victim span (projection={length:.2e})'
        )
    beam = projection / length
    return beam


def calculate_cluster_precoders(
    channels_by_ap: Dict[int, Dict[int, np.ndarray]],
    stream_assignments: List[Tuple[int, int]],
    scheme: str = 'zf',
    mmse_alpha: float = 0.1,
    zf_rank_tolerance: float = 1e-9,
    zf_min_desired_projection: float = 1e-8,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Return beams keyed by (serving AP, STA), constrained per AP.

    ZF: null every other *intra-cluster* scheduled STA's 1xNt channel.
    MMSE/RZF: for each AP form H (K x Nt), then V = H^H (H H^H + lambda I)^-1,
    selecting only columns for that AP's own STAs. lambda = mmse_alpha *
    trace(H H^H) / K. This regularization is user-tunable and does not claim
    global network-wide WMMSE optimality. Each output column is unit norm;
    equal AP stream power is applied in the event simulator.
    """
    if not stream_assignments:
        return {}
    if scheme not in {'zf', 'mmse'}:
        raise ValueError('scheme must be zf or mmse')
    if scheme == 'mmse' and mmse_alpha <= 0:
        raise ValueError('mmse_alpha must be positive')
    scheduled_stas = [sta_id for _, sta_id in stream_assignments]
    if len(scheduled_stas) != len(set(scheduled_stas)):
        raise ValueError('Duplicate STA in one cluster')
    if len(stream_assignments) > 4:
        raise ValueError('At most four scheduled STAs in a two-AP cluster')

    beams: Dict[Tuple[int, int], np.ndarray] = {}
    ap_ids = {ap_id for ap_id, _ in stream_assignments}
    for ap_id in sorted(ap_ids):
        H = np.vstack([np.asarray(channels_by_ap[ap_id][sta_id], dtype=complex).reshape(1, -1)
                       for sta_id in scheduled_stas])
        if not np.isfinite(H).all():
            raise ValueError('Non-finite channel')
        if scheme == 'mmse':
            power_scale = float(np.mean(np.sum(np.abs(H) ** 2, axis=1)))
            if power_scale <= 0:
                raise InfeasiblePrecoderError('Zero channel norm')
            # Normalize to make alpha dimensionless and physically interpretable.
            Hn = H / math.sqrt(power_scale)
            gram = Hn @ Hn.conj().T
            W = Hn.conj().T @ np.linalg.solve(
                gram + mmse_alpha * np.eye(len(scheduled_stas)),
                np.eye(len(scheduled_stas), dtype=complex),
            )
        for j, sta_id in enumerate(scheduled_stas):
            if (ap_id, sta_id) not in stream_assignments:
                continue
            if scheme == 'zf':
                w = _zf_beam_for_victim_rows(H, j, zf_rank_tolerance,
                                              zf_min_desired_projection)
            else:
                w = W[:, j]
                norm = np.linalg.norm(w)
                if norm <= 1e-16:
                    raise InfeasiblePrecoderError('RZF beam numerically collapsed')
                w = w / norm
            beams[(ap_id, sta_id)] = w
    return beams


def calculate_sinr_dl(
    desired_tx_id: int,
    rx_sta_id: int,
    get_channel_fn: Callable[[ActiveTransmission, int], np.ndarray],
    active_txs: List[ActiveTransmission],
    mode: str,
    serving_cluster_id: int,
    p_noise_mw: float,
) -> Tuple[float, float]:
    """True one-antenna post-channel SINR; no receive combiner."""
    serving = next((tx for tx in active_txs if tx.tx_id == desired_tx_id), None)
    if serving is None:
        return 0.0, -120.0
    h = get_channel_fn(serving, rx_sta_id)
    desired = serving.tx_power_mw * float(abs((h @ serving.beamforming_vector).item()) ** 2)
    interference = 0.0
    for tx in active_txs:
        if tx.tx_id == desired_tx_id:
            continue
        if mode == 'C' and serving_cluster_id >= 0 and tx.cluster_id != serving_cluster_id:
            continue
        hi = get_channel_fn(tx, rx_sta_id)
        interference += tx.tx_power_mw * float(abs((hi @ tx.beamforming_vector).item()) ** 2)
    sinr = desired / max(p_noise_mw + interference, 1e-18)
    return sinr, 10.0 * math.log10(max(sinr, 1e-12))
