# Explainable Wi-Fi 8 Multi-AP Co-BF Simulator — Version 3

A runnable **downlink**, geometry-based, discrete-event Wi-Fi multi-AP experiment for studying **inter-cluster interference (ICI)** under coordinated beamforming (Co-BF). The simulator preserves CSMA/CA, AP pairing, physical carrier sensing, omnidirectional NAV advertisement, and per-AP throughput accounting from the user's earlier angle-based simulator. It adds single-antenna STAs, genuinely simultaneous two-cluster snapshots, user-selectable ZF / MMSE-style (RZF) precoders, and test_ula-style physical beam-pattern diagnostics.



## 1. Configuration choices

| Feature | Default |
|---|---|
| Deployment | Random 10 APs / 30 STAs in 100 m × 100 m, reproducible by seed |
| AP antenna configuration | 4-element ULA with half-wavelength spacing |
| STA antenna configuration | **1 omnidirectional antenna, no receive combining** |
| Channel | **One geometric LOS path only; no reflectors/scatterers** |
| Shadowing | Disabled (`shadowing_std_db: 0`) for verification; optional large-scale field |
| Path loss | Reference-distance Friis scale plus exponent 3 (empirical indoor attenuation; change to 2 for ideal free space) |
| AP power | 20 dBm **total conducted power per AP**, split equally among 1–2 streams |
| Mode A | CSMA/CA with MRT; one STA per TXOP |
| Mode B | Two-AP Co-BF with **all active external-cluster interference** |
| Mode C | Counterfactual idealization: Mode B PHY/MAC except **external ICI omitted from receiver SINR** |
| Co-BF users | Up to 2 own STAs per AP; up to 4 streams per cluster |
| Co-BF precoders | `zf` or `mmse` (regularized ZF) |
| NAV | Omnidirectional announcement, decode if NAV SINR >= minimum data-MCS SINR (currently 4 dB) |
| Snapshot condition | Only when **at least two distinct two-AP Co-BF clusters actually transmit data simultaneously** (B/C) |

### Important Co-BF distinction: no joint data transmission

Each AP transmits **only to STAs associated with that AP**. The cooperating APs share CSI, and each AP uses all same-cluster STA channel rows to form interference-aware transmit beams. It is **not** joint transmission / network MIMO with every AP sending every user's payload. This choice keeps the model aligned with coordinated beamforming rather than covertly changing the physical transmission mode.

## 2. PHY equations

For one downlink stream from AP `a` to single-antenna STA `k`,

\[
\mathbf h_{k,a} = \sqrt{N_t G_{k,a}}\, e^{-j2\pi r/\lambda}\, \mathbf a_a(\theta_{k,a})^H,
\qquad \|\mathbf a_a(\theta)\|_2=1.
\]

The desired received power is `P_stream * |h @ w|^2`, with `||w||2=1`, and all independent interfering streams contribute powers using **the same actual channel and instantaneous beam vectors**:

\[
\mathrm{SINR}_k = \frac{P_k |\mathbf h_{k,a}\mathbf w_k|^2}
{N_0 B F + \sum_{j\ne k}P_j |\mathbf h_{k,a(j)}\mathbf w_j|^2}.
\]

There is no STA receive filter or combiner. A 4-element half-wavelength ULA can have **front/back (mirror-angle) ambiguity**: symmetric beams and nulls in 360° are expected consequences of this array geometry, not necessarily errors.

### Beamforming choice (`config.yaml`)

For each participating AP `a`, form a `K × 4` channel matrix `H_a` containing **every scheduled intra-cluster STA**, including users served by its partner AP:

- **ZF**: for each **locally served** stream, project the desired channel vector onto the null space of the other `K-1` channel rows. This enforces `h_victim,a @ w_a,k = 0` for all other scheduled same-cluster users, to numerical precision when feasible.
- **MMSE/RZF**: form `V_a = H_a^H (H_a H_a^H + λ_a I)^{-1}` and select only columns for **locally served** STAs. The regularizer is `λ_a = mmse_alpha × tr(H_a H_a^H) / K`. This is a tunable, equal-power **MMSE-style / RZF** scheme, not a claim of globally optimal weighted-MMSE precoding. It reduces hard nulling as alpha increases.

Each output beam is unit norm. Total transmitted AP power is `sum_streams(P_stream * ||w||²) = P_AP`. This is the fair **per-AP conducted-power** comparison; **derived aggregate EIRP** is also plotted. Peak EIRP can exceed conducted power because of array gain; this example does **not** impose a jurisdiction-specific EIRP limit.

If a particular geometry makes exact ZF infeasible, the event engine removes a stream while keeping at least one STA per AP (counted in `reduced_stream_txops`), or aborts the infeasible TXOP (counted in `aborted_infeasible_txops`). The simulator **never silently substitutes an MRT beam for ZF**.

## 3. Installation

Use Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

From the project root, choose `beamforming.scheme: zf` or `mmse` in `config.yaml`. For MMSE-style RZF, set `beamforming.mmse_alpha` (e.g., `0.001`, `0.01`, `0.1`), **a dimensionless relative regularization**.

## 4. First, validate

```bash
python tests/smoke_test.py
python scripts/channel_diagnostics.py
```

Tests verify the analytical LOS/ULA gain relationship; single-STA-antenna PHY; perfect feasible intra-cluster ZF; RZF soft suppression; coherent grid-vs-actual-STA received power; simultaneous true two-cluster snapshot; NAV threshold; strict per-AP transmit power; and all three MAC/PHY modes. The tests are designed to fail on a channel/plot discrepancy.

## 5. Generate the real Mode B snapshot, gains and interference plots

```bash
python scripts/generate_mode_b_ici_verification.py
```

This tries consecutive **random deployment** seeds until a genuine two-cluster data-TX overlap is found. The default target is at least a 3 dB peak external ICI SINR penalty; the script can accept any valid overlap if the target is not reached in its search window. It **does not fabricate concurrent clusters** or bypass CCA/NAV.

For faster verification using a known overlapping LOS topology:

```bash
python scripts/generate_mode_b_ici_verification.py --seed 12345 --search-seeds 1 --sim-time 0.6 --grid-res 80
```

To visualize RZF instead of exact ZF:

```bash
python scripts/generate_mode_b_ici_verification.py --scheme mmse --mmse-alpha 0.1 --seed 12358 --search-seeds 1 --sim-time 1.4 --out results/figures/mode_b_mmse
```

The seed and captured TXOP determine the physical positions and active streams. Random deployment does **not** mean that every seed contains two simultaneous clusters; if none occur, the program exits with a clear diagnostic instead of showing a fake "ICI" plot.

### Outputs (under `results/figures/mode_b/`)

- `two_cluster_topology.png`: both actual active Co-BF clusters and served STAs at the **same instant**.
- `beam_snapshot_all_aps.png`: one Cartesian overview for each transmitting AP.
- `ap<ID>_cartesian_polar.png`: requested **test_ula-style** plots with green circles at desired beam targets, orange crosses for other **scheduled intra-cluster STA directions**, red squares for **scheduled external-cluster STA directions**, and small gray circles for **nearby inactive STA positions**. Orange marks are hard ZF null constraints, but merely suppressed under MMSE/RZF. Pattern curves are **real array gain in dBi**, not each beam independently peak-normalized; polar radius is **linear gain**.
- `ap<ID>_total_eirp.png`: physically scaled aggregate AP EIRP (dBm) across its independent spatial streams. This makes transmit-power accounting explicit.
- `focus_cluster_sinr_heatmaps.png`: per-STA single-antenna SINR with real concurrent interferers.
- `ici_verification.png`: five maps for the **actual focus STA with the largest ICI penalty**: desired power, external ICI, SINR with ICI, counterfactual no-external-ICI SINR, and ICI penalty.
- `focus_cluster_ici_breakdown.csv`: actual-STA signal/intra/ICI/noise/SINR decomposition.
- `intra_cluster_leakage_matrix.csv`: actual `|h_victim @ w|²` ZF vs RZF soft-leakage matrix.
- `full_interference_matrix.csv`: every **active beam→active STA** coupling and physical ICI received power for both clusters.
- `snapshot_data.json`: snapshot seed, time, stream assignment, cluster membership, beam vectors, per-stream power and observed SINRs.

**Units and interpretation**: the array gain plots omit distance/path loss by design; the SINR and received-interference heatmaps include path loss, transmit power, and every active independent stream. A red STA marker on an AP's pattern is **not** a null constraint in this initial intra-cluster-only simulator.

## 6. Additional runs

```bash
python scripts/compare_modes.py --quick
python scripts/compare_modes.py --sim-time 3
python scripts/generate_figures.py --quick
python scripts/generate_beam_sinr_figures.py --quick
python scripts/compare_precoders.py --runs 5 --sim-time 2 --mmse-alpha 0.01
python scripts/run_monte_carlo.py --runs 20 --sim-time 3
python scripts/sweep_mmse_alpha.py --runs 10 --sim-time 1 --alphas 0.001 0.01 0.1
```

`compare_precoders.py` records paired random seeds across algorithms/modes in `results/summaries/zf_mmse_comparison.csv`, including whether there was a genuine two-cluster snapshot. It does not pretend that subsequent MAC contention trajectories remain identical after different beamforming decisions. Mode C is a **counterfactual PHY upper-bound at a fixed captured state**; its end-to-end throughput need not be an exact paired upper bound for Mode B if the MAC evolves differently.



## 7. Project tree

```text
wifi8_cobf_explainable_v3/
  config.yaml
  pyproject.toml
  requirements.txt
  wifi_sim/
    config.py, models.py, channel.py, beamforming.py, simulator.py
    beam_visualization.py, plotting.py, metrics.py, experiment.py, __init__.py
  scripts/
    generate_mode_b_ici_verification.py, channel_diagnostics.py
    generate_beam_sinr_figures.py, generate_figures.py, compare_modes.py
    compare_precoders.py, sweep_mmse_alpha.py, run_monte_carlo.py, smoke_test.py
  tests/
    smoke_test.py
  results/
    figures/mode_b/  # sample evidence from verified seed 12345
```
