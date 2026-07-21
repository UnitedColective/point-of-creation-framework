"""
run_dmt_bandpower.py
====================
PCF δ and R computed on EEG band-power time series.

Why band-power, not raw EEG
----------------------------
δ = ρ₁(Δ²X) + 2/3 was validated on event-sequence data where consecutive
samples carry physiologically meaningful temporal memory (prime gaps, RR
intervals, hourly gene expression). Raw EEG at 500 Hz is too fast — the
second difference is indistinguishable from white noise. Narrowband filtered
EEG is too sinusoidal — δ saturates near 1.66 (the pure sine limit).

The correct application is to the POWER ENVELOPE: how does oscillatory
power in each band fluctuate over time? This gives a slow sequence
(one value per second) that carries real temporal structure — exactly
the regime where δ is meaningful.

Method
------
For each subject, for each frequency band:
  1. Highpass filter raw EEG at 0.5 Hz (remove DC)
  2. Bandpass filter into band
  3. Compute instantaneous power via Hilbert transform (power = |analytic|²)
  4. Average power in 1-second windows → power envelope time series
  5. Compute δ and R on that envelope

Then compare DMT vs EC across subjects with paired t-test + shuffle control.

Frequency bands
---------------
  delta : 1–4 Hz
  theta : 4–8 Hz
  alpha : 8–12 Hz
  beta  : 12–30 Hz
  gamma : 30–80 Hz

Expected DMT effects (from Timmermann et al. 2019, Cavanna et al. 2021):
  alpha power DECREASES under DMT
  theta and gamma power INCREASE under DMT
  delta power shows mixed results

Whether δ on the power envelope tracks these changes is an open question.
The shuffle control is the integrity check.

Usage
-----
  python run_dmt_bandpower.py --data_dir dmt_data
  python run_dmt_bandpower.py --data_dir dmt_data --save dmt_bandpower.csv

Requirements
------------
  pip install mne numpy scipy

Reference
---------
  Dataset: Cavanna et al. (2021) J Psychopharmacology
  Zenodo:  https://zenodo.org/records/3992359

  PCF Framework: Sterling A.M. (2026)
  DOI: https://doi.org/10.5281/zenodo.20984797
"""

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
from scipy import signal, stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Subject file pairs — verified against Zenodo metadata
# ---------------------------------------------------------------------------
PAIRS = [
    ( 1, "S01-DMT.bdf",  "S01-EC.bdf"),
    ( 2, "S02_DMT.bdf",  "S02_EC.bdf"),
    ( 3, "S03-DMT.bdf",  "S03-EC.bdf"),
    ( 4, "s04-DMT.bdf",  "s04-EC.bdf"),
    ( 5, "s05-DMT.bdf",  "s05-EC.bdf"),
    ( 6, "S06_DMT.bdf",  "S06_EC.bdf"),
    ( 7, "S07_DMT.bdf",  "S07_EC.bdf"),
    ( 8, "S08_DMT.bdf",  "S08_EC.bdf"),
    ( 9, "S09_DMT.bdf",  "S09_EC.bdf"),
    (10, "S10_DMT.bdf",  "S10_EC.bdf"),
    (11, "S11-DMT.bdf",  "S11-EC.bdf"),
    (12, "S12-DMT.bdf",  "S12-EC.bdf"),
    (13, "S13-DMT.bdf",  "S13-EC.bdf"),
    (14, "S14-DMT.bdf",  "S14-EC.bdf"),
    (15, "S15-DMT.bdf",  "S15-EC.bdf"),
    (16, "S16_DMT.bdf",  "S16_EC.bdf"),
    (17, "S17_DMT.bdf",  "S17_EC.bdf"),
    (18, "S18_DMT.bdf",  "S18_EC.bdf"),
    (19, "S19_DMT.bdf",  "S19_EC.bdf"),
    (20, "S20_DMT.bdf",  "S20_EC.bdf"),
    (22, "S22-DMT.bdf",  "S22-EC.bdf"),
    (23, "S23-DMT.bdf",  "S23-Ec.bdf"),
    (24, "S24-DMT.bdf",  "S24-EC.bdf"),
    (25, "S25_DMT.bdf",  "S25_EC.bdf"),
    (26, "S26_DMT.bdf",  "S26_EC.bdf"),
    (27, "S27_DMT.bdf",  "S27_EC.bdf"),
    (28, "S28_DMT.bdf",  "S28_EC.bdf"),
    (29, "S29-DMT.bdf",  "S29-EC.bdf"),
    (30, "S30-DMT.bdf",  "S30-EC.bdf"),
    (31, "S31-DMT.bdf",  "S31-EC.bdf"),
    (32, "S32-DMT.bdf",  "S32-EC.bdf"),
    (33, "S33-DMT.bdf",  "S33-EC.bdf"),
    (34, "S34-DMT.bdf",  "S34-EC.bdf"),
    (35, "S35-DMT.bdf",  "S35-EC.bdf"),
]

BANDS = {
    "delta": (1,   4),
    "theta": (4,   8),
    "alpha": (8,  12),
    "beta":  (12, 30),
    "gamma": (30, 80),
}

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_delta(ts):
    ts = np.asarray(ts, dtype=float)
    d2 = np.diff(np.diff(ts))
    if len(d2) < 3:
        return float("nan")
    rho1 = float(np.corrcoef(d2[:-1], d2[1:])[0, 1])
    return rho1 + 2.0 / 3.0


def compute_R(ts):
    ts = np.asarray(ts, dtype=float)
    if len(ts) < 10:
        return float("nan")
    phase = np.angle(signal.hilbert(ts))
    return float(np.abs(np.mean(np.exp(1j * phase))))


def band_power_envelope(eeg, fs, flo, fhi, window_sec=1.0):
    """
    Bandpass filter EEG, compute instantaneous power via Hilbert,
    then average into non-overlapping windows.
    Returns power envelope as 1D array (one value per window).
    """
    sos = signal.butter(4, [flo, fhi], btype="band", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, eeg)
    power = np.abs(signal.hilbert(filtered)) ** 2
    win = int(window_sec * fs)
    n_windows = len(power) // win
    envelope = np.array([
        power[i*win:(i+1)*win].mean()
        for i in range(n_windows)
    ])
    return envelope


# ---------------------------------------------------------------------------
# BDF loading
# ---------------------------------------------------------------------------

def load_eeg(bdf_path, hp=0.5):
    import mne
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
    raw.filter(hp, None, verbose=False)
    fs = raw.info["sfreq"]
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(picks) == 0:
        picks = [0]
    eeg = raw.get_data(picks=[picks[0]])[0]
    return eeg, float(fs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_dir, window_sec=1.0, n_shuffle=5000, save_path=None):
    data_dir = Path(data_dir)

    print(f"\n{'='*60}")
    print("PCF δ on Band-Power Envelopes — DMT vs EC Baseline")
    print("Cavanna et al. (2021) / Zenodo 3992359")
    print(f"{'='*60}")
    print(f"Power window: {window_sec}s | Shuffle: {n_shuffle} iterations\n")

    # Per-subject results: band -> list of (delta_dmt, delta_ec, R_dmt, R_ec)
    band_results = {b: [] for b in BANDS}
    skipped = []

    for sid, dmt_f, ec_f in PAIRS:
        dmt_path = data_dir / dmt_f
        ec_path  = data_dir / ec_f
        if not dmt_path.exists() or not ec_path.exists():
            skipped.append(sid)
            continue

        try:
            dmt_eeg, fs = load_eeg(str(dmt_path))
            ec_eeg,  _  = load_eeg(str(ec_path))

            row_parts = [f"S{sid:02d}"]
            for band, (flo, fhi) in BANDS.items():
                env_dmt = band_power_envelope(dmt_eeg, fs, flo, fhi, window_sec)
                env_ec  = band_power_envelope(ec_eeg,  fs, flo, fhi, window_sec)

                d_dmt = compute_delta(env_dmt)
                d_ec  = compute_delta(env_ec)
                r_dmt = compute_R(env_dmt)
                r_ec  = compute_R(env_ec)

                band_results[band].append({
                    "subject": sid,
                    "delta_dmt": d_dmt, "delta_ec": d_ec,
                    "R_dmt": r_dmt,     "R_ec": r_ec,
                    "delta_diff": d_dmt - d_ec,
                    "R_diff": r_dmt - r_ec,
                })
                row_parts.append(f"{band} δΔ={d_dmt-d_ec:+.3f}")

            print("  " + "  ".join(row_parts))

        except Exception as e:
            print(f"  S{sid:02d}  ERROR: {e}")
            skipped.append(sid)

    if skipped:
        print(f"\n  Skipped: {skipped}")

    # ---------------------------------------------------------------------------
    # Statistics per band
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("RESULTS BY FREQUENCY BAND")
    print(f"{'='*60}")

    summary_rows = []
    rng = np.random.default_rng(seed=0)

    for band in BANDS:
        rows = band_results[band]
        if len(rows) < 3:
            continue

        d_dmt = np.array([r["delta_dmt"] for r in rows])
        d_ec  = np.array([r["delta_ec"]  for r in rows])
        r_dmt = np.array([r["R_dmt"]     for r in rows])
        r_ec  = np.array([r["R_ec"]      for r in rows])
        diff  = d_dmt - d_ec

        t_d, p_d = stats.ttest_rel(d_dmt, d_ec)
        t_r, p_r = stats.ttest_rel(r_dmt, r_ec)

        # Shuffle control on δ difference
        null = []
        for _ in range(n_shuffle):
            signs = rng.choice([-1, 1], size=len(diff))
            null.append(float((signs * diff).mean()))
        p_perm = float(np.mean(np.abs(null) >= abs(diff.mean())))

        sig_t    = "***" if p_d < 0.001 else ("**" if p_d < 0.01 else ("*" if p_d < 0.05 else "ns"))
        sig_perm = "***" if p_perm < 0.001 else ("**" if p_perm < 0.01 else ("*" if p_perm < 0.05 else "ns"))

        print(f"\n  {band.upper()} ({BANDS[band][0]}–{BANDS[band][1]} Hz)  N={len(rows)}")
        print(f"    δ  EC={d_ec.mean():+.4f}±{d_ec.std():.4f}  "
              f"DMT={d_dmt.mean():+.4f}±{d_dmt.std():.4f}  "
              f"Δ={diff.mean():+.4f}  t={t_d:+.3f}  p={p_d:.4g} {sig_t}  "
              f"shuffle_p={p_perm:.4g} {sig_perm}")
        print(f"    R  EC={r_ec.mean():.4f}±{r_ec.std():.4f}  "
              f"DMT={r_dmt.mean():.4f}±{r_dmt.std():.4f}  "
              f"t={t_r:+.3f}  p={p_r:.4g}")

        summary_rows.append({
            "band": band,
            "flo": BANDS[band][0], "fhi": BANDS[band][1],
            "n": len(rows),
            "delta_ec_mean": float(d_ec.mean()),
            "delta_ec_sd":   float(d_ec.std()),
            "delta_dmt_mean": float(d_dmt.mean()),
            "delta_dmt_sd":   float(d_dmt.std()),
            "delta_diff_mean": float(diff.mean()),
            "delta_diff_sd":   float(diff.std()),
            "delta_t": float(t_d), "delta_p": float(p_d),
            "delta_shuffle_p": float(p_perm),
            "R_ec_mean":  float(r_ec.mean()),
            "R_dmt_mean": float(r_dmt.mean()),
            "R_t": float(t_r), "R_p": float(p_r),
        })

    # ---------------------------------------------------------------------------
    # Honest interpretation
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("HONEST INTERPRETATION")
    print(f"{'='*60}")
    print("""
  δ is computed on the power ENVELOPE time series, not raw EEG.
  Each data point = mean band power in a 1-second window.
  A recording of ~5 min gives ~300 data points per condition.

  δ > 0 means the power envelope has temporal memory above noise.
  δ change DMT vs EC means the DYNAMICS of oscillatory power shifted.

  This is a first exploratory run. No artifact rejection applied.
  Significant results should be replicated with proper preprocessing.

  Bands showing shuffle_p < 0.05 have passed the primary integrity check.
  Bands showing shuffle_p >= 0.05 are null results — report them as such.
    """)

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    if save_path and summary_rows:
        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Summary saved to: {save_path}")

    print(f"\n{'='*60}\n")
    return summary_rows


def parse_args():
    p = argparse.ArgumentParser(
        description="PCF δ on band-power envelopes — DMT vs baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",    default="dmt_data")
    p.add_argument("--window_sec",  type=float, default=1.0,
                   help="Power averaging window in seconds")
    p.add_argument("--n_shuffle",   type=int,   default=5000)
    p.add_argument("--save",        default=None,
                   help="Save band summary CSV to this path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        data_dir   = args.data_dir,
        window_sec = args.window_sec,
        n_shuffle  = args.n_shuffle,
        save_path  = args.save,
    )
