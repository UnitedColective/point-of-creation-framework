"""
run_dmt_pcf.py
==============
Runs PCF metrics (δ and R) on the Cavanna DMT EEG dataset.

For each subject:
  - Loads DMT.bdf and EC.bdf (eyes-closed baseline)
  - Extracts the first EEG channel (Fp1)
  - Cuts into 5-second non-overlapping epochs
  - Computes δ and R per epoch
  - Averages across epochs per subject per condition

Then compares DMT vs EC across all subjects with:
  - Paired t-test (same subjects, two conditions)
  - Shuffle/permutation control on δ

Honest framing
--------------
This is an exploratory analysis. We do not know whether δ will increase,
decrease, or show no change during DMT relative to baseline. The shuffle
control is the primary integrity check — any result must beat shuffle
to be taken seriously.

We also report R separately. Based on prior results (sleep EEG), R and δ
are expected to dissociate. Whether they do here is an open question.

Usage
-----
  python run_dmt_pcf.py --data_dir dmt_data
  python run_dmt_pcf.py --data_dir dmt_data --save results_dmt.csv

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
import os
import warnings
from pathlib import Path

import numpy as np
from scipy import signal, stats

warnings.filterwarnings("ignore")

# Verified subject pairs — exact filenames from Zenodo metadata
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


def epoch_metrics(raw_signal, fs, epoch_sec=5.0):
    """Cut signal into epochs and return per-epoch δ and R arrays."""
    n = int(epoch_sec * fs)
    n_epochs = len(raw_signal) // n
    deltas, Rs = [], []
    for i in range(n_epochs):
        ep = raw_signal[i*n:(i+1)*n]
        deltas.append(compute_delta(ep))
        Rs.append(compute_R(ep))
    return np.array(deltas), np.array(Rs)


# ---------------------------------------------------------------------------
# BDF loading
# ---------------------------------------------------------------------------

def load_first_eeg_channel(bdf_path):
    """Load BDF file and return (signal_array, sample_rate)."""
    try:
        import mne
    except ImportError:
        raise ImportError("pip install mne")

    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
    fs  = raw.info["sfreq"]

    # Pick first EEG channel
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(eeg_picks) == 0:
        eeg_picks = [0]   # fallback to first channel

    data = raw.get_data(picks=[eeg_picks[0]])[0]
    ch_name = raw.ch_names[eeg_picks[0]]
    return data, float(fs), ch_name


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run(data_dir, epoch_sec=5.0, n_shuffle=5000, save_path=None):
    try:
        import mne
    except ImportError:
        raise ImportError("mne required: pip install mne")

    data_dir = Path(data_dir)
    results = []

    print(f"\n{'='*60}")
    print("PCF δ and R — DMT vs Eyes-Closed Baseline")
    print("Cavanna et al. (2021) / Zenodo 3992359")
    print(f"{'='*60}")
    print(f"Epoch length: {epoch_sec}s | Shuffle iterations: {n_shuffle}")
    print(f"Data directory: {data_dir}\n")

    skipped = []
    for sid, dmt_f, ec_f in PAIRS:
        dmt_path = data_dir / dmt_f
        ec_path  = data_dir / ec_f

        if not dmt_path.exists() or not ec_path.exists():
            skipped.append(sid)
            continue

        try:
            dmt_sig, fs, ch = load_first_eeg_channel(str(dmt_path))
            ec_sig,  _,  _  = load_first_eeg_channel(str(ec_path))

            d_dmt, r_dmt = epoch_metrics(dmt_sig, fs, epoch_sec)
            d_ec,  r_ec  = epoch_metrics(ec_sig,  fs, epoch_sec)

            row = {
                "subject":       sid,
                "channel":       ch,
                "fs":            fs,
                "n_epochs_dmt":  len(d_dmt),
                "n_epochs_ec":   len(d_ec),
                "delta_dmt":     float(np.nanmean(d_dmt)),
                "delta_ec":      float(np.nanmean(d_ec)),
                "delta_diff":    float(np.nanmean(d_dmt) - np.nanmean(d_ec)),
                "R_dmt":         float(np.nanmean(r_dmt)),
                "R_ec":          float(np.nanmean(r_ec)),
                "R_diff":        float(np.nanmean(r_dmt) - np.nanmean(r_ec)),
            }
            results.append(row)
            print(f"  S{sid:02d}  δ: EC={row['delta_ec']:+.3f} → DMT={row['delta_dmt']:+.3f} "
                  f"(Δ={row['delta_diff']:+.3f})  "
                  f"R: EC={row['R_ec']:.3f} → DMT={row['R_dmt']:.3f}")

        except Exception as e:
            print(f"  S{sid:02d}  ERROR: {e}")
            skipped.append(sid)

    if not results:
        print("\nNo subjects successfully processed.")
        return

    if skipped:
        print(f"\n  Skipped subjects (file not found or error): {skipped}")

    # ----- Group statistics -----
    delta_dmt  = np.array([r["delta_dmt"]  for r in results])
    delta_ec   = np.array([r["delta_ec"]   for r in results])
    r_dmt      = np.array([r["R_dmt"]      for r in results])
    r_ec       = np.array([r["R_ec"]       for r in results])
    delta_diff = delta_dmt - delta_ec
    r_diff     = r_dmt - r_ec

    t_d, p_d = stats.ttest_rel(delta_dmt, delta_ec)
    t_r, p_r = stats.ttest_rel(r_dmt,     r_ec)

    # Shuffle control on δ difference
    rng = np.random.default_rng(seed=0)
    null = []
    for _ in range(n_shuffle):
        signs = rng.choice([-1, 1], size=len(delta_diff))
        null.append(float((signs * delta_diff).mean()))
    p_perm = float(np.mean(np.abs(null) >= abs(delta_diff.mean())))

    n = len(results)
    print(f"\n{'='*60}")
    print(f"RESULTS  (N = {n} subjects)")
    print(f"{'='*60}")
    print(f"\n  δ (generative structure):")
    print(f"    EC  baseline : {delta_ec.mean():+.4f} ± {delta_ec.std():.4f}")
    print(f"    DMT          : {delta_dmt.mean():+.4f} ± {delta_dmt.std():.4f}")
    print(f"    Difference   : {delta_diff.mean():+.4f} ± {delta_diff.std():.4f}")
    print(f"    Paired t     : t = {t_d:+.3f},  p = {p_d:.4g}")
    print(f"    Shuffle p    : {p_perm:.4g}  ({n_shuffle} permutations)")

    print(f"\n  R (phase coherence):")
    print(f"    EC  baseline : {r_ec.mean():.4f} ± {r_ec.std():.4f}")
    print(f"    DMT          : {r_dmt.mean():.4f} ± {r_dmt.std():.4f}")
    print(f"    Difference   : {r_diff.mean():+.4f} ± {r_diff.std():.4f}")
    print(f"    Paired t     : t = {t_r:+.3f},  p = {p_r:.4g}")

    print(f"\n{'='*60}")
    print("HONEST INTERPRETATION")
    print(f"{'='*60}")
    if p_perm < 0.05:
        direction = "HIGHER" if delta_diff.mean() > 0 else "LOWER"
        print(f"  δ is {direction} during DMT vs baseline.")
        print(f"  The shuffle control confirms this exceeds chance (p={p_perm:.4g}).")
        print(f"  This is a real signal. Interpretation requires care:")
        if delta_diff.mean() > 0:
            print(f"  Higher δ during DMT = more temporal regularity.")
            print(f"  Consistent with the known increase in delta-band power under DMT.")
        else:
            print(f"  Lower δ during DMT = less temporal regularity / more entropic.")
            print(f"  Consistent with increased neural complexity/entropy under psychedelics.")
    else:
        print(f"  δ does NOT significantly differ between DMT and baseline (p_perm={p_perm:.4g}).")
        print(f"  This is a null result — report it as such.")

    print(f"\n  NOTE: This is exploratory analysis on a naturalistic dataset.")
    print(f"  No artifact rejection was applied. Results should be replicated")
    print(f"  with proper preprocessing before drawing strong conclusions.")

    # ----- Save -----
    if save_path:
        import csv
        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Per-subject results saved to: {save_path}")

    print(f"\n{'='*60}\n")
    return results


def parse_args():
    p = argparse.ArgumentParser(
        description="PCF δ and R on Cavanna DMT EEG dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",   default="dmt_data",
                   help="Directory containing downloaded .bdf files")
    p.add_argument("--epoch_sec",  type=float, default=5.0,
                   help="Epoch length in seconds")
    p.add_argument("--n_shuffle",  type=int,   default=5000,
                   help="Permutation iterations for shuffle control")
    p.add_argument("--save",       default=None,
                   help="Save per-subject CSV to this path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        data_dir  = args.data_dir,
        epoch_sec = args.epoch_sec,
        n_shuffle = args.n_shuffle,
        save_path = args.save,
    )
