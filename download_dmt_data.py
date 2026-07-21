"""
download_dmt_data.py
====================
Downloads the Cavanna et al. DMT EEG dataset from Zenodo.

Dataset: "Neural and subjective effects of inhaled DMT in natural settings"
Zenodo:  https://zenodo.org/records/3992359
Authors: Cavanna, Tagliazucchi, Pallavicini, Timmermann et al. (2021)
License: Open access

34 usable subject pairs (BDF format):
  S##-DMT.bdf  — EEG during DMT effects (~6 min)
  S##-EC.bdf   — EEG eyes-closed baseline before DMT (~5 min)

Note: S21-EC is .xdf format and is excluded.

Usage
-----
  # Quick test — download 3 subjects (~130 MB):
  python download_dmt_data.py --out_dir dmt_data --n_subjects 3

  # Full dataset — all 34 subjects (~1.4 GB):
  python download_dmt_data.py --out_dir dmt_data

  # Resume interrupted download:
  python download_dmt_data.py --out_dir dmt_data --skip_existing
"""

import argparse
import os
import urllib.request

BASE_URL = "https://zenodo.org/api/records/3992359/files/{filename}/content"

# Verified against Zenodo metadata — exact filenames
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
    # S21 skipped — EC file is .xdf format
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


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        bar = "#" * int(pct / 2)
        print(f"\r    [{bar:<50}] {pct:5.1f}%", end="", flush=True)


def download_file(filename, out_dir, skip_existing):
    dest = os.path.join(out_dir, filename)
    if skip_existing and os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"  ✓ {filename} ({size_mb:.0f} MB)")
        return True
    url = BASE_URL.format(filename=filename)
    print(f"  ↓ {filename}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()
        return True
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def main():
    p = argparse.ArgumentParser(
        description="Download Cavanna DMT EEG dataset (Zenodo 3992359)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out_dir",       default="dmt_data")
    p.add_argument("--n_subjects",    type=int, default=34,
                   help="Subjects to download (1–34). Use 3 for a quick test.")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pairs = PAIRS[:args.n_subjects]
    total_files = len(pairs) * 2

    print(f"\nDownloading {len(pairs)} subjects ({total_files} files)")
    print(f"Destination: {os.path.abspath(args.out_dir)}")
    print(f"Source: https://zenodo.org/records/3992359\n")

    failed = []
    for i, (sid, dmt_f, ec_f) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] Subject {sid:02d}")
        for fname in [dmt_f, ec_f]:
            if not download_file(fname, args.out_dir, args.skip_existing):
                failed.append(fname)

    print(f"\n{'='*55}")
    if failed:
        print(f"Failed: {failed}")
    else:
        print(f"All files downloaded to: {os.path.abspath(args.out_dir)}")
    print(f"\nNext:")
    print(f"  python run_dmt_pcf.py --data_dir {args.out_dir}")

if __name__ == "__main__":
    main()
