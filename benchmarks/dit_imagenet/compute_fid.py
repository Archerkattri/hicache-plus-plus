#!/usr/bin/env python3
"""Compute FID from the per-cell Inception stats written by bench_dit.py.

Primary metric: **FID vs the uncached baseline** — the cache-induced distribution drift, which is
exactly what a lossless feature cache must keep near zero (needs no external dataset). If a
reference-stats file is present (built once from the ImageNet-256 reference batch via --build-ref),
also reports the **absolute FID vs ImageNet**. Prints a markdown table with speedup + compute count.
"""
import argparse
import glob
import os

import numpy as np
from scipy import linalg


def frechet(mu1, s1, mu2, s2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1 + s2 - 2 * covmean))


def order_key(cell):
    if cell == "baseline":
        return (0, 0)
    m, _, i = cell.rpartition("_i")
    rank = {"taylor": 1, "hermite": 2, "dmd": 3}.get(m, 9)
    return (rank, int(i))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells_dir")
    ap.add_argument("--ref", default=None, help="optional imagenet ref-stats .npz (mu,sigma)")
    args = ap.parse_args()

    cells = {}
    for f in glob.glob(os.path.join(args.cells_dir, "*.npz")):
        name = os.path.basename(f)[:-4]
        if name == "_ref_stats":
            continue
        d = np.load(f)
        cells[name] = dict(mu=d["mu"], sigma=d["sigma"], latency=float(d["latency_ms"]),
                           compute=int(d["compute_calls"]), steps=int(d["steps"]))
    if "baseline" not in cells:
        print("no baseline cell yet"); return
    base = cells["baseline"]
    ref = None
    if args.ref and os.path.exists(args.ref):
        r = np.load(args.ref); ref = (r["mu"], r["sigma"])

    hdr = "| cell | compute / steps | ms/img | speedup | FID↓ vs baseline |"
    sep = "|---|---:|---:|---:|---:|"
    if ref:
        hdr += " FID↓ vs ImageNet |"; sep += "---:|"
    print(hdr); print(sep)
    for name in sorted(cells, key=order_key):
        c = cells[name]
        fid_b = 0.0 if name == "baseline" else frechet(c["mu"], c["sigma"], base["mu"], base["sigma"])
        row = (f"| {name} | {c['compute']}/{c['steps']} | {c['latency']:.0f} | "
               f"{base['latency']/c['latency']:.2f}× | {fid_b:.2f} |")
        if ref:
            row += f" {frechet(c['mu'], c['sigma'], ref[0], ref[1]):.2f} |"
        print(row)


if __name__ == "__main__":
    main()
