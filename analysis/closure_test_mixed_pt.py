#!/usr/bin/env python3
"""
closure_test_mixed_pt.py

pT-DIFFERENTIAL closure test for the MIXED-EVENT SUBTRACTION method.
closure_test_mixed.py answers "is the mixed-event difference calibrated
for a dataset shaped like my WHOLE sample" -- this script answers the
sharper question that mirrors closure_test_pt.py's upgrade over
closure_test.py: "is it calibrated INSIDE EACH leading-jet-pT bin
specifically", since both the trivial-kinematic contribution and the
per-bin statistics change with pT (that's the entire reason the
pT-differential measurement, Sec. "Application to the real validation
sample", exists in the first place).

For each leading-jet-pT bin, and for BOTH scenarios (null s=0, injected
--signal-strength):
  1. Select that bin's events only.
  2. Fit lambda_j(pT) = a_j + b_j ln(pT) to THAT BIN's own (n, pT) pairs
     -- not the whole-sample fit -- exactly as closure_test_pt.py does
     for the raw estimators' Gamma-Poisson model.
  3. Kinematics for both the pseudo-truth reference sample and every
     closure repeat are resampled from THAT BIN's own real (pT1, pT2)
     pairs, so the real momentum-balance correlation local to that bin
     is preserved, not the whole sample's.
  4. Establish a bin-specific pseudo-truth from a large reference sample
     (naive estimator, bias negligible at that size), analyzed with the
     identical k-NN-matched mixed-event difference.
  5. Run N_repeats closure repeats at that bin's ACTUAL N, each with the
     full joint-bootstrap machinery (all four methods, difference,
     joint-bootstrap sigma), and record the NSB-difference pull
     distribution's mean and std.

Output: a summary table and one summary plot -- pull mean (top row) and
pull std (bottom row) vs. pT, one column per scenario (null | injected)
-- directly showing whether the mixed-event difference's bootstrap
uncertainty can be trusted bin-by-bin, which was flagged as the missing
piece before quoting any per-bin beyond-kinematics significance.

Purely diagnostic, same as every other closure test in this project: no
correction is applied anywhere, only reported.

Usage:
    python3 closure_test_mixed_pt.py file.root --level truth --alphabet 80 \
        --pt-bins 4 --signal-strength 0.15 --n-repeats 20 --n-boot 20
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_dijet_entropy import load_dijet_tree, make_pt_bins, naive_MI
from analyze_dijet_mixed import build_mixed_partner_idx
from closure_test_mixed import fit_lambda_vs_pt, generate, diff_with_joint_bootstrap


SCENARIOS = [("null", 0.0), ("signal", None)]  # "signal" s filled in from args at runtime


# ---------------------------------------------------------------------------
# One (bin, scenario) combination: bin-specific fit + truth + closure
# ---------------------------------------------------------------------------

def analyze_one_bin_scenario(lo, hi, na_bin, nb_bin, pt1_bin, pt2_bin, s,
                              alphabet, k_neighbors, n_shuffles, n_boot,
                              n_repeats, ref_n, rng, tag):
    n_bin = len(na_bin)
    print(f"\n{'-'*70}\npT bin [{lo:.1f},{hi:.1f}) GeV, N={n_bin}, scenario={tag}\n{'-'*70}")

    print("  fitting bin-specific lambda(pT) = a + b ln(pT) ...")
    a1, b1 = fit_lambda_vs_pt(na_bin, pt1_bin)
    a2, b2 = fit_lambda_vs_pt(nb_bin, pt2_bin)
    print(f"    jet A: a={a1:.3f} b={b1:.3f}   jet B: a={a2:.3f} b={b2:.3f}")

    # ---- bin-specific pseudo-truth, from this bin's own real kinematics ----
    ref_n_bin = min(ref_n, max(50_000, n_bin * 20))  # don't over-promise precision a tiny bin can't support
    sel = rng.integers(0, n_bin, size=ref_n_bin)
    p1r, p2r = pt1_bin[sel], pt2_bin[sel]
    naR, nbR = generate(p1r, p2r, a1, b1, a2, b2, s, rng)
    I_same_ref, _ = naive_MI(naR, nbR)
    idxR, _ = build_mixed_partner_idx(p2r, k_neighbors, rng)
    I_mix_ref, _ = naive_MI(naR, nbR[idxR])
    diff_true = I_same_ref - I_mix_ref
    print(f"    pseudo-truth (ref N={ref_n_bin:,}): diff_true = {diff_true:+.5f} nats")

    # ---- closure loop at this bin's actual N ----
    diffs = np.empty(n_repeats)
    errs = np.empty(n_repeats)
    for rep in range(n_repeats):
        rsel = rng.integers(0, n_bin, size=n_bin)
        p1, p2 = pt1_bin[rsel], pt2_bin[rsel]
        na, nb = generate(p1, p2, a1, b1, a2, b2, s, rng)
        d, e = diff_with_joint_bootstrap(na, nb, p2, alphabet, k_neighbors,
                                          n_shuffles, n_boot, rng)
        diffs[rep] = d['nsb']
        errs[rep] = e['nsb']

    pulls = (diffs - diff_true) / errs
    out = dict(
        pt_lo=lo, pt_hi=hi, pt_mid=float(pt1_bin.mean()), N=n_bin, scenario=tag,
        diff_true=diff_true, pull_mean=float(pulls.mean()), pull_std=float(pulls.std(ddof=1)),
    )
    print(f"    NSB-diff pull: mean={out['pull_mean']:+.3f}  std={out['pull_std']:.3f}")
    if abs(out['pull_std'] - 1.0) > 0.3:
        d = "UNDER-covering" if out['pull_std'] > 1.3 else "OVER-covering"
        print(f"    NOTE: pull std deviates from 1 by >0.3 -> {d} in this bin/scenario.")
    return out


def print_summary_table(bin_results):
    print(f"\n{'='*92}\npT-DIFFERENTIAL CLOSURE TEST SUMMARY -- mixed-event difference (NSB)\n{'='*92}")
    print(f"{'pT range':>16} | {'N':>7} | {'scenario':>8} | {'diff_true':>10} | "
          f"{'pull mean':>10} | {'pull std':>9}")
    print("-" * 92)
    for r in bin_results:
        print(f"[{r['pt_lo']:6.1f},{r['pt_hi']:6.1f}) | {r['N']:>7} | {r['scenario']:>8} | "
              f"{r['diff_true']:>+10.5f} | {r['pull_mean']:>+10.3f} | {r['pull_std']:>9.3f}")


def plot_summary(bin_results, out_prefix):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    cols = [("null", "NULL (s=0)", "#9467bd"), ("signal", "INJECTED", "#d62728")]

    for col, (key, label, color) in enumerate(cols):
        rows = [r for r in bin_results if r['scenario'] == key]
        if not rows:
            continue
        pt = [r['pt_mid'] for r in rows]
        means = [r['pull_mean'] for r in rows]
        stds = [r['pull_std'] for r in rows]

        ax = axes[0, col]
        ax.axhline(0, color='gray', lw=1)
        ax.plot(pt, means, 'o-', color=color)
        ax.set_xlabel(r'leading jet $p_T$ (GeV)')
        ax.set_ylabel('NSB-diff pull mean')
        ax.set_title(f'{label}: pull mean vs. $p_T$\n(should be ~0)')

        ax = axes[1, col]
        ax.axhline(1, color='red', ls='--', lw=1.5, label='well-calibrated (std=1)')
        ax.plot(pt, stds, 'o-', color=color)
        ax.set_xlabel(r'leading jet $p_T$ (GeV)')
        ax.set_ylabel('NSB-diff pull std')
        ax.set_title(f'{label}: pull std vs. $p_T$\n(>1: under-cov., <1: over-cov.)')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_closure_mixed_pt_summary.png", dpi=150)
    print(f"\n  saved {out_prefix}_closure_mixed_pt_summary.png")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root_file")
    p.add_argument("--level", choices=["truth", "det"], default="truth")
    p.add_argument("--alphabet", type=int, default=80)
    p.add_argument("--mix-neighbors", type=int, default=10)
    p.add_argument("--signal-strength", type=float, default=0.15,
                    help="s of the injected shared latent Gamma(1/s^2, s^2) for "
                         "the signal scenario (default 0.15). The null scenario "
                         "(s=0) always runs as well, in every bin.")
    p.add_argument("--pt-bins", type=int, default=4)
    p.add_argument("--pt-bin-mode", choices=["quantile", "fixed"], default="quantile")
    p.add_argument("--min-events", type=int, default=300,
                    help="skip bins with fewer events than this -- need enough "
                         "statistics to fit the per-bin lambda(pT) model at all "
                         "(default 300)")
    p.add_argument("--n-repeats", type=int, default=20,
                    help="independent draws per (bin, scenario) (default 20 -- "
                         "lower than the whole-sample script's 30 since cost "
                         "multiplies by n_bins x 2 scenarios)")
    p.add_argument("--n-boot", type=int, default=20,
                    help="joint bootstrap resamples per repeat (default 20)")
    p.add_argument("--n-shuffles", type=int, default=15)
    p.add_argument("--ref-n", type=int, default=500_000,
                    help="reference sample size for pseudo-truth, PER BIN, capped "
                         "further for small bins (default 500,000 -- smaller than "
                         "the whole-sample script's 1,000,000 since this cost "
                         "multiplies by n_bins x 2 scenarios)")
    p.add_argument("--out-prefix", default=None)
    p.add_argument("--figs-dir", default="figs")
    args = p.parse_args()

    if args.out_prefix is None:
        args.out_prefix = f"closure_mixed_pt_{args.level}"
    os.makedirs(args.figs_dir, exist_ok=True)
    args.out_prefix = os.path.join(args.figs_dir, args.out_prefix)
    print(f"Figures will be saved under: {args.figs_dir}")

    rng = np.random.default_rng(20260728)
    data = load_dijet_tree(args.root_file, level=args.level)
    n_a = data["jet1_mult"].astype(int)
    n_b = data["jet2_mult"].astype(int)
    pt1 = data["jet1_pt"].astype(float)
    pt2 = data["jet2_pt"].astype(float)
    print(f"Loaded {len(n_a)} events.")

    edges = make_pt_bins(pt1, args.pt_bins, args.pt_bin_mode)
    print(f"pT bin edges ({args.pt_bin_mode}): {np.round(edges, 1)}")

    scenarios = [("null", 0.0), ("signal", args.signal_strength)]

    bin_results = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (pt1 >= lo) & (pt1 < hi) if i < len(edges) - 2 else \
               (pt1 >= lo) & (pt1 <= hi)
        n_bin = int(mask.sum())
        if n_bin < args.min_events:
            print(f"\nSkipping bin [{lo:.1f},{hi:.1f}): only {n_bin} events "
                  f"(< --min-events={args.min_events})")
            continue

        for tag, s in scenarios:
            result = analyze_one_bin_scenario(
                lo, hi, n_a[mask], n_b[mask], pt1[mask], pt2[mask], s,
                args.alphabet, args.mix_neighbors, args.n_shuffles, args.n_boot,
                args.n_repeats, args.ref_n, rng, tag)
            bin_results.append(result)

    if len(bin_results) == 0:
        print("\nNo bins had enough events -- nothing to summarize.")
        return

    print_summary_table(bin_results)
    plot_summary(bin_results, args.out_prefix)
    print("\nDone.")


if __name__ == "__main__":
    main()
