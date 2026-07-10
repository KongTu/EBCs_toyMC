#!/usr/bin/env python3
"""
closure_test_pt.py

pT-DIFFERENTIAL Tier-2 closure test. closure_test.py answers "is NSB
calibrated for a dataset shaped like my WHOLE sample" -- this script
answers the sharper, more useful question: "is NSB calibrated INSIDE
EACH leading-jet-pT bin specifically", since both the underlying
multiplicity/correlation structure AND the per-bin statistics change
with pT (that's the whole point of the differential measurement).

For each leading-jet-pT bin:
  1. Select that bin's events only.
  2. Fit a Gamma-Poisson model to THAT BIN's own marginals/correlation
     (not the whole-sample fit) -- this is the key upgrade over running
     closure_test.py once with --closure-n set to a bin's N, which
     reuses the whole-sample model everywhere and would miss any real
     pT-dependence in the correlation structure itself.
  3. Generate a reference sample from that bin-specific fitted model for
     a bin-specific pseudo-truth (I_true, SBA_true, NMI_true).
  4. Run the closure test (naive/Miller-Madow/shuffle/NSB) at that bin's
     ACTUAL N, and record NSB's pull-distribution mean/std for I, S(B|A),
     and NMI.

Output: a summary table and three summary plots (pull mean+std vs. pT,
for I, S(B|A), NMI) -- directly showing whether NSB's reported
uncertainty can be trusted bin-by-bin, which is the number you actually
need before quoting a per-bin significance in the real analysis.

Purely diagnostic, same as closure_test.py: no correction is applied
anywhere, only reported.

Usage:
    python3 closure_test_pt.py pythia_dijet_pp200.root --level truth \\
        --alphabet 90 --pt-bins 4 --n-repeats 100 --n-boot 150 --ref-n 1000000
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from analyze_dijet_entropy import load_dijet_tree, make_pt_bins
from closure_test import fit_model, compute_pseudo_truth, run_closure_test


def analyze_one_bin(lo, hi, na_bin, nb_bin, pt_mid, alphabet, n_repeats,
                     n_boot, n_shuffles, ref_n, rng):
    n_bin = len(na_bin)
    print(f"\n{'='*70}\npT bin [{lo:.1f}, {hi:.1f}) GeV, N={n_bin}\n{'='*70}")

    print("--- fitting bin-specific Gamma-Poisson model ---")
    fit = fit_model(na_bin, nb_bin)

    print("--- bin-specific pseudo-truth ---")
    I_true, SBA_true, NMI_true, _, _ = compute_pseudo_truth(
        fit['k'], fit['theta'], fit['c'], ref_n, rng)

    print(f"--- closure test: {n_repeats} repeats at this bin's N={n_bin} ---")
    results = run_closure_test(
        fit['k'], fit['theta'], fit['c'], n_bin, alphabet, alphabet,
        I_true, SBA_true, n_repeats, n_boot, n_shuffles, rng)

    I_pulls = (results['I_nsb'] - I_true) / results['I_nsb_err']
    SBA_pulls = (results['SBA_nsb'] - SBA_true) / results['SBA_nsb_err']
    NMI_pulls = (results['NMI_nsb'] - NMI_true) / results['NMI_nsb_err']

    out = dict(
        pt_lo=lo, pt_hi=hi, pt_mid=pt_mid, N=n_bin,
        k=fit['k'], theta=fit['theta'], c=fit['c'],
        I_true=I_true, SBA_true=SBA_true, NMI_true=NMI_true,
        I_pull_mean=I_pulls.mean(), I_pull_std=I_pulls.std(ddof=1),
        SBA_pull_mean=SBA_pulls.mean(), SBA_pull_std=SBA_pulls.std(ddof=1),
        NMI_pull_mean=NMI_pulls.mean(), NMI_pull_std=NMI_pulls.std(ddof=1),
    )
    print(f"  I pull:   mean={out['I_pull_mean']:+.3f}  std={out['I_pull_std']:.3f}")
    print(f"  S(B|A) pull: mean={out['SBA_pull_mean']:+.3f}  std={out['SBA_pull_std']:.3f}")
    print(f"  NMI pull: mean={out['NMI_pull_mean']:+.3f}  std={out['NMI_pull_std']:.3f}")
    for label, std in [('I', out['I_pull_std']), ('S(B|A)', out['SBA_pull_std']),
                        ('NMI', out['NMI_pull_std'])]:
        if abs(std - 1.0) > 0.3:
            direction = "UNDER-covering" if std > 1.3 else "OVER-covering"
            print(f"  NOTE: {label} pull std={std:.2f} -> NSB {direction} in this bin.")

    return out


def print_summary_table(bin_results):
    print(f"\n{'='*100}\npT-DIFFERENTIAL CLOSURE TEST SUMMARY\n{'='*100}")
    print(f"{'pT range':>16} | {'N':>7} | {'I pull (m/s)':>14} | "
          f"{'S(B|A) pull (m/s)':>18} | {'NMI pull (m/s)':>16}")
    print("-" * 100)
    for r in bin_results:
        print(f"[{r['pt_lo']:6.1f},{r['pt_hi']:6.1f}) | {r['N']:>7} | "
              f"{r['I_pull_mean']:+5.2f} / {r['I_pull_std']:4.2f}    | "
              f"{r['SBA_pull_mean']:+5.2f} / {r['SBA_pull_std']:4.2f}       | "
              f"{r['NMI_pull_mean']:+5.2f} / {r['NMI_pull_std']:4.2f}")


def plot_summary(bin_results, out_prefix):
    pt = [r['pt_mid'] for r in bin_results]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for col, (key_mean, key_std, label, color) in enumerate([
        ('I_pull_mean', 'I_pull_std', '$I(A:B)$', '#9467bd'),
        ('SBA_pull_mean', 'SBA_pull_std', '$S(B|A)$', 'purple'),
        ('NMI_pull_mean', 'NMI_pull_std', 'NMI', '#2ca02c'),
    ]):
        means = [r[key_mean] for r in bin_results]
        stds = [r[key_std] for r in bin_results]

        ax = axes[0, col]
        ax.axhline(0, color='gray', lw=1)
        ax.plot(pt, means, 'o-', color=color)
        ax.set_xlabel(r'leading jet $p_T$ (GeV)')
        ax.set_ylabel('NSB pull mean')
        ax.set_title(f'{label}: pull mean vs. $p_T$\n(should be ~0)')

        ax = axes[1, col]
        ax.axhline(1, color='red', ls='--', lw=1.5, label='well-calibrated (std=1)')
        ax.plot(pt, stds, 'o-', color=color)
        ax.set_xlabel(r'leading jet $p_T$ (GeV)')
        ax.set_ylabel('NSB pull std')
        ax.set_title(f'{label}: pull std vs. $p_T$\n(>1: under-covering, <1: over-covering)')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_closure_test_pt_summary.png", dpi=150)
    print(f"\n  saved {out_prefix}_closure_test_pt_summary.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root_file", help="PYTHIA8 dijet ROOT file")
    parser.add_argument("--level", choices=["truth", "det"], default="truth")
    parser.add_argument("--alphabet", type=int, default=80,
                         help="same alphabet you use in the real analysis")
    parser.add_argument("--pt-bins", type=int, default=4)
    parser.add_argument("--pt-bin-mode", choices=["quantile", "fixed"], default="quantile")
    parser.add_argument("--min-events", type=int, default=200,
                         help="skip bins with fewer events than this -- a closure "
                              "test needs reasonable statistics just to FIT the "
                              "per-bin model in the first place (default 200)")
    parser.add_argument("--n-repeats", type=int, default=100,
                         help="independent draws per bin (default 100; this is "
                              "what determines how trustworthy the reported pull "
                              "std itself is -- see chat notes)")
    parser.add_argument("--n-boot", type=int, default=150,
                         help="bootstrap resamples per repeat for NSB uncertainty "
                              "(default 150)")
    parser.add_argument("--n-shuffles", type=int, default=100)
    parser.add_argument("--ref-n", type=int, default=1_000_000,
                         help="reference sample size for pseudo-truth, PER BIN "
                              "(default 1,000,000 -- smaller than closure_test.py's "
                              "default since this cost multiplies by n_bins)")
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--figs-dir", default="figs")
    args = parser.parse_args()

    if args.out_prefix is None:
        args.out_prefix = f"closure_test_pt_{args.level}"
    os.makedirs(args.figs_dir, exist_ok=True)
    args.out_prefix = os.path.join(args.figs_dir, args.out_prefix)

    rng = np.random.default_rng(20260710)

    print(f"Loading {args.root_file} (level='{args.level}') ...")
    data = load_dijet_tree(args.root_file, level=args.level)
    n_a = data["jet1_mult"].astype(int)
    n_b = data["jet2_mult"].astype(int)
    jet_pt = data["jet1_pt"].astype(float)
    print(f"Loaded {len(n_a)} events.")

    edges = make_pt_bins(jet_pt, args.pt_bins, args.pt_bin_mode)
    print(f"pT bin edges ({args.pt_bin_mode}): {np.round(edges, 1)}")

    bin_results = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i+1]
        mask = (jet_pt >= lo) & (jet_pt < hi) if i < len(edges)-2 else \
               (jet_pt >= lo) & (jet_pt <= hi)
        n_bin = mask.sum()
        if n_bin < args.min_events:
            print(f"\nSkipping bin [{lo:.1f},{hi:.1f}): only {n_bin} events "
                  f"(< --min-events={args.min_events})")
            continue

        result = analyze_one_bin(
            lo, hi, n_a[mask], n_b[mask], jet_pt[mask].mean(),
            args.alphabet, args.n_repeats, args.n_boot, args.n_shuffles,
            args.ref_n, rng)
        bin_results.append(result)

    if len(bin_results) == 0:
        print("\nNo bins had enough events -- nothing to summarize.")
        return

    print_summary_table(bin_results)
    plot_summary(bin_results, args.out_prefix)
    print("\nDone.")


if __name__ == "__main__":
    main()
