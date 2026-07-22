#!/usr/bin/env python3
"""
closure_test_mixed.py

Closure test for the MIXED-EVENT SUBTRACTION method
(analyze_dijet_mixed.py): does I_same - I_mix recover the true
beyond-kinematics MI, with honest (well-calibrated) joint-bootstrap
errors, at the real dataset's statistics?

Model (calibrated to the real data, in the Tier-2 spirit of
closure_test.py, but now with the pT structure that the mixed-event
method exists to handle):
  - Kinematics (pT1, pT2) pairs are RESAMPLED directly from the real
    data -- no spectrum model needed, the real kinematic correlation is
    reproduced exactly.
  - Per-jet multiplicity scale: lambda_j(pT) = a_j + b_j ln(pT), with
    (a_j, b_j) fitted to the real data's multiplicity-vs-pT profile.
  - Genuine (beyond-kinematics) correlation knob: a SHARED latent
    g ~ Gamma(1/s^2, s^2) (mean 1, variance s^2) multiplying both
    lambdas. s = 0 -> conditional independence given pT (null: the true
    beyond-kinematics MI is exactly zero). s > 0 -> a genuine injected
    signal of known size.
  - N_A ~ Poisson(lambda_1 * g), N_B ~ Poisson(lambda_2 * g).

Pseudo-truth: a huge reference sample from the same model, analyzed
with the same k-NN-matched mixed-event difference (naive estimator --
bias negligible at reference size). This defines the truth of exactly
the estimand the method targets, including the matching definition.

Closure: nRepeats fresh draws at the real N; each analyzed with the
full machinery (all four methods per term, difference, JOINT bootstrap
error). Pull = (diff_est - diff_true)/sigma_joint per repeat; a
well-calibrated method gives pull mean ~0 and std ~1.

Runs TWO scenarios by default: the NULL (s=0; false-positive control --
the single most important check before claiming any beyond-kinematics
signal on real data) and an INJECTED signal (--signal-strength).

Purely diagnostic -- no corrections are applied anywhere.

Usage:
    python3 closure_test_mixed.py file.root --level truth --alphabet 80 \
        --n-repeats 30 --n-boot 30 --signal-strength 0.15
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from analyze_dijet_entropy import load_dijet_tree, naive_MI
from analyze_dijet_mixed import (
    build_mixed_partner_idx, all_point_estimates, METHODS, LABELS, COLORS,
)


# ---------------------------------------------------------------------------
# Calibrated generative model
# ---------------------------------------------------------------------------

def fit_lambda_vs_pt(n, pt):
    """Fit <N> = a + b ln(pT) (the MLLA-motivated form used throughout
    this project) by least squares on the raw event list."""
    b, a = np.polyfit(np.log(pt), n, 1)
    return a, b


def generate(pt1, pt2, a1, b1, a2, b2, s, rng):
    lam1 = np.maximum(a1 + b1 * np.log(pt1), 0.1)
    lam2 = np.maximum(a2 + b2 * np.log(pt2), 0.1)
    if s > 0:
        g = rng.gamma(shape=1.0 / s**2, scale=s**2, size=len(pt1))
        lam1 = lam1 * g
        lam2 = lam2 * g
    return rng.poisson(lam1), rng.poisson(lam2)


# ---------------------------------------------------------------------------
# One repeat: full mixed-event difference with joint bootstrap (quiet)
# ---------------------------------------------------------------------------

def diff_with_joint_bootstrap(n_a, n_b, pt2, alphabet, k_neighbors,
                                n_shuffles, n_boot, rng):
    idx, _ = build_mixed_partner_idx(pt2, k_neighbors, rng)
    same = all_point_estimates(n_a, n_b, alphabet, n_shuffles, rng)
    mix = all_point_estimates(n_a, n_b[idx], alphabet, n_shuffles, rng)
    diff = {m: same[m] - mix[m] for m in METHODS}
    N = len(n_a)
    boot = {m: np.empty(n_boot) for m in METHODS}
    for b in range(n_boot):
        sel = rng.integers(0, N, size=N)
        ra, rb, rp = n_a[sel], n_b[sel], pt2[sel]
        ridx, _ = build_mixed_partner_idx(rp, k_neighbors, rng)
        s1 = all_point_estimates(ra, rb, alphabet, n_shuffles, rng)
        s2 = all_point_estimates(ra, rb[ridx], alphabet, n_shuffles, rng)
        for m in METHODS:
            boot[m][b] = s1[m] - s2[m]
    return diff, {m: boot[m].std(ddof=1) for m in METHODS}


# ---------------------------------------------------------------------------
# One scenario (fixed signal strength s): truth + closure + report + plot
# ---------------------------------------------------------------------------

def run_scenario(s, pt1_all, pt2_all, a1, b1, a2, b2, N_closure, alphabet,
                  k_neighbors, n_shuffles, n_boot, n_repeats, ref_n, rng,
                  out_prefix):
    tag = "NULL (s=0, true signal = 0)" if s == 0 else f"INJECTED (s={s})"
    print(f"\n{'='*72}\nSCENARIO: {tag}\n{'='*72}")

    # ---- pseudo-truth from a huge reference sample ----
    print(f"  Pseudo-truth: reference sample N={ref_n:,} (naive estimator, "
          f"bias negligible at this size; same k-NN matching as the method)...")
    sel = rng.integers(0, len(pt1_all), size=ref_n)
    p1r, p2r = pt1_all[sel], pt2_all[sel]
    naR, nbR = generate(p1r, p2r, a1, b1, a2, b2, s, rng)
    I_same_ref, _ = naive_MI(naR, nbR)
    idxR, _ = build_mixed_partner_idx(p2r, k_neighbors, rng)
    I_mix_ref, _ = naive_MI(naR, nbR[idxR])
    diff_true = I_same_ref - I_mix_ref
    print(f"  I_same(ref)={I_same_ref:.5f}  I_mix(ref)={I_mix_ref:.5f}  "
          f"->  diff_true = {diff_true:+.5f} nats")

    # ---- closure loop ----
    print(f"  Closure: {n_repeats} repeats at N={N_closure:,} "
          f"(joint bootstrap x{n_boot} per repeat)...")
    diffs = {m: np.empty(n_repeats) for m in METHODS}
    errs = {m: np.empty(n_repeats) for m in METHODS}
    for rep in range(n_repeats):
        sel = rng.integers(0, len(pt1_all), size=N_closure)
        p1, p2 = pt1_all[sel], pt2_all[sel]
        na, nb = generate(p1, p2, a1, b1, a2, b2, s, rng)
        d, e = diff_with_joint_bootstrap(na, nb, p2, alphabet, k_neighbors,
                                          n_shuffles, n_boot, rng)
        for m in METHODS:
            diffs[m][rep] = d[m]
            errs[m][rep] = e[m]
        if (rep + 1) % max(1, n_repeats // 5) == 0:
            print(f"    ... {rep+1}/{n_repeats} repeats done")

    # ---- report ----
    print(f"\n  diff_true = {diff_true:+.5f} nats")
    print(f"  {'method':>14} | {'mean diff':>11} | {'std/reps':>9} | "
          f"{'bias':>9} | {'pull mean':>9} | {'pull std':>8}")
    print("  " + "-" * 74)
    pulls_by_m = {}
    for m in METHODS:
        pulls = (diffs[m] - diff_true) / errs[m]
        pulls_by_m[m] = pulls
        bias = diffs[m].mean() - diff_true
        print(f"  {LABELS[m]:>14} | {diffs[m].mean():>+11.5f} | "
              f"{diffs[m].std(ddof=1):>9.5f} | {bias:>+9.5f} | "
              f"{pulls.mean():>+9.3f} | {pulls.std(ddof=1):>8.3f}")
    p_nsb = pulls_by_m['nsb']
    f1 = np.mean(np.abs(p_nsb) < 1); f2 = np.mean(np.abs(p_nsb) < 2)
    print(f"\n  NSB-difference calibration: {f1:.1%} within +/-1 sigma "
          f"(expect ~68%), {f2:.1%} within +/-2 sigma (expect ~95%)")
    if abs(p_nsb.std(ddof=1) - 1.0) > 0.3:
        d = "UNDER-covering (errors too small)" if p_nsb.std(ddof=1) > 1.3 \
            else "OVER-covering (too conservative)"
        print(f"  NOTE: NSB-diff pull std deviates from 1 by >0.3 -> {d}.")

    # ---- plot: recovery / NSB pull distribution / per-repeat ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    ax.bar([LABELS[m] for m in METHODS], [diffs[m].mean() for m in METHODS],
           yerr=[diffs[m].std(ddof=1) for m in METHODS], capsize=5,
           color=[COLORS[m] for m in METHODS])
    ax.axhline(diff_true, color='black', ls='--', lw=2,
               label=f'truth = {diff_true:+.4f}')
    ax.axhline(0, color='gray', lw=1)
    ax.set_ylabel(r'recovered $I_{\rm same}-I_{\rm mix}$ (nats)')
    ax.set_title(f'Mixed-subtraction closure, {tag}\n'
                  f'(bars: mean over {n_repeats} repeats; err: spread)')
    ax.tick_params(axis='x', rotation=15)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(p_nsb, bins=max(8, n_repeats // 4), density=True,
            color=COLORS['nsb'], alpha=0.6, edgecolor='black',
            label='NSB-diff pulls')
    x = np.linspace(-4, 4, 200)
    ax.plot(x, stats.norm.pdf(x), 'k--', lw=2, label='standard normal')
    ax.set_xlabel(r'$(\hat D - D_{\rm true})/\sigma_{\rm joint}$')
    ax.set_ylabel('density')
    ax.set_title('NSB difference: pull distribution')
    ax.legend(fontsize=9)

    ax = axes[2]
    reps = np.arange(n_repeats)
    ax.errorbar(reps, diffs['nsb'], yerr=errs['nsb'], fmt='o',
                color=COLORS['nsb'], capsize=2, markersize=4, alpha=0.7)
    ax.axhline(diff_true, color='black', ls='--', lw=2, label='truth')
    ax.axhline(0, color='gray', lw=1)
    ax.set_xlabel('repeat #')
    ax.set_ylabel(r'NSB $I_{\rm same}-I_{\rm mix}$ $\pm$ joint-boot $\sigma$')
    ax.set_title('NSB difference per repeat vs. truth')
    ax.legend(fontsize=9)

    plt.tight_layout()
    suffix = "null" if s == 0 else f"s{s:g}".replace('.', 'p')
    plt.savefig(f"{out_prefix}_closure_mixed_{suffix}.png", dpi=150)
    print(f"  saved {out_prefix}_closure_mixed_{suffix}.png")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root_file")
    p.add_argument("--level", choices=["truth", "det"], default="truth")
    p.add_argument("--alphabet", type=int, default=80)
    p.add_argument("--mix-neighbors", type=int, default=10)
    p.add_argument("--signal-strength", type=float, default=0.15,
                    help="s of the injected shared latent Gamma(1/s^2, s^2) "
                         "for the signal scenario (default 0.15). The null "
                         "scenario (s=0) always runs as well.")
    p.add_argument("--n-repeats", type=int, default=30)
    p.add_argument("--n-boot", type=int, default=30,
                    help="joint bootstrap resamples per repeat (default 30)")
    p.add_argument("--n-shuffles", type=int, default=15)
    p.add_argument("--ref-n", type=int, default=1_000_000)
    p.add_argument("--closure-n", type=int, default=None,
                    help="events per closure repeat (default: real N)")
    p.add_argument("--figs-dir", default="figs")
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args()

    if args.out_prefix is None:
        args.out_prefix = f"closure_mixed_{args.level}"
    os.makedirs(args.figs_dir, exist_ok=True)
    args.out_prefix = os.path.join(args.figs_dir, args.out_prefix)
    print(f"Figures will be saved under: {args.figs_dir}")

    rng = np.random.default_rng(20260723)
    data = load_dijet_tree(args.root_file, level=args.level)
    n_a = data["jet1_mult"].astype(int)
    n_b = data["jet2_mult"].astype(int)
    pt1 = data["jet1_pt"].astype(float)
    pt2 = data["jet2_pt"].astype(float)
    N_real = len(n_a)
    N_closure = args.closure_n if args.closure_n else N_real
    print(f"Loaded {N_real} events; closure repeats at N={N_closure}.")

    print("\n--- Calibrating lambda(pT) = a + b ln(pT) to the real data ---")
    a1, b1 = fit_lambda_vs_pt(n_a, pt1)
    a2, b2 = fit_lambda_vs_pt(n_b, pt2)
    print(f"  jet A: a={a1:.3f}  b={b1:.3f}   jet B: a={a2:.3f}  b={b2:.3f}")
    print("  (kinematics resampled directly from the real (pT1,pT2) pairs)")

    for s in (0.0, args.signal_strength):
        run_scenario(s, pt1, pt2, a1, b1, a2, b2, N_closure, args.alphabet,
                      args.mix_neighbors, args.n_shuffles, args.n_boot,
                      args.n_repeats, args.ref_n, rng, args.out_prefix)

    print("\nDone.")


if __name__ == "__main__":
    main()
