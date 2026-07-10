#!/usr/bin/env python3
"""
closure_test.py

Tier-2 closure test for the entropy/MI estimator pipeline: since real
PYTHIA8 output has no known "true" MI to check against (unlike the
original toy-model study, where we built the generative model ourselves),
this script constructs a CALIBRATED pseudo-truth instead:

  1. Fit the same Gamma-Poisson shared-latent model used throughout the
     toy study to the REAL PYTHIA8 marginals and correlation strength
     (mean_A, var_A, mean_B, var_B, corr(A,B)) -- via least-squares over
     all 5 targets, not just moment-matching one marginal.
  2. Generate a huge sample (default 5,000,000 events) from THAT fitted
     model. At that scale, plug-in bias is negligible (~1/N), giving a
     numerically-exact reference I_true and S(B|A)_true FOR A SCENARIO
     THAT MATCHES YOUR REAL DATA'S SHAPE AND CORRELATION STRENGTH -- not
     an arbitrary toy setup.
  3. Draw many independent synthetic samples at your ACTUAL sample size N
     (and using the SAME alphabet as your real analysis) from the fitted
     model, run the full naive/Miller-Madow/shuffle/NSB pipeline on each,
     and check:
       - bias: does the mean recovered estimate match I_true?
       - calibration: does the PULL distribution (estimate - truth)/sigma
         look standard-normal? (The gold-standard closure-test diagnostic:
         if NSB's reported uncertainty is honest, ~68% of pulls should
         fall within +/-1, ~95% within +/-2.)

This tells you whether the pipeline can be trusted on data that looks
like yours, at your actual statistics -- the closest thing to "ground
truth validation" available without an actual known-truth dataset.

Usage:
    python3 closure_test.py pythia_dijet_pp200.root --level truth --alphabet 90
    python3 closure_test.py pythia_dijet_pp200.root --level truth --alphabet 90 \
        --n-repeats 40 --n-boot 60 --ref-n 5000000
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats

# Reuse the validated estimator library -- single source of truth, no
# reimplementation/duplication of the entropy/MI machinery.
from analyze_dijet_entropy import (
    load_dijet_tree, naive_MI, miller_madow_MI, shuffle_correction_MI,
    nsb_all, entropies_from_samples,
)


# =============================================================================
# Step 1: fit the Gamma-Poisson shared-latent model to the real data
# =============================================================================

def model_moments(k, theta, c):
    """Analytic moments of the shared-latent Gamma-Poisson model:
    Lambda ~ Gamma(k, theta), N_A|Lambda ~ Poisson(Lambda),
    N_B|Lambda ~ Poisson(c*Lambda). Closed-form, no simulation needed."""
    mean_A = k * theta
    var_A = k * theta * (1 + theta)
    mean_B = c * k * theta
    var_B = c * k * theta * (1 + c * theta)
    cov_AB = c * k * theta**2
    corr_AB = cov_AB / np.sqrt(var_A * var_B)
    return mean_A, var_A, mean_B, var_B, corr_AB


def fit_model(n_a, n_b):
    """Least-squares fit of (k, theta, c) to match the real data's
    (mean_A, var_A, mean_B, var_B, corr_AB) as closely as a 3-parameter
    model can. Reports fit quality explicitly -- an honest closure test
    needs to know how good this calibration actually is."""
    n_a = np.asarray(n_a, dtype=float)
    n_b = np.asarray(n_b, dtype=float)

    mean_A_emp = n_a.mean()
    var_A_emp = n_a.var(ddof=1)
    mean_B_emp = n_b.mean()
    var_B_emp = n_b.var(ddof=1)
    corr_emp = np.corrcoef(n_a, n_b)[0, 1]

    targets = np.array([mean_A_emp, var_A_emp, mean_B_emp, var_B_emp, corr_emp])

    def loss(params):
        k, theta, c = np.exp(params)  # optimize in log-space: keeps k,theta,c > 0
        pred = np.array(model_moments(k, theta, c))
        # relative error on the moments, absolute error on correlation
        # (correlation is already O(1), a relative error there would
        # blow up if corr_emp is near zero)
        rel = (pred[:4] - targets[:4]) / targets[:4]
        corr_err = pred[4] - targets[4]
        return np.sum(rel**2) + corr_err**2

    # initial guess: k,theta from a simple NBD moment fit to N_A alone;
    # c from the ratio of means
    if var_A_emp > mean_A_emp:
        p0 = mean_A_emp / var_A_emp
        k0 = mean_A_emp**2 / (var_A_emp - mean_A_emp)
    else:
        k0, p0 = 3.0, 0.3
    theta0 = mean_A_emp / k0 if k0 > 0 else 5.0
    c0 = mean_B_emp / mean_A_emp

    x0 = np.log([max(k0, 0.1), max(theta0, 0.1), max(c0, 0.01)])
    res = minimize(loss, x0, method='Nelder-Mead',
                    options=dict(xatol=1e-8, fatol=1e-10, maxiter=5000))
    k_fit, theta_fit, c_fit = np.exp(res.x)

    pred = np.array(model_moments(k_fit, theta_fit, c_fit))
    print("  Model fit (Gamma-Poisson shared-latent): "
          f"k={k_fit:.4f}  theta={theta_fit:.4f}  c={c_fit:.4f}")
    print(f"  {'quantity':>12} | {'empirical':>12} | {'model':>12} | {'rel. diff':>10}")
    for name, emp, mod in zip(
        ['mean_A', 'var_A', 'mean_B', 'var_B', 'corr_AB'], targets, pred
    ):
        reldiff = (mod - emp) / emp if abs(emp) > 1e-12 else (mod - emp)
        print(f"  {name:>12} | {emp:>12.4f} | {mod:>12.4f} | {reldiff:>+9.2%}")
    max_reldiff = np.max(np.abs((pred[:4] - targets[:4]) / targets[:4]))
    if max_reldiff > 0.15:
        print(f"  NOTE: largest moment relative mismatch is {max_reldiff:.1%} -- "
              f"the 3-parameter model is a limited-flexibility stand-in, not a "
              f"perfect fit to real QCD multiplicity correlations. The closure "
              f"test below validates the ESTIMATOR PIPELINE under a scenario "
              f"calibrated to resemble your data, not a perfect replica of it.")

    return dict(k=k_fit, theta=theta_fit, c=c_fit,
                mean_A_emp=mean_A_emp, var_A_emp=var_A_emp,
                mean_B_emp=mean_B_emp, var_B_emp=var_B_emp, corr_emp=corr_emp)


def generate_from_model(n_events, k, theta, c, rng):
    lam = rng.gamma(shape=k, scale=theta, size=n_events)
    n_a = rng.poisson(lam)
    n_b = rng.poisson(c * lam)
    return n_a, n_b


# =============================================================================
# Step 2: pseudo-truth from a huge reference sample
# =============================================================================

def compute_pseudo_truth(k, theta, c, ref_n, rng):
    print(f"\n  Generating reference sample (N={ref_n:,}) from the fitted model "
          f"for pseudo-truth (naive estimator bias ~1/N is negligible at this scale)...")
    na_ref, nb_ref = generate_from_model(ref_n, k, theta, c, rng)
    I_true, info = naive_MI(na_ref, nb_ref)
    S_A_true = shannon_entropy_wrap(na_ref)
    S_B_true = shannon_entropy_wrap(nb_ref)
    S_AB_true = joint_entropy_wrap(na_ref, nb_ref)
    SBA_true = S_AB_true - S_A_true
    NMI_true = I_true / min(S_A_true, S_B_true)
    print(f"  I_true (pseudo-truth)     = {I_true:.5f} nats")
    print(f"  S(B|A)_true (pseudo-truth) = {SBA_true:.5f} nats")
    print(f"  NMI_true (pseudo-truth)    = {NMI_true:.5f}  "
          f"(S_A_true={S_A_true:.4f}, S_B_true={S_B_true:.4f})")
    return I_true, SBA_true, NMI_true, na_ref, nb_ref


def shannon_entropy_wrap(counts_array):
    vals, counts = np.unique(counts_array, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log(p))


def joint_entropy_wrap(n_a, n_b):
    pairs = np.stack([n_a, n_b], axis=1)
    _, counts = np.unique(pairs, axis=0, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log(p))


# =============================================================================
# Step 3: closure test -- repeated draws at real N, check recovery
# =============================================================================

def run_closure_test(k, theta, c, N, alphabet_a, alphabet_b, I_true, SBA_true,
                      n_repeats, n_boot, n_shuffles, rng):
    print(f"\n  Running closure test: {n_repeats} independent draws at N={N:,} "
          f"from the fitted model, each analyzed with naive/MM/shuffle/NSB...")

    results = {k_: [] for k_ in ['I_naive', 'I_mm', 'I_shuffle', 'I_nsb', 'I_nsb_err',
                                   'SBA_nsb', 'SBA_nsb_err', 'NMI_nsb', 'NMI_nsb_err']}

    for rep in range(n_repeats):
        na, nb = generate_from_model(N, k, theta, c, rng)

        I_naive, _ = naive_MI(na, nb)
        I_mm, _, _ = miller_madow_MI(na, nb)
        I_sh, _, _, _ = shuffle_correction_MI(na, nb, n_shuffles=n_shuffles, rng=rng)
        nsb_res = nsb_all(na, nb, alphabet_a, alphabet_b, n_boot=n_boot, rng=rng)

        results['I_naive'].append(I_naive)
        results['I_mm'].append(I_mm)
        results['I_shuffle'].append(I_sh)
        results['I_nsb'].append(nsb_res['I'])
        results['I_nsb_err'].append(nsb_res['I_err'])
        results['SBA_nsb'].append(nsb_res['SBA'])
        results['SBA_nsb_err'].append(nsb_res['SBA_err'])
        results['NMI_nsb'].append(nsb_res['NMI'])
        results['NMI_nsb_err'].append(nsb_res['NMI_err'])

        if (rep + 1) % max(1, n_repeats // 5) == 0:
            print(f"    ... {rep+1}/{n_repeats} repeats done")

    return {k_: np.array(v) for k_, v in results.items()}


def report_and_plot(results, I_true, SBA_true, NMI_true, N, out_prefix):
    print(f"\n{'='*70}\nCLOSURE TEST RESULTS (N={N:,} per repeat)\n{'='*70}")
    print(f"I_true (pseudo-truth) = {I_true:.5f} nats\n")

    print(f"{'method':>14} | {'mean estimate':>14} | {'std across reps':>16} | {'bias':>10} | {'bias/std':>9}")
    print("-" * 78)
    for key, label in [('I_naive', 'naive'), ('I_mm', 'Miller-Madow'),
                        ('I_shuffle', 'shuffle'), ('I_nsb', 'NSB')]:
        vals = results[key]
        mean = vals.mean()
        std = vals.std(ddof=1)
        bias = mean - I_true
        print(f"{label:>14} | {mean:>14.5f} | {std:>16.5f} | {bias:>+10.5f} | {bias/std:>+9.2f}")

    # ---- NSB calibration: pull distribution ----
    pulls = (results['I_nsb'] - I_true) / results['I_nsb_err']
    print(f"\nNSB pull distribution (estimate - truth) / reported_sigma:")
    print(f"  mean pull = {pulls.mean():.3f}  (should be ~0 if unbiased)")
    print(f"  std  pull = {pulls.std(ddof=1):.3f}  (should be ~1 if uncertainty is well-calibrated)")
    frac_1sigma = np.mean(np.abs(pulls) < 1.0)
    frac_2sigma = np.mean(np.abs(pulls) < 2.0)
    print(f"  fraction within +/-1 sigma: {frac_1sigma:.1%}  (expect ~68% if calibrated)")
    print(f"  fraction within +/-2 sigma: {frac_2sigma:.1%}  (expect ~95% if calibrated)")
    if results['I_nsb_err'].mean() > 0 and abs(pulls.std(ddof=1) - 1.0) > 0.3:
        direction = "UNDER-covering (error bars too small)" if pulls.std(ddof=1) > 1.3 \
            else "OVER-covering (error bars too large / conservative)"
        print(f"  NOTE: pull std deviates from 1 by >0.3 -- NSB uncertainty appears "
              f"{direction} for data shaped like this, at N={N:,}.")

    # ---- plots ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    methods = ['naive', 'Miller-Madow', 'shuffle', 'NSB']
    means = [results['I_naive'].mean(), results['I_mm'].mean(),
             results['I_shuffle'].mean(), results['I_nsb'].mean()]
    stds = [results['I_naive'].std(ddof=1), results['I_mm'].std(ddof=1),
            results['I_shuffle'].std(ddof=1), results['I_nsb'].std(ddof=1)]
    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#9467bd']
    ax.bar(methods, means, yerr=stds, capsize=5, color=colors)
    ax.axhline(I_true, color='black', ls='--', lw=2, label=f'$I_{{true}}$={I_true:.4f}')
    ax.set_ylabel('recovered I(A:B) (nats)')
    ax.set_title(f'Closure test: recovery at N={N:,}\n(error bars: spread across {len(results["I_naive"])} repeats)')
    ax.tick_params(axis='x', rotation=20)
    ax.legend()

    ax = axes[1]
    ax.hist(pulls, bins=max(8, len(pulls)//4), density=True, color='#9467bd',
            alpha=0.6, edgecolor='black', label='NSB pulls')
    x = np.linspace(-4, 4, 200)
    ax.plot(x, stats.norm.pdf(x), 'k--', lw=2, label='standard normal')
    ax.set_xlabel(r'$(\hat I_{NSB} - I_{true}) / \sigma_{NSB}$')
    ax.set_ylabel('density')
    ax.set_title('NSB calibration: pull distribution')
    ax.legend(fontsize=9)

    ax = axes[2]
    reps_idx = np.arange(len(results['I_nsb']))
    ax.errorbar(reps_idx, results['I_nsb'], yerr=results['I_nsb_err'], fmt='o',
                color='#9467bd', capsize=2, markersize=4, alpha=0.7)
    ax.axhline(I_true, color='black', ls='--', lw=2, label=f'$I_{{true}}$')
    ax.set_xlabel('repeat #')
    ax.set_ylabel('NSB estimate ± bootstrap σ')
    ax.set_title('NSB per-repeat estimates vs. truth')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_closure_test.png", dpi=150)
    print(f"\n  saved {out_prefix}_closure_test.png")

    # ---- NEW: NMI closure test (separate report + separate plot file;
    # the figure above is untouched) ----
    print(f"\n{'='*70}\nNMI CLOSURE TEST (N={N:,} per repeat)\n{'='*70}")
    print(f"NMI_true (pseudo-truth) = {NMI_true:.5f}  "
          f"(classically bounded in [0,1]; NMI>1 would be a classical-bound violation)\n")

    nmi_mean = results['NMI_nsb'].mean()
    nmi_std = results['NMI_nsb'].std(ddof=1)
    nmi_bias = nmi_mean - NMI_true
    print(f"{'method':>14} | {'mean estimate':>14} | {'std across reps':>16} | {'bias':>10} | {'bias/std':>9}")
    print("-" * 78)
    print(f"{'NSB':>14} | {nmi_mean:>14.5f} | {nmi_std:>16.5f} | {nmi_bias:>+10.5f} | {nmi_bias/nmi_std:>+9.2f}")

    nmi_pulls = (results['NMI_nsb'] - NMI_true) / results['NMI_nsb_err']
    print(f"\nNSB NMI pull distribution (estimate - truth) / reported_sigma:")
    print(f"  mean pull = {nmi_pulls.mean():.3f}  (should be ~0 if unbiased)")
    print(f"  std  pull = {nmi_pulls.std(ddof=1):.3f}  (should be ~1 if uncertainty is well-calibrated)")
    nmi_frac_1sigma = np.mean(np.abs(nmi_pulls) < 1.0)
    nmi_frac_2sigma = np.mean(np.abs(nmi_pulls) < 2.0)
    print(f"  fraction within +/-1 sigma: {nmi_frac_1sigma:.1%}  (expect ~68% if calibrated)")
    print(f"  fraction within +/-2 sigma: {nmi_frac_2sigma:.1%}  (expect ~95% if calibrated)")

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes2[0]
    ax.bar(['NSB'], [nmi_mean], yerr=[nmi_std], capsize=5, color='#2ca02c')
    ax.axhline(NMI_true, color='black', ls='--', lw=2, label=f'$NMI_{{true}}$={NMI_true:.4f}')
    ax.axhline(1, color='red', ls=':', lw=1.5, label='classical bound (NMI=1)')
    ax.set_ylabel('recovered NMI')
    ax.set_title(f'NMI closure test: recovery at N={N:,}\n(error bar: spread across {len(results["NMI_nsb"])} repeats)')
    ax.legend(fontsize=8)

    ax = axes2[1]
    ax.hist(nmi_pulls, bins=max(8, len(nmi_pulls)//4), density=True, color='#2ca02c',
            alpha=0.6, edgecolor='black', label='NSB NMI pulls')
    ax.plot(x, stats.norm.pdf(x), 'k--', lw=2, label='standard normal')
    ax.set_xlabel(r'$(\hat{NMI}_{NSB} - NMI_{true}) / \sigma_{NMI}$')
    ax.set_ylabel('density')
    ax.set_title('NSB NMI calibration: pull distribution')
    ax.legend(fontsize=9)

    ax = axes2[2]
    ax.errorbar(reps_idx, results['NMI_nsb'], yerr=results['NMI_nsb_err'], fmt='o',
                color='#2ca02c', capsize=2, markersize=4, alpha=0.7)
    ax.axhline(NMI_true, color='black', ls='--', lw=2, label='$NMI_{true}$')
    ax.set_xlabel('repeat #')
    ax.set_ylabel('NSB NMI estimate ± bootstrap σ')
    ax.set_title('NSB NMI per-repeat estimates vs. truth')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_closure_test_NMI.png", dpi=150)
    print(f"\n  saved {out_prefix}_closure_test_NMI.png")

    # ---- NEW: S(B|A) closure test (separate report + separate plot file;
    # the I and NMI figures above are untouched). This is the actual
    # quantum-witness quantity -- worth its own dedicated calibration
    # check rather than inferring its behavior from I's. Purely
    # diagnostic: no bias/coverage correction is applied anywhere here. ----
    print(f"\n{'='*70}\nS(B|A) CLOSURE TEST (N={N:,} per repeat)\n{'='*70}")
    print(f"S(B|A)_true (pseudo-truth) = {SBA_true:.5f} nats  "
          f"(classically must be >=0; a well-calibrated pipeline should never "
          f"call this negative at high significance when the true value is "
          f"this solidly positive)\n")

    sba_mean = results['SBA_nsb'].mean()
    sba_std = results['SBA_nsb'].std(ddof=1)
    sba_bias = sba_mean - SBA_true
    print(f"{'method':>14} | {'mean estimate':>14} | {'std across reps':>16} | {'bias':>10} | {'bias/std':>9}")
    print("-" * 78)
    print(f"{'NSB':>14} | {sba_mean:>14.5f} | {sba_std:>16.5f} | {sba_bias:>+10.5f} | {sba_bias/sba_std:>+9.2f}")

    sba_pulls = (results['SBA_nsb'] - SBA_true) / results['SBA_nsb_err']
    print(f"\nNSB S(B|A) pull distribution (estimate - truth) / reported_sigma:")
    print(f"  mean pull = {sba_pulls.mean():.3f}  (should be ~0 if unbiased)")
    print(f"  std  pull = {sba_pulls.std(ddof=1):.3f}  (should be ~1 if uncertainty is well-calibrated)")
    sba_frac_1sigma = np.mean(np.abs(sba_pulls) < 1.0)
    sba_frac_2sigma = np.mean(np.abs(sba_pulls) < 2.0)
    print(f"  fraction within +/-1 sigma: {sba_frac_1sigma:.1%}  (expect ~68% if calibrated)")
    print(f"  fraction within +/-2 sigma: {sba_frac_2sigma:.1%}  (expect ~95% if calibrated)")
    if results['SBA_nsb_err'].mean() > 0 and abs(sba_pulls.std(ddof=1) - 1.0) > 0.3:
        direction = "UNDER-covering (error bars too small)" if sba_pulls.std(ddof=1) > 1.3 \
            else "OVER-covering (error bars too large / conservative)"
        print(f"  NOTE: pull std deviates from 1 by >0.3 -- NSB uncertainty on "
              f"S(B|A) specifically appears {direction} for data shaped like "
              f"this, at N={N:,}. Since S(B|A) is the quantity actually used "
              f"for the quantum-witness claim, check this number directly "
              f"rather than assuming it matches I's or NMI's calibration.")

    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes3[0]
    ax.bar(['NSB'], [sba_mean], yerr=[sba_std], capsize=5, color='purple')
    ax.axhline(SBA_true, color='black', ls='--', lw=2, label=f'$S(B|A)_{{true}}$={SBA_true:.4f}')
    ax.axhline(0, color='red', ls=':', lw=1.5, label='classical floor')
    ax.set_ylabel('recovered S(B|A) (nats)')
    ax.set_title(f'S(B|A) closure test: recovery at N={N:,}\n(error bar: spread across {len(results["SBA_nsb"])} repeats)')
    ax.legend(fontsize=8)

    ax = axes3[1]
    ax.hist(sba_pulls, bins=max(8, len(sba_pulls)//4), density=True, color='purple',
            alpha=0.6, edgecolor='black', label='NSB S(B|A) pulls')
    ax.plot(x, stats.norm.pdf(x), 'k--', lw=2, label='standard normal')
    ax.set_xlabel(r'$(\hat{S}(B|A)_{NSB} - S(B|A)_{true}) / \sigma_{S(B|A)}$')
    ax.set_ylabel('density')
    ax.set_title('NSB S(B|A) calibration: pull distribution')
    ax.legend(fontsize=9)

    ax = axes3[2]
    ax.errorbar(reps_idx, results['SBA_nsb'], yerr=results['SBA_nsb_err'], fmt='o',
                color='purple', capsize=2, markersize=4, alpha=0.7)
    ax.axhline(SBA_true, color='black', ls='--', lw=2, label='$S(B|A)_{true}$')
    ax.axhline(0, color='red', ls=':', lw=1.5, label='classical floor')
    ax.set_xlabel('repeat #')
    ax.set_ylabel('NSB S(B|A) estimate ± bootstrap σ')
    ax.set_title('NSB S(B|A) per-repeat estimates vs. truth')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_closure_test_SBA.png", dpi=150)
    print(f"\n  saved {out_prefix}_closure_test_SBA.png")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root_file", help="PYTHIA8 dijet ROOT file")
    parser.add_argument("--level", choices=["truth", "det"], default="truth")
    parser.add_argument("--alphabet", type=int, default=80,
                         help="same alphabet you use in the real analysis -- the "
                              "closure test should match your actual analysis settings")
    parser.add_argument("--n-repeats", type=int, default=30,
                         help="independent draws at real N for the closure test (default 30)")
    parser.add_argument("--n-boot", type=int, default=50,
                         help="bootstrap resamples per repeat for NSB uncertainty (default 50)")
    parser.add_argument("--n-shuffles", type=int, default=100,
                         help="shuffles for the shuffle-correction estimator (default 100)")
    parser.add_argument("--ref-n", type=int, default=5_000_000,
                         help="reference sample size for pseudo-truth (default 5,000,000)")
    parser.add_argument("--closure-n", type=int, default=None,
                         help="sample size to use per closure-test repeat "
                              "(default: same as your real dataset's N)")
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--figs-dir", default="figs",
                         help="directory to save figures in (relative to current working "
                              "directory by default; pass an absolute path if running "
                              "from elsewhere, e.g. $HOME)")
    args = parser.parse_args()

    if args.out_prefix is None:
        args.out_prefix = f"closure_test_{args.level}"
    os.makedirs(args.figs_dir, exist_ok=True)
    args.out_prefix = os.path.join(args.figs_dir, args.out_prefix)
    print(f"Figures will be saved under: {args.figs_dir}")

    rng = np.random.default_rng(20260709)

    print(f"Loading {args.root_file} (level='{args.level}') ...")
    data = load_dijet_tree(args.root_file, level=args.level)
    n_a = data["jet1_mult"].astype(int)
    n_b = data["jet2_mult"].astype(int)
    N_real = len(n_a)
    print(f"Loaded {N_real} events.")

    N_closure = args.closure_n if args.closure_n is not None else N_real

    print("\n--- Step 1: fitting Gamma-Poisson model to real data ---")
    fit = fit_model(n_a, n_b)

    print("\n--- Step 2: pseudo-truth from fitted model ---")
    I_true, SBA_true, NMI_true, na_ref, nb_ref = compute_pseudo_truth(
        fit['k'], fit['theta'], fit['c'], args.ref_n, rng)

    print("\n--- Step 3: closure test (recovery at your actual N) ---")
    results = run_closure_test(
        fit['k'], fit['theta'], fit['c'], N_closure,
        args.alphabet, args.alphabet, I_true, SBA_true,
        args.n_repeats, args.n_boot, args.n_shuffles, rng)

    report_and_plot(results, I_true, SBA_true, NMI_true, N_closure, args.out_prefix)

    print("\nDone.")


if __name__ == "__main__":
    main()
