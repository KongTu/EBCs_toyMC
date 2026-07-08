#!/usr/bin/env python3
"""
analyze_dijet_entropy.py

Read the PYTHIA8 dijet ROOT ntuple produced by main_dijet_pp200.cc and
measure the mutual information I(A:B) and conditional entropy S(B|A)
between the two leading back-to-back jets' charged-hadron multiplicities,
using the same four estimators developed and validated on the toy Monte
Carlo model:

    naive (plug-in)  ->  known to be biased upward at finite N
    Miller-Madow     ->  first-order analytic bias correction
    shuffle-subtraction -> data-driven bias correction (label permutation)
    NSB              ->  Bayesian posterior-mean entropy (recommended)

For NSB we also compute the analytic posterior VARIANCE (in addition to
the mean), and cross-check it against a bootstrap of the actual dataset
(the real-data analogue of the toy study's Monte-Carlo-repetition noise
floor -- here we resample the one dataset we actually have, since there
is no known generative model to redraw fresh samples from).

Usage:
    python3 analyze_dijet_entropy.py pythia_dijet_pp200.root
    python3 analyze_dijet_entropy.py pythia_dijet_pp200.root --alphabet 80
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, digamma, polygamma
from scipy.stats import nbinom, poisson

try:
    import uproot
except ImportError:
    sys.exit("This script needs uproot: pip install uproot")


# =============================================================================
# Estimator library -- ported and validated on the toy Monte Carlo study.
# See the companion notebook (dijet_MI_bias_demo.ipynb) for derivations,
# numerical validation against brute-force sampling, and worked examples.
# =============================================================================

def shannon_entropy(counts):
    """Plug-in Shannon entropy (nats) from an array of raw category counts."""
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return -np.sum(p * np.log(p))


def entropies_from_samples(n_a, n_b):
    """Return (S_A, S_B, S_AB, K_A, K_B, K_AB, N) naive plug-in estimates."""
    N = len(n_a)
    vals_a, counts_a = np.unique(n_a, return_counts=True)
    vals_b, counts_b = np.unique(n_b, return_counts=True)
    S_A = shannon_entropy(counts_a)
    S_B = shannon_entropy(counts_b)
    K_A, K_B = len(vals_a), len(vals_b)

    pairs = np.stack([n_a, n_b], axis=1)
    _, joint_counts = np.unique(pairs, axis=0, return_counts=True)
    S_AB = shannon_entropy(joint_counts)
    K_AB = len(joint_counts)

    return S_A, S_B, S_AB, K_A, K_B, K_AB, N


def naive_MI(n_a, n_b):
    S_A, S_B, S_AB, K_A, K_B, K_AB, N = entropies_from_samples(n_a, n_b)
    I = S_A + S_B - S_AB
    return I, dict(S_A=S_A, S_B=S_B, S_AB=S_AB, K_A=K_A, K_B=K_B, K_AB=K_AB, N=N)


def miller_madow_MI(n_a, n_b):
    """Exact leading-order Miller-Madow correction (uses OBSERVED K_AB)."""
    S_A, S_B, S_AB, K_A, K_B, K_AB, N = entropies_from_samples(n_a, n_b)
    I_naive = S_A + S_B - S_AB
    bias = (K_AB - K_A - K_B + 1) / (2.0 * N)
    return I_naive - bias, I_naive, bias


def shuffle_correction_MI(n_a, n_b, n_shuffles=200, rng=None):
    """Bias correction via label-permutation null (see toy-study notebook
    for the caveat that this tends to OVERcorrect for strong correlations)."""
    if rng is None:
        rng = np.random.default_rng()
    I_naive, _ = naive_MI(n_a, n_b)
    n_b = np.asarray(n_b)
    I_shuffled = np.empty(n_shuffles)
    for i in range(n_shuffles):
        perm = rng.permutation(len(n_b))
        I_shuffled[i], _ = naive_MI(n_a, n_b[perm])
    bias_est = I_shuffled.mean()
    bias_err = I_shuffled.std(ddof=1) / np.sqrt(n_shuffles)
    return I_naive - bias_est, I_naive, bias_est, bias_err


def nsb_entropy_and_variance(counts_nonzero, K, beta_grid=None):
    """
    Vectorized NSB posterior MEAN and VARIANCE of entropy (nats), given
    occupied-bin counts and total alphabet size K. Validated against
    brute-force Dirichlet sampling in the toy-study notebook (agreement
    within ~0.3% for the mean/single-beta variance, ~3% for the full
    beta-integrated variance vs. hierarchical Monte Carlo).

    IMPORTANT: K (the alphabet size) must be fixed from physical/detector
    considerations BEFORE looking at the data -- e.g. a generous upper
    bound on plausible charged multiplicity for an R=0.4 jet at these
    kinematics -- not read off the observed max multiplicity, or a
    data-dependent bias is reintroduced through the back door.
    """
    counts_nonzero = np.asarray(counts_nonzero, dtype=float)
    N = counts_nonzero.sum()
    K_obs = len(counts_nonzero)
    K_zero = K - K_obs
    if K_zero < 0:
        raise ValueError(
            f"K={K} is smaller than the observed number of occupied bins "
            f"({K_obs}) -- increase the alphabet size."
        )

    if beta_grid is None:
        beta_grid = np.logspace(-6, 3, 300)
    b = beta_grid[:, None]
    c = counts_nonzero[None, :]
    A = N + K * beta_grid

    # ---- posterior mean entropy given beta ----
    alpha_occ = c + b
    alpha_zero = b[:, 0]
    sum_term = np.sum(alpha_occ * digamma(alpha_occ + 1), axis=1) \
        + K_zero * alpha_zero * digamma(alpha_zero + 1)
    means = digamma(A + 1) - sum_term / A

    # ---- posterior variance of entropy given beta (Wolpert-Wolf 1994) ----
    Ap2 = (A + 2)[:, None]
    d_occ = digamma(alpha_occ + 1) - digamma(Ap2)
    d_zero = digamma(alpha_zero + 1) - digamma(A + 2)

    def term1_piece_occ(a):
        return (a * (a + 1) / (A[:, None] * (A[:, None] + 1))) * (
            polygamma(1, a + 2) - polygamma(1, Ap2)
            + (digamma(a + 2) - digamma(Ap2)) ** 2)

    term1_zero = K_zero * (alpha_zero * (alpha_zero + 1) / (A * (A + 1))) * (
        polygamma(1, alpha_zero + 2) - polygamma(1, A + 2)
        + (digamma(alpha_zero + 2) - digamma(A + 2)) ** 2)
    term1 = np.sum(term1_piece_occ(alpha_occ), axis=1) + term1_zero

    sum_a2 = np.sum(alpha_occ ** 2, axis=1) + K_zero * alpha_zero ** 2
    sum_ad = np.sum(alpha_occ * d_occ, axis=1) + K_zero * alpha_zero * d_zero
    sum_a2d2 = np.sum((alpha_occ ** 2) * (d_occ ** 2), axis=1) \
        + K_zero * (alpha_zero ** 2) * (d_zero ** 2)
    term2 = (1.0 / (A * (A + 1))) * (
        -polygamma(1, A + 2) * (A ** 2 - sum_a2) + sum_ad ** 2 - sum_a2d2)

    varis = (term1 + term2) - means ** 2

    # ---- NSB weighting over beta, then law of total variance ----
    logP = (gammaln(K * beta_grid) - gammaln(N + K * beta_grid) + gammaln(N + 1)
            - np.sum(gammaln(c + 1)) + np.sum(gammaln(c + b) - gammaln(b), axis=1))
    rho = K * polygamma(1, K * beta_grid + 1) - polygamma(1, beta_grid + 1)
    logw = logP + np.log(rho)
    logw -= logw.max()
    w = np.exp(logw)

    t = np.log(beta_grid)
    weight_t = w * beta_grid
    Z = np.trapezoid(weight_t, t)

    S_nsb = np.trapezoid(weight_t * means, t) / Z
    E_var_given_beta = np.trapezoid(weight_t * varis, t) / Z
    var_of_mean_given_beta = np.trapezoid(weight_t * (means - S_nsb) ** 2, t) / Z
    Var_nsb = E_var_given_beta + var_of_mean_given_beta

    return S_nsb, Var_nsb


def nsb_entropy_from_counts(counts_nonzero, K, beta_grid=None):
    """Mean-only convenience wrapper (faster call sites that don't need variance)."""
    S_nsb, _ = nsb_entropy_and_variance(counts_nonzero, K, beta_grid)
    return S_nsb


def nsb_MI(n_a, n_b, alphabet_a, alphabet_b):
    va, ca = np.unique(n_a, return_counts=True)
    vb, cb = np.unique(n_b, return_counts=True)
    pairs = np.stack([n_a, n_b], axis=1)
    _, cab = np.unique(pairs, axis=0, return_counts=True)

    K_joint = alphabet_a * alphabet_b
    S_A_nsb = nsb_entropy_from_counts(ca, alphabet_a)
    S_B_nsb = nsb_entropy_from_counts(cb, alphabet_b)
    S_AB_nsb = nsb_entropy_from_counts(cab, K_joint)

    return S_A_nsb + S_B_nsb - S_AB_nsb, dict(S_A=S_A_nsb, S_B=S_B_nsb, S_AB=S_AB_nsb)


def sba_nsb_with_analytic_uncertainty(n_a, n_b, alphabet_a, alphabet_b):
    """S(B|A) via NSB, plus an analytic sigma using the independence
    approximation Var(S_AB - S_A) ~= Var(S_AB) + Var(S_A). See toy-study
    notebook Section 7 for why this is a conservative (slightly too large)
    approximation: it drops the (typically positive) covariance between
    S_A and S_AB that arises because both are estimated from the same
    event sample."""
    va, ca = np.unique(n_a, return_counts=True)
    pairs = np.stack([n_a, n_b], axis=1)
    _, cab = np.unique(pairs, axis=0, return_counts=True)

    K_joint = alphabet_a * alphabet_b
    S_A_nsb, Var_A_nsb = nsb_entropy_and_variance(ca, alphabet_a)
    S_AB_nsb, Var_AB_nsb = nsb_entropy_and_variance(cab, K_joint)

    SBA_nsb = S_AB_nsb - S_A_nsb
    sigma_analytic = np.sqrt(Var_AB_nsb + Var_A_nsb)
    return SBA_nsb, sigma_analytic


def sba_bootstrap_uncertainty(n_a, n_b, alphabet_a, alphabet_b,
                                n_boot=200, rng=None):
    """Real-data uncertainty via bootstrap resampling of the ACTUAL dataset
    (resample events with replacement). This is the real-data analogue of
    the toy study's Monte-Carlo-repetition noise floor -- here we don't
    have a known generative model to redraw fresh independent samples
    from, so we resample the one dataset we have instead."""
    if rng is None:
        rng = np.random.default_rng()
    n_a = np.asarray(n_a)
    n_b = np.asarray(n_b)
    N = len(n_a)
    SBA_boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, N, size=N)
        SBA_boot[i], _ = sba_nsb_with_analytic_uncertainty(
            n_a[idx], n_b[idx], alphabet_a, alphabet_b)
    return SBA_boot.mean(), SBA_boot.std(ddof=1)


def bootstrap_all_estimators(n_a, n_b, alphabet_a, alphabet_b,
                               n_boot=50, n_shuffles_inner=30, rng=None):
    """
    Unified bootstrap: for each of n_boot resamples (with replacement) of
    the actual dataset, compute ALL of naive/Miller-Madow/shuffle/NSB I(A:B)
    and naive/Miller-Madow/NSB S(B|A) from the SAME resampled indices, then
    report (mean, std) for each. This gives every method's point estimate
    and uncertainty from one consistent procedure, so bars/curves for
    different methods are directly comparable rather than mixing different
    uncertainty conventions.

    Note: there is no shuffle-subtraction analogue for S(B|A) -- shuffle-
    subtraction specifically targets the joint-vs-marginal bias mechanism
    behind I(A:B) (see toy-study Section 5); it doesn't have an established,
    validated equivalent for a single conditional entropy. Miller-Madow's
    per-entropy correction, by contrast, applies directly to S_A and S_AB
    individually and so generalizes to S(B|A) trivially -- that's why the
    S(B|A) comparison below has three methods, not four.

    n_shuffles_inner is deliberately smaller than a typical standalone
    shuffle-correction call (default 200 elsewhere) since this function
    already runs shuffle-correction once per bootstrap replicate -- the
    outer bootstrap loop is what's driving statistical precision here, not
    the inner shuffle-null precision at any single replicate.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_a = np.asarray(n_a)
    n_b = np.asarray(n_b)
    N = len(n_a)

    keys = ['I_naive', 'I_mm', 'I_shuffle', 'I_nsb', 'SBA_naive', 'SBA_mm', 'SBA_nsb']
    vals = {k: np.empty(n_boot) for k in keys}

    for i in range(n_boot):
        idx = rng.integers(0, N, size=N)
        ra, rb = n_a[idx], n_b[idx]

        S_A_naive, S_B_naive, S_AB_naive, K_A, K_B, K_AB, Nr = entropies_from_samples(ra, rb)
        I_naive = S_A_naive + S_B_naive - S_AB_naive
        SBA_naive = S_AB_naive - S_A_naive

        mm_bias = (K_AB - K_A - K_B + 1) / (2.0 * Nr)
        I_mm = I_naive - mm_bias
        S_A_mm = S_A_naive - (K_A - 1) / (2.0 * Nr)
        S_AB_mm = S_AB_naive - (K_AB - 1) / (2.0 * Nr)
        SBA_mm = S_AB_mm - S_A_mm

        I_shuffle, _, _, _ = shuffle_correction_MI(ra, rb, n_shuffles=n_shuffles_inner, rng=rng)

        I_nsb, nsb_info = nsb_MI(ra, rb, alphabet_a, alphabet_b)
        SBA_nsb = nsb_info['S_AB'] - nsb_info['S_A']

        vals['I_naive'][i] = I_naive
        vals['I_mm'][i] = I_mm
        vals['I_shuffle'][i] = I_shuffle
        vals['I_nsb'][i] = I_nsb
        vals['SBA_naive'][i] = SBA_naive
        vals['SBA_mm'][i] = SBA_mm
        vals['SBA_nsb'][i] = SBA_nsb

    return {k: (vals[k].mean(), vals[k].std(ddof=1)) for k in keys}


# =============================================================================
# Data loading
# =============================================================================

def load_dijet_tree(root_file, level="truth", tree_name="dijet"):
    """
    level = "truth" -> jet1_mult_truth, jet2_mult_truth, jet1_pt_truth, ...
            (wide/idealized acceptance, no tracking inefficiency)
    level = "det"   -> jet1_mult_det, jet2_mult_det, ... (STAR-TPC-like
            acceptance + tracking efficiency applied at generation time),
            automatically filtered to hasDetDijet==1 (events where a
            back-to-back dijet also survived detector-level reconstruction;
            see main_dijet_pp200.cc for the acceptance/efficiency model)
    """
    suffix = {"truth": "_truth", "det": "_det"}[level]
    branches = [f"jet1_mult{suffix}", f"jet2_mult{suffix}",
                f"jet1_pt{suffix}", f"jet2_pt{suffix}", f"dPhi{suffix}"]
    if level == "det":
        branches.append("hasDetDijet")

    with uproot.open(root_file) as f:
        tree = f[tree_name]
        raw = tree.arrays(branches, library="np")

    if level == "det":
        mask = raw["hasDetDijet"] == 1
        n_before = len(mask)
        n_after = mask.sum()
        print(f"  detector-level filter (hasDetDijet==1): kept {n_after}/{n_before} "
              f"({100.0*n_after/n_before:.1f}%) of truth-selected events")
    else:
        mask = np.ones(len(raw[branches[0]]), dtype=bool)

    return {
        "jet1_mult": raw[f"jet1_mult{suffix}"][mask].astype(int),
        "jet2_mult": raw[f"jet2_mult{suffix}"][mask].astype(int),
        "jet1_pt":   raw[f"jet1_pt{suffix}"][mask].astype(float),
        "jet2_pt":   raw[f"jet2_pt{suffix}"][mask].astype(float),
        "dPhi":      raw[f"dPhi{suffix}"][mask].astype(float),
    }


# =============================================================================
# Diagnostics ported directly from the toy study: bias-vs-subsample-size,
# now run on subsamples of the REAL generated statistics rather than fresh
# draws from a known model (we don't have a "true" answer here, but we can
# still see the naive estimator drift as a function of subsample size,
# exactly the effect demonstrated on the toy model).
# =============================================================================

def bias_vs_subsample_scan(n_a_full, n_b_full, N_grid, n_rep=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n_a_full = np.asarray(n_a_full)
    n_b_full = np.asarray(n_b_full)
    N_total = len(n_a_full)

    I_means, I_stds = [], []
    for N in N_grid:
        if N > N_total:
            I_means.append(np.nan); I_stds.append(np.nan)
            continue
        vals = []
        for _ in range(n_rep):
            idx = rng.choice(N_total, size=N, replace=False)
            I, _ = naive_MI(n_a_full[idx], n_b_full[idx])
            vals.append(I)
        I_means.append(np.mean(vals))
        I_stds.append(np.std(vals))
    return np.array(I_means), np.array(I_stds)


# =============================================================================
# i) Multiplicity distribution shape check: is it Negative-Binomial-like,
# as assumed in the toy Monte Carlo model (Section 4 of the toy study)?
# =============================================================================

def fit_nbd_moments(counts):
    """
    Method-of-moments fit of a Negative Binomial distribution to integer
    count data. NBD parametrization here follows scipy.stats.nbinom(r, p):
    mean = r(1-p)/p, var = r(1-p)/p^2 = mean/p.
    So: p = mean/var,  r = mean^2 / (var - mean).
    Requires var > mean (overdispersion relative to Poisson); if the data
    is under-dispersed (var <= mean) the NBD limit degenerates and the fit
    is not meaningful -- callers should check for this (we return None).
    """
    counts = np.asarray(counts, dtype=float)
    mean = counts.mean()
    var = counts.var(ddof=1)
    if var <= mean:
        return None  # not overdispersed; NBD moment fit undefined/degenerate
    p = mean / var
    r = mean**2 / (var - mean)
    return dict(r=r, p=p, mean=mean, var=var)


def plot_joint_multiplicity_distribution(n_a, n_b, out_prefix):
    """2D joint histogram of (N_A, N_B) -- the direct input to S_AB, and
    the quantity whose sparsity (occupied cells vs. total N_A x N_B grid)
    drives the finite-sample bias discussed throughout the toy study."""
    n_a = np.asarray(n_a); n_b = np.asarray(n_b)
    N = len(n_a)

    nmax_a, nmax_b = n_a.max(), n_b.max()
    edges_a = np.arange(-0.5, nmax_a + 1.5, 1.0)
    edges_b = np.arange(-0.5, nmax_b + 1.5, 1.0)

    H, _, _ = np.histogram2d(n_a, n_b, bins=[edges_a, edges_b])
    K_AB = int((H > 0).sum())
    K_A_full = len(edges_a) - 1
    K_B_full = len(edges_b) - 1

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # --- linear count scale ---
    ax = axes[0]
    im = ax.pcolormesh(edges_a, edges_b, H.T, cmap='inferno', shading='auto')
    fig.colorbar(im, ax=ax, label='events per cell')
    ax.set_xlabel(r'$N_A$ (leading jet mult.)')
    ax.set_ylabel(r'$N_B$ (subleading jet mult.)')
    ax.set_title('Joint distribution (linear scale)')
    ax.set_aspect('equal')

    # --- sqrt count scale: makes low-occupancy cells visible, same
    # convention used throughout the toy-study notebook ---
    ax = axes[1]
    im2 = ax.pcolormesh(edges_a, edges_b, np.sqrt(H.T), cmap='inferno', shading='auto')
    fig.colorbar(im2, ax=ax, label=r'$\sqrt{\mathrm{events\ per\ cell}}$')
    ax.set_xlabel(r'$N_A$ (leading jet mult.)')
    ax.set_ylabel(r'$N_B$ (subleading jet mult.)')
    ax.set_title('Joint distribution (sqrt scale, low occupancy visible)')
    ax.set_aspect('equal')

    fig.suptitle(
        rf'$N_{{events}}$={N}   occupied cells $K_{{AB}}$={K_AB} / '
        rf'{K_A_full}$\times${K_B_full}={K_A_full*K_B_full} grid '
        rf'(filled fraction = {K_AB/(K_A_full*K_B_full):.3f})',
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_joint_multiplicity_dist.png", dpi=150, bbox_inches='tight')
    print(f"  saved {out_prefix}_joint_multiplicity_dist.png")
    print(f"  N={N}, K_AB (occupied) = {K_AB}, full grid = {K_A_full}x{K_B_full} "
          f"= {K_A_full*K_B_full}, filled fraction = {K_AB/(K_A_full*K_B_full):.4f}")
    if K_AB / (K_A_full * K_B_full) < 0.1:
        print("  NOTE: joint grid is <10% filled -- exactly the sparse regime "
              "where the naive/Miller-Madow estimators are least reliable and "
              "NSB matters most (see toy-study Section 4/6).")


def plot_multiplicity_distributions(n_a, n_b, out_prefix):
    """Histogram jet A and jet B multiplicity, overlay method-of-moments
    NBD fit and a same-mean Poisson curve for visual contrast."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for ax, counts, label, color in [
        (axes[0], n_a, 'jet A (leading)', '#1f77b4'),
        (axes[1], n_b, 'jet B (subleading)', '#d62728'),
    ]:
        counts = np.asarray(counts)
        N = len(counts)
        nmax = counts.max()
        bins = np.arange(-0.5, nmax + 1.5, 1.0)
        hist_counts, _ = np.histogram(counts, bins=bins)
        n_vals = np.arange(0, nmax + 1)
        p_hat = hist_counts / N
        p_err = np.sqrt(hist_counts) / N   # Poisson counting error per bin

        ax.errorbar(n_vals, p_hat, yerr=p_err, fmt='o', color=color,
                    markersize=4, capsize=2, label=f'{label} data (N={N})')

        fit = fit_nbd_moments(counts)
        if fit is not None:
            nbd_pmf = nbinom.pmf(n_vals, fit['r'], fit['p'])
            ax.plot(n_vals, nbd_pmf, '-', color='black', lw=2,
                    label=(f"NBD fit (moments): r={fit['r']:.2f}, "
                           f"p={fit['p']:.3f}\n"
                           f"mean={fit['mean']:.2f}, var={fit['var']:.2f} "
                           f"(var/mean={fit['var']/fit['mean']:.2f})"))
            poisson_pmf = poisson.pmf(n_vals, fit['mean'])
            ax.plot(n_vals, poisson_pmf, '--', color='gray', lw=1.5,
                    label=f"Poisson, same mean ({fit['mean']:.2f})")
        else:
            ax.text(0.5, 0.5, 'var <= mean:\nunder-dispersed,\nNBD moment fit undefined',
                    transform=ax.transAxes, ha='center', va='center', fontsize=9)

        ax.set_xlabel('charged multiplicity')
        ax.set_ylabel('probability')
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_multiplicity_dists.png", dpi=150)
    print(f"  saved {out_prefix}_multiplicity_dists.png")


def plot_pt_distributions(jet1_pt, jet2_pt, out_prefix, n_bins=40):
    """i) Jet pT distributions (leading and subleading), with Poisson
    counting error bars per bin -- the pT-space analogue of the
    multiplicity distribution plot above."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for ax, pt, label, color in [
        (axes[0], jet1_pt, 'jet 1 (leading)', '#1f77b4'),
        (axes[1], jet2_pt, 'jet 2 (subleading)', '#d62728'),
    ]:
        pt = np.asarray(pt)
        N = len(pt)
        bins = np.linspace(pt.min(), pt.max(), n_bins + 1)
        counts, edges = np.histogram(pt, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        p_hat = counts / (N * widths)              # normalized density
        p_err = np.sqrt(counts) / (N * widths)      # Poisson counting error

        ax.errorbar(centers, p_hat, yerr=p_err, fmt='o', color=color,
                    markersize=4, capsize=2, label=f'{label} (N={N})')
        ax.set_xlabel(r'jet $p_T$ (GeV)')
        ax.set_ylabel(r'probability density (GeV$^{-1}$)')
        ax.set_title(label)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pt_dists.png", dpi=150)
    print(f"  saved {out_prefix}_pt_dists.png")


def plot_joint_pt_distribution(jet1_pt, jet2_pt, out_prefix, n_bins=40):
    """i) 2D joint distribution of (jet1_pt, jet2_pt) -- same linear +
    sqrt-scale convention as the joint multiplicity plot. A 2D histogram
    conveys its own statistical precision through per-cell counts (color
    scale) rather than individual error bars, so no yerr-style annotation
    is added here, consistent with how the joint multiplicity plot is
    handled."""
    jet1_pt = np.asarray(jet1_pt); jet2_pt = np.asarray(jet2_pt)
    N = len(jet1_pt)

    edges1 = np.linspace(jet1_pt.min(), jet1_pt.max(), n_bins + 1)
    edges2 = np.linspace(jet2_pt.min(), jet2_pt.max(), n_bins + 1)
    H, _, _ = np.histogram2d(jet1_pt, jet2_pt, bins=[edges1, edges2])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    im0 = axes[0].pcolormesh(edges1, edges2, H.T, cmap='inferno', shading='auto')
    fig.colorbar(im0, ax=axes[0], label='events per cell')
    axes[0].set_xlabel(r'jet 1 (leading) $p_T$ (GeV)')
    axes[0].set_ylabel(r'jet 2 (subleading) $p_T$ (GeV)')
    axes[0].set_title('Joint $p_T$ distribution (linear scale)')

    im1 = axes[1].pcolormesh(edges1, edges2, np.sqrt(H.T), cmap='inferno', shading='auto')
    fig.colorbar(im1, ax=axes[1], label=r'$\sqrt{\mathrm{events\ per\ cell}}$')
    axes[1].set_xlabel(r'jet 1 (leading) $p_T$ (GeV)')
    axes[1].set_ylabel(r'jet 2 (subleading) $p_T$ (GeV)')
    axes[1].set_title('Joint $p_T$ distribution (sqrt scale)')

    fig.suptitle(f'N_events = {N}', fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_joint_pt_dist.png", dpi=150, bbox_inches='tight')
    print(f"  saved {out_prefix}_joint_pt_dist.png")


# =============================================================================
# ii)-iv) Differential measurement vs. leading-jet pT: S_A, S_B, S_AB, I(A:B),
# S(B|A), each with NSB point estimates and analytic-posterior-variance
# error bars (same independence-approximation error propagation validated
# in the toy study, applied per pT bin). Quantile (equal-count) binning by
# default, since the jet pT spectrum falls steeply and fixed-width bins
# would leave some bins nearly empty -- exactly the regime these estimators
# are least reliable in.
# =============================================================================

def make_pt_bins(jet_pt, n_bins, mode='quantile'):
    jet_pt = np.asarray(jet_pt)
    if mode == 'quantile':
        edges = np.quantile(jet_pt, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)  # guard against degenerate/duplicate edges
    else:
        edges = np.linspace(jet_pt.min(), jet_pt.max(), n_bins + 1)
    return edges


def pt_differential_analysis(n_a, n_b, jet_pt, alphabet_a, alphabet_b,
                               n_bins=4, mode='quantile', min_events=30,
                               compare_methods=False, n_boot_compare=50,
                               n_shuffles_inner=30, rng=None):
    """
    Bin events by leading-jet pT and compute, in each bin:
    S_A, S_B, S_AB (NSB mean + analytic variance), I(A:B), S(B|A),
    each with propagated uncertainty (independence approximation, as
    validated/caveated in the toy study).

    If compare_methods=True, ALSO compute naive/Miller-Madow/shuffle/NSB
    I(A:B) and naive/Miller-Madow/NSB S(B|A) per bin via
    bootstrap_all_estimators, for the multi-method comparison plots.
    This roughly (n_bins x n_boot_compare) times more expensive than the
    NSB-only pass above, so it's opt-in.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_a = np.asarray(n_a); n_b = np.asarray(n_b); jet_pt = np.asarray(jet_pt)
    edges = make_pt_bins(jet_pt, n_bins, mode)

    results = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i+1]
        # last bin edge inclusive on the right to not drop the max-pT event
        if i < len(edges) - 2:
            mask = (jet_pt >= lo) & (jet_pt < hi)
        else:
            mask = (jet_pt >= lo) & (jet_pt <= hi)
        n_bin = mask.sum()
        if n_bin < min_events:
            print(f"  WARNING: pT bin [{lo:.1f},{hi:.1f}) has only {n_bin} events "
                  f"(< {min_events}) -- results in this bin will be noisy/unreliable.")
        if n_bin < 5:
            continue  # too few to even attempt

        na_bin, nb_bin = n_a[mask], n_b[mask]

        va, ca = np.unique(na_bin, return_counts=True)
        vb, cb = np.unique(nb_bin, return_counts=True)
        pairs = np.stack([na_bin, nb_bin], axis=1)
        _, cab = np.unique(pairs, axis=0, return_counts=True)

        K_joint = alphabet_a * alphabet_b
        S_A, Var_A = nsb_entropy_and_variance(ca, alphabet_a)
        S_B, Var_B = nsb_entropy_and_variance(cb, alphabet_b)
        S_AB, Var_AB = nsb_entropy_and_variance(cab, K_joint)

        I_val = S_A + S_B - S_AB
        # independence approximation (conservative; see toy-study caveats)
        I_var = Var_A + Var_B + Var_AB

        SBA_val = S_AB - S_A
        SBA_var = Var_AB + Var_A

        bin_result = dict(
            pt_lo=lo, pt_hi=hi, pt_mid=jet_pt[mask].mean(), N=n_bin,
            S_A=S_A, S_A_err=np.sqrt(Var_A),
            S_B=S_B, S_B_err=np.sqrt(Var_B),
            S_AB=S_AB, S_AB_err=np.sqrt(Var_AB),
            I=I_val, I_err=np.sqrt(I_var),
            SBA=SBA_val, SBA_err=np.sqrt(SBA_var),
        )

        if compare_methods:
            boot = bootstrap_all_estimators(na_bin, nb_bin, alphabet_a, alphabet_b,
                                             n_boot=n_boot_compare,
                                             n_shuffles_inner=n_shuffles_inner, rng=rng)
            for k, (mean, std) in boot.items():
                bin_result[k] = mean
                bin_result[k + '_err'] = std

        results.append(bin_result)

    return results


def plot_method_comparison_vs_pt(results, out_prefix):
    """iii) Compare naive / Miller-Madow / shuffle / NSB estimators of
    I(A:B), and naive / Miller-Madow / NSB estimators of S(B|A), as a
    function of leading-jet pT. All error bars come from the SAME
    bootstrap procedure (bootstrap_all_estimators) applied consistently
    across methods, so the comparison isn't mixing different uncertainty
    conventions. (No shuffle curve for S(B|A) -- see
    bootstrap_all_estimators' docstring for why.)"""
    pt = np.array([r['pt_mid'] for r in results])

    # small horizontal offsets so overlapping error bars stay legible
    n_methods_I = 4
    offsets_I = np.linspace(-0.6, 0.6, n_methods_I)
    n_methods_S = 3
    offsets_S = np.linspace(-0.45, 0.45, n_methods_S)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    for off, key, label, color, marker in zip(
        offsets_I,
        ['I_naive', 'I_mm', 'I_shuffle', 'I_nsb'],
        ['naive', 'Miller-Madow', 'shuffle', 'NSB'],
        ['#d62728', '#ff7f0e', '#1f77b4', '#9467bd'],
        ['o', 's', '^', 'D'],
    ):
        vals = np.array([r[key] for r in results])
        errs = np.array([r[key + '_err'] for r in results])
        ax.errorbar(pt + off, vals, yerr=errs, fmt=marker, color=color,
                    capsize=3, markersize=6, label=label)
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel(r'leading jet $p_T$ (GeV)')
    ax.set_ylabel(r'$I(A:B)$ (nats)')
    ax.set_title('Method comparison: MI vs. $p_T$ (bootstrap error bars)')
    ax.legend(fontsize=9)

    ax = axes[1]
    for off, key, label, color, marker in zip(
        offsets_S,
        ['SBA_naive', 'SBA_mm', 'SBA_nsb'],
        ['naive', 'Miller-Madow', 'NSB'],
        ['#d62728', '#ff7f0e', '#9467bd'],
        ['o', 's', 'D'],
    ):
        vals = np.array([r[key] for r in results])
        errs = np.array([r[key + '_err'] for r in results])
        ax.errorbar(pt + off, vals, yerr=errs, fmt=marker, color=color,
                    capsize=3, markersize=6, label=label)
    ax.axhline(0, color='red', ls=':', label='classical floor')
    ax.set_xlabel(r'leading jet $p_T$ (GeV)')
    ax.set_ylabel(r'$S(B|A)$ (nats)')
    ax.set_title('Method comparison: $S(B|A)$ vs. $p_T$ (bootstrap error bars)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_method_comparison_vs_pt.png", dpi=150)
    print(f"  saved {out_prefix}_method_comparison_vs_pt.png")


def plot_entropy_vs_pt(results, out_prefix):
    """ii) S_A, S_B, S_AB vs. leading jet pT."""
    pt = [r['pt_mid'] for r in results]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for key, label, color in [('S_A', r'$S_A$ (leading jet)', '#1f77b4'),
                                ('S_B', r'$S_B$ (subleading jet)', '#d62728'),
                                ('S_AB', r'$S_{AB}$ (joint)', '#2ca02c')]:
        vals = [r[key] for r in results]
        errs = [r[key + '_err'] for r in results]
        ax.errorbar(pt, vals, yerr=errs, fmt='o-', capsize=3, color=color, label=label)
    ax.set_xlabel(r'leading jet $p_T$ (GeV)')
    ax.set_ylabel('entropy (nats)')
    ax.set_title(r'NSB entropy vs. leading-jet $p_T$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_entropy_vs_pt.png", dpi=150)
    print(f"  saved {out_prefix}_entropy_vs_pt.png")


def plot_MI_vs_pt(results, out_prefix):
    """iii) I(A:B) vs. leading jet pT."""
    pt = [r['pt_mid'] for r in results]
    I_vals = [r['I'] for r in results]
    I_errs = [r['I_err'] for r in results]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(pt, I_vals, yerr=I_errs, fmt='o-', capsize=3, color='#9467bd')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel(r'leading jet $p_T$ (GeV)')
    ax.set_ylabel(r'$I(A:B)$ (nats), NSB')
    ax.set_title(r'Mutual information vs. leading-jet $p_T$')
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_MI_vs_pt.png", dpi=150)
    print(f"  saved {out_prefix}_MI_vs_pt.png")


def plot_SBA_vs_pt(results, out_prefix):
    """iv) S(B|A) vs. leading jet pT -- the quantum witness, differentially."""
    pt = [r['pt_mid'] for r in results]
    SBA_vals = [r['SBA'] for r in results]
    SBA_errs = [r['SBA_err'] for r in results]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(pt, SBA_vals, yerr=SBA_errs, fmt='o-', capsize=3, color='purple')
    ax.axhline(0, color='red', ls=':', label='classical floor')
    ax.set_xlabel(r'leading jet $p_T$ (GeV)')
    ax.set_ylabel(r'$S(B|A)$ (nats), NSB')
    ax.set_title(r'Conditional entropy vs. leading-jet $p_T$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_SBA_vs_pt.png", dpi=150)
    print(f"  saved {out_prefix}_SBA_vs_pt.png")


def print_pt_bin_table(results):
    print(f"\n{'pT range':>16} | {'N':>6} | {'S_A':>14} | {'S_B':>14} | "
          f"{'S_AB':>14} | {'I(A:B)':>14} | {'S(B|A)':>14}")
    print("-"*110)
    for r in results:
        print(f"[{r['pt_lo']:5.1f},{r['pt_hi']:5.1f}) | {r['N']:>6} | "
              f"{r['S_A']:>6.3f}+/-{r['S_A_err']:<5.3f} | "
              f"{r['S_B']:>6.3f}+/-{r['S_B_err']:<5.3f} | "
              f"{r['S_AB']:>6.3f}+/-{r['S_AB_err']:<5.3f} | "
              f"{r['I']:>6.3f}+/-{r['I_err']:<5.3f} | "
              f"{r['SBA']:>6.3f}+/-{r['SBA_err']:<5.3f}")


# =============================================================================
# Main analysis
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_file", help="PYTHIA8 dijet ROOT file (from main_dijet_pp200)")
    parser.add_argument("--level", choices=["truth", "det"], default="truth",
                         help="'truth' = wide/idealized acceptance, no tracking "
                              "inefficiency; 'det' = STAR-TPC-like acceptance "
                              "(pT>0.15 GeV, |eta|<1.0) + tracking efficiency, "
                              "automatically filtered to events where a "
                              "back-to-back dijet also survives detector-level "
                              "reconstruction (default: truth)")
    parser.add_argument("--alphabet", type=int, default=80,
                         help="Fixed upper bound on plausible charged multiplicity "
                              "per jet, used as the NSB alphabet size K_A=K_B "
                              "(fix BEFORE looking at the data; default 80, "
                              "generous for R=0.4 charged jets at these kinematics; "
                              "consider a smaller value for --level det, since "
                              "acceptance+efficiency lowers observed multiplicities)")
    parser.add_argument("--n-boot", type=int, default=200,
                         help="bootstrap resamples for the real-data NSB noise floor")
    parser.add_argument("--n-boot-compare", type=int, default=50,
                         help="bootstrap resamples PER pT BIN PER METHOD for the "
                              "multi-method comparison plots (iii) and the summary "
                              "bar chart's error bars (default 50 -- kept smaller "
                              "than --n-boot since this cost multiplies by number "
                              "of pT bins and methods)")
    parser.add_argument("--n-shuffles-inner", type=int, default=30,
                         help="shuffles per bootstrap replicate when computing the "
                              "shuffle-corrected estimator inside bootstrap_all_estimators "
                              "(default 30; the outer bootstrap loop is what drives "
                              "precision here, not this inner value)")
    parser.add_argument("--skip-method-comparison", action="store_true",
                         help="skip the multi-method vs. pT comparison plots (iii) "
                              "-- these are the most expensive part of the analysis "
                              "(n_pt_bins x n_boot_compare x 4 methods)")
    parser.add_argument("--out-prefix", default=None,
                         help="prefix for output plot files "
                              "(default: 'dijet_entropy_result_<level>')")
    parser.add_argument("--pt-bins", type=int, default=4,
                         help="number of leading-jet-pT bins for the differential "
                              "S_A/S_B/S_AB, I(A:B), S(B|A) vs. pT plots (default 4)")
    parser.add_argument("--pt-bin-mode", choices=["quantile", "fixed"], default="quantile",
                         help="quantile (equal-count, recommended for steeply falling "
                              "spectra) or fixed-width pT bins (default quantile)")
    parser.add_argument("--min-bin-events", type=int, default=30,
                         help="warn if a pT bin has fewer than this many events "
                              "(default 30 -- still likely too few for a solid NSB "
                              "estimate; treat as a lower alarm threshold, not a "
                              "green light)")
    parser.add_argument("--skip-pt-differential", action="store_true",
                         help="skip the pT-binned analysis (e.g. for very small samples)")
    args = parser.parse_args()

    if args.out_prefix is None:
        args.out_prefix = f"dijet_entropy_result_{args.level}"

    os.makedirs("figs", exist_ok=True)
    args.out_prefix = os.path.join("figs", args.out_prefix)

    rng = np.random.default_rng(12345)

    print(f"Loading {args.root_file} (level='{args.level}') ...")
    data = load_dijet_tree(args.root_file, level=args.level)
    n_a = data["jet1_mult"].astype(int)
    n_b = data["jet2_mult"].astype(int)
    jet1_pt = data["jet1_pt"].astype(float)
    jet2_pt = data["jet2_pt"].astype(float)
    N = len(n_a)
    print(f"Loaded {N} back-to-back dijet events.")
    print(f"  <N_A> = {n_a.mean():.2f}   <N_B> = {n_b.mean():.2f}   "
          f"corr(N_A,N_B) = {np.corrcoef(n_a, n_b)[0,1]:.3f}")
    print(f"  max observed multiplicity: A={n_a.max()}  B={n_b.max()}  "
          f"(alphabet K={args.alphabet} chosen ahead of time)")

    if n_a.max() >= args.alphabet or n_b.max() >= args.alphabet:
        print("  WARNING: observed multiplicity reaches the chosen alphabet "
              "bound -- increase --alphabet and rerun.")

    ALPHABET_A = ALPHABET_B = args.alphabet

    # ---- point estimates, all four methods ----
    print("\n--- Mutual information I(A:B) ---")
    I_naive, info = naive_MI(n_a, n_b)
    I_mm, _, mm_bias = miller_madow_MI(n_a, n_b)
    I_sh, _, sh_bias, sh_err = shuffle_correction_MI(n_a, n_b, n_shuffles=200, rng=rng)
    I_nsb, nsb_info = nsb_MI(n_a, n_b, ALPHABET_A, ALPHABET_B)

    print(f"  K_A={info['K_A']}  K_B={info['K_B']}  K_AB={info['K_AB']}  N={info['N']}")
    print(f"  naive          : {I_naive:.4f} nats")
    print(f"  Miller-Madow   : {I_mm:.4f} nats  (bias correction: {mm_bias:.4f})")
    print(f"  shuffle-corr.  : {I_sh:.4f} nats  (bias estimate: {sh_bias:.4f} +/- {sh_err:.4f})")
    print(f"  NSB            : {I_nsb:.4f} nats   <-- recommended")

    # ---- conditional entropy S(B|A): the quantum witness ----
    print("\n--- Conditional entropy S(B|A) = S_AB - S_A (quantum witness) ---")
    SBA_nsb, sigma_analytic = sba_nsb_with_analytic_uncertainty(
        n_a, n_b, ALPHABET_A, ALPHABET_B)
    SBA_boot_mean, sigma_boot = sba_bootstrap_uncertainty(
        n_a, n_b, ALPHABET_A, ALPHABET_B, n_boot=args.n_boot, rng=rng)

    print(f"  S(B|A)_NSB              = {SBA_nsb:.4f} nats")
    print(f"  sigma (analytic, single dataset) = {sigma_analytic:.4f} nats")
    print(f"  sigma (bootstrap, {args.n_boot} resamples)  = {sigma_boot:.4f} nats")

    z_score = SBA_nsb / sigma_boot if sigma_boot > 0 else np.nan
    print(f"\n  S(B|A) is {z_score:.2f} sigma from zero (bootstrap sigma).")
    if SBA_nsb < 0:
        print(f"  NEGATIVE conditional entropy -- classically impossible; "
              f"significance = {abs(z_score):.2f} sigma. Check both the "
              f"analytic and bootstrap sigma agree before trusting this, "
              f"and check S(A|B) as well.")
    else:
        print("  Positive, consistent with a classical description at "
              "this precision. (Consistent with a classical description "
              "is not the same as proof of no entanglement -- see the "
              "toy-study notebook's caveats.)")

    # Also compute S(A|B) for completeness (conditional entropy is not symmetric).
    SAB_nsb_other, sigma_analytic_other = sba_nsb_with_analytic_uncertainty(
        n_b, n_a, ALPHABET_B, ALPHABET_A)
    print(f"\n  S(A|B)_NSB = {SAB_nsb_other:.4f} nats "
          f"(analytic sigma = {sigma_analytic_other:.4f})")

    # ---- bias-vs-subsample-size diagnostic on the real data ----
    print("\n--- Bias-vs-statistics diagnostic (subsampling the real dataset) ---")
    N_grid = np.array(sorted(set(
        int(x) for x in np.geomspace(200, N, 8).astype(int)
    )))
    I_means, I_stds = bias_vs_subsample_scan(n_a, n_b, N_grid, n_rep=15, rng=rng)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(N_grid, I_means, yerr=I_stds, fmt='o-', capsize=3,
                label=r'naive $\hat I(N)$, subsampled from real data')
    ax.axhline(I_nsb, color='purple', ls='--', label=f'NSB (full sample): {I_nsb:.3f}')
    ax.set_xscale('log')
    ax.set_xlabel('N (subsample size)')
    ax.set_ylabel('Mutual information (nats)')
    ax.set_title('Naive MI bias vs. statistics -- real PYTHIA8 dijet sample')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_bias_scan.png", dpi=150)
    print(f"  saved {args.out_prefix}_bias_scan.png")

    # ---- summary plot: all four MI estimates + S(B|A) with uncertainty ----
    # (ii) error bars added here via the unified whole-sample bootstrap --
    # previously this bar chart had no uncertainty at all.
    print("\n--- Bootstrapping all estimators on the full sample (for error bars) ---")
    full_boot = bootstrap_all_estimators(n_a, n_b, ALPHABET_A, ALPHABET_B,
                                          n_boot=args.n_boot_compare,
                                          n_shuffles_inner=args.n_shuffles_inner, rng=rng)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    methods = ['naive', 'Miller-Madow', 'shuffle', 'NSB']
    values = [I_naive, I_mm, I_sh, I_nsb]
    errors = [full_boot['I_naive'][1], full_boot['I_mm'][1],
              full_boot['I_shuffle'][1], full_boot['I_nsb'][1]]
    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#9467bd']
    ax.bar(methods, values, yerr=errors, capsize=5, color=colors)
    ax.set_ylabel('I(A:B) (nats)')
    ax.set_title('MI estimators, this dataset\n(error bars: bootstrap, N_boot='
                  f'{args.n_boot_compare})')
    ax.tick_params(axis='x', rotation=20)

    ax = axes[1]
    ax.errorbar([0], [SBA_nsb], yerr=[sigma_boot], fmt='o', capsize=5,
                color='purple', markersize=10, label='S(B|A), NSB ± bootstrap')
    ax.axhline(0, color='red', ls=':', label='classical floor')
    ax.set_xlim(-1, 1)
    ax.set_xticks([])
    ax.set_ylabel('S(B|A) (nats)')
    ax.set_title('Conditional entropy (quantum witness)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_summary.png", dpi=150)
    print(f"  saved {args.out_prefix}_summary.png")

    # ---- i) multiplicity distribution shape check (NBD assumption) ----
    print("\n--- i) Multiplicity distribution shape (vs. NBD assumption) ---")
    plot_multiplicity_distributions(n_a, n_b, args.out_prefix)
    plot_joint_multiplicity_distribution(n_a, n_b, args.out_prefix)

    # ---- i) jet pT distributions (new) ----
    print("\n--- i) Jet pT distributions ---")
    plot_pt_distributions(jet1_pt, jet2_pt, args.out_prefix)
    plot_joint_pt_distribution(jet1_pt, jet2_pt, args.out_prefix)

    # ---- ii)-iv) differential vs. leading-jet pT ----
    if args.skip_pt_differential:
        print("\n(Skipping pT-differential analysis: --skip-pt-differential)")
    else:
        print(f"\n--- ii)-iv) Differential vs. leading-jet pT "
              f"({args.pt_bins} {args.pt_bin_mode} bins) ---")
        compare = not args.skip_method_comparison
        if compare:
            print(f"  (multi-method comparison enabled: {args.pt_bins} bins x "
                  f"{args.n_boot_compare} bootstrap resamples x 4 methods -- "
                  f"this is the slowest part of the analysis)")
        pt_results = pt_differential_analysis(
            n_a, n_b, jet1_pt, ALPHABET_A, ALPHABET_B,
            n_bins=args.pt_bins, mode=args.pt_bin_mode,
            min_events=args.min_bin_events,
            compare_methods=compare, n_boot_compare=args.n_boot_compare,
            n_shuffles_inner=args.n_shuffles_inner, rng=rng,
        )
        if len(pt_results) == 0:
            print("  No pT bins had enough events to analyze -- sample too "
                  "small for a differential measurement yet. Skipping "
                  "pT-differential plots.")
        else:
            print_pt_bin_table(pt_results)
            plot_entropy_vs_pt(pt_results, args.out_prefix)
            plot_MI_vs_pt(pt_results, args.out_prefix)
            plot_SBA_vs_pt(pt_results, args.out_prefix)
            if compare:
                plot_method_comparison_vs_pt(pt_results, args.out_prefix)

    print("\nDone.")


if __name__ == "__main__":
    main()
