#!/usr/bin/env python3
"""
analyze_dijet_mixed.py

Mixed-event subtraction of the TRIVIAL KINEMATIC contribution to the
dijet multiplicity mutual information. Separate script from
analyze_dijet_entropy.py on purpose -- that one measures raw
(pT-inclusive or pT-binned) MI; this one measures the
beyond-kinematics ("signal") MI:

    I_signal = I_corrected(same-event pairs) - I_corrected(mixed pairs)

Why: N_A and N_B each depend on their jet's pT, and pT1/pT2 are strongly
correlated (momentum balance), so integrating over ANY finite pT window
induces real but kinematically-trivial MI (common-cause structure). The
mixed sample keeps event i's jet A intact and takes jet B from a
different event j with pT2_j ~= pT2_i (k-nearest-neighbor matching in
pT2), reproducing the same kinematic structure under conditional
independence -- its MI IS the trivial part.

Composition rule (order matters -- see analysis notes): bias correction
is applied PER TERM (each of I_same and I_mix is a finite-sample
estimate with its own estimator bias), and the kinematic subtraction is
taken BETWEEN the two corrected terms. Never bias-correct the
already-subtracted difference -- that double-corrects, because the
leading naive bias ~(K_AB-K_A-K_B+1)/(2N) largely CANCELS between the
two terms already (same marginals by construction, same N). The QA plot
here shows this near-cancellation directly: naive-minus-naive lands
close to NSB-minus-NSB even though the raw naive terms are visibly
biased.

Uncertainties: JOINT bootstrap of the full difference -- each resample
rebuilds both the same-event pairs and the mixed pairs (re-matched
within the resample) and recomputes the difference for all four methods.
Never propagate the two terms' errors separately (they share events and
are strongly correlated).

Usage:
    python3 analyze_dijet_mixed.py file.root --level truth --alphabet 80
    python3 analyze_dijet_mixed.py file.root --level truth --alphabet 80 \
        --mix-neighbors 10 --n-boot 50 --pt-bins 4
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_dijet_entropy import (
    load_dijet_tree, make_pt_bins, entropies_from_samples,
    shuffle_correction_MI, nsb_entropy_from_counts,
)


# ---------------------------------------------------------------------------
# Mixed-pair construction: k-nearest-neighbor matching in pT2
# ---------------------------------------------------------------------------

def build_mixed_partner_idx(pt2, k_neighbors, rng):
    """For each event i, pick a partner j != i whose pT2 is among the
    k nearest to pT2_i (in sorted order). Returns idx array such that
    the mixed pairs are (n_a[i], n_b[idx[i]]). Also returns the mean
    absolute pT2 mismatch actually achieved -- the matching-tolerance
    systematic knob."""
    n = len(pt2)
    order = np.argsort(pt2, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    # random neighbor offset in sorted order, in [-k, +k] \ {0}
    off = rng.integers(1, k_neighbors + 1, size=n) * rng.choice([-1, 1], size=n)
    partner_rank = np.clip(rank + off, 0, n - 1)
    # if clipping landed us on ourselves (at the edges), step inward
    same = partner_rank == rank
    partner_rank[same] = np.clip(rank[same] + np.where(rank[same] < n // 2, 1, -1), 0, n - 1)
    idx = order[partner_rank]
    mean_dpt = float(np.mean(np.abs(pt2 - pt2[idx])))
    return idx, mean_dpt


# ---------------------------------------------------------------------------
# The four point estimators, applied to one pair sample (no bootstrap here)
# ---------------------------------------------------------------------------

def nsb_point_I(n_a, n_b, alphabet_a, alphabet_b):
    va, ca = np.unique(n_a, return_counts=True)
    vb, cb = np.unique(n_b, return_counts=True)
    pairs = np.stack([n_a, n_b], axis=1)
    _, cab = np.unique(pairs, axis=0, return_counts=True)
    S_A = nsb_entropy_from_counts(ca, alphabet_a)
    S_B = nsb_entropy_from_counts(cb, alphabet_b)
    S_AB = nsb_entropy_from_counts(cab, alphabet_a * alphabet_b)
    return S_A + S_B - S_AB


def all_point_estimates(n_a, n_b, alphabet, n_shuffles, rng):
    """Point estimates of I for naive / Miller-Madow / shuffle / NSB on
    ONE sample of pairs. Returned as a dict."""
    S_A, S_B, S_AB, K_A, K_B, K_AB, N = entropies_from_samples(n_a, n_b)
    I_naive = S_A + S_B - S_AB
    I_mm = I_naive - (K_AB - K_A - K_B + 1) / (2.0 * N)
    I_sh, _, _, _ = shuffle_correction_MI(n_a, n_b, n_shuffles=n_shuffles, rng=rng)
    I_nsb = nsb_point_I(n_a, n_b, alphabet, alphabet)
    return dict(naive=I_naive, mm=I_mm, shuffle=I_sh, nsb=I_nsb)


METHODS = ["naive", "mm", "shuffle", "nsb"]
LABELS = {"naive": "naive", "mm": "Miller-Madow", "shuffle": "shuffle", "nsb": "NSB"}
COLORS = {"naive": "#d62728", "mm": "#ff7f0e", "shuffle": "#1f77b4", "nsb": "#9467bd"}


def same_vs_mixed(n_a, n_b, pt2, alphabet, k_neighbors, n_shuffles,
                   n_boot, rng, tag=""):
    """Full same-vs-mixed analysis on one event set: point estimates for
    both samples and all four methods, plus JOINT bootstrap of the
    difference (re-matching within each resample)."""
    idx_mix, mean_dpt = build_mixed_partner_idx(pt2, k_neighbors, rng)
    same = all_point_estimates(n_a, n_b, alphabet, n_shuffles, rng)
    mix = all_point_estimates(n_a, n_b[idx_mix], alphabet, n_shuffles, rng)
    diff = {m: same[m] - mix[m] for m in METHODS}

    # ---- joint bootstrap of the difference ----
    N = len(n_a)
    boot = {m: np.empty(n_boot) for m in METHODS}
    for b in range(n_boot):
        sel = rng.integers(0, N, size=N)
        ra, rb, rpt2 = n_a[sel], n_b[sel], pt2[sel]
        ridx, _ = build_mixed_partner_idx(rpt2, k_neighbors, rng)
        s = all_point_estimates(ra, rb, alphabet, n_shuffles, rng)
        x = all_point_estimates(ra, rb[ridx], alphabet, n_shuffles, rng)
        for m in METHODS:
            boot[m][b] = s[m] - x[m]
    diff_err = {m: boot[m].std(ddof=1) for m in METHODS}

    print(f"\n--- same vs. mixed {tag} (N={N}, k_neighbors={k_neighbors}, "
          f"mean |dpT2| matched = {mean_dpt:.2f} GeV) ---")
    print(f"{'method':>14} | {'I_same':>9} | {'I_mix':>9} | {'I_same - I_mix':>16}")
    print("-" * 60)
    for m in METHODS:
        print(f"{LABELS[m]:>14} | {same[m]:>9.5f} | {mix[m]:>9.5f} | "
              f"{diff[m]:>+9.5f} +/- {diff_err[m]:.5f}")
    return dict(same=same, mix=mix, diff=diff, diff_err=diff_err,
                mean_dpt=mean_dpt, N=N)


# ---------------------------------------------------------------------------
# QA plots
# ---------------------------------------------------------------------------

def plot_whole_sample_qa(res, out_prefix):
    """Two panels: (left) I_same vs I_mix per method -- shows the trivial
    kinematic MI is the bulk of the raw signal AND that both terms carry
    similar estimator bias; (right) the per-method DIFFERENCES with
    jointly-bootstrapped errors -- the four-way triangulation. Near-
    agreement of naive-minus-naive with NSB-minus-NSB is the bias
    near-cancellation, demonstrated rather than assumed."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(METHODS))
    w = 0.38
    ax = axes[0]
    ax.bar(x - w/2, [res['same'][m] for m in METHODS], w,
           color=[COLORS[m] for m in METHODS], label="same-event")
    ax.bar(x + w/2, [res['mix'][m] for m in METHODS], w,
           color=[COLORS[m] for m in METHODS], alpha=0.45, hatch='//',
           label="mixed (pT2-matched)")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=15)
    ax.set_ylabel("I(A:B) (nats)")
    ax.set_title(f"Same-event vs. mixed-event MI, all methods\n"
                  f"(mixed = trivial kinematic part; matched to "
                  f"|dpT2|~{res['mean_dpt']:.1f} GeV)")
    ax.legend(fontsize=9)

    ax = axes[1]
    vals = [res['diff'][m] for m in METHODS]
    errs = [res['diff_err'][m] for m in METHODS]
    ax.bar([LABELS[m] for m in METHODS], vals, yerr=errs, capsize=5,
           color=[COLORS[m] for m in METHODS])
    ax.axhline(0, color='gray', lw=1)
    ax.set_ylabel(r"$I_{\rm same} - I_{\rm mix}$ (nats)")
    ax.set_title("Beyond-kinematics MI, per method\n"
                  "(errors: joint bootstrap of the difference; agreement of\n"
                  "naive with NSB here = the bias near-cancellation)")
    ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_mixed_qa.png", dpi=150)
    print(f"  saved {out_prefix}_mixed_qa.png")


def plot_diff_vs_pt(bin_results, out_prefix):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    pts = np.array([r['pt_mid'] for r in bin_results])
    offs = np.linspace(-0.35, 0.35, len(METHODS))
    for off, m, mk in zip(offs, METHODS, ['o', 's', '^', 'D']):
        vals = [r['res']['diff'][m] for r in bin_results]
        errs = [r['res']['diff_err'][m] for r in bin_results]
        ax.errorbar(pts + off, vals, yerr=errs, fmt=mk, color=COLORS[m],
                    capsize=3, markersize=6, label=LABELS[m])
    ax.axhline(0, color='gray', lw=1)
    ax.set_xlabel(r"leading jet $p_T$ (GeV)")
    ax.set_ylabel(r"$I_{\rm same} - I_{\rm mix}$ (nats)")
    ax.set_title("Beyond-kinematics MI vs. $p_T$ (mixed-event subtracted)")
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_mixed_diff_vs_pt.png", dpi=150)
    print(f"  saved {out_prefix}_mixed_diff_vs_pt.png")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root_file")
    p.add_argument("--level", choices=["truth", "det"], default="truth")
    p.add_argument("--alphabet", type=int, default=80)
    p.add_argument("--mix-neighbors", type=int, default=10,
                    help="k for k-nearest-neighbor pT2 matching (default 10). "
                         "Vary this and check stability -- residual within-"
                         "window pT variation is the subtraction systematic.")
    p.add_argument("--n-boot", type=int, default=50,
                    help="joint bootstrap resamples for the difference (default 50)")
    p.add_argument("--n-shuffles", type=int, default=30)
    p.add_argument("--pt-bins", type=int, default=4,
                    help="quantile bins in leading-jet pT for the differential "
                         "version (0 = whole-sample only)")
    p.add_argument("--min-bin-events", type=int, default=500)
    p.add_argument("--figs-dir", default="figs")
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args()

    if args.out_prefix is None:
        args.out_prefix = f"dijet_mixed_{args.level}"
    os.makedirs(args.figs_dir, exist_ok=True)
    args.out_prefix = os.path.join(args.figs_dir, args.out_prefix)
    print(f"Figures will be saved under: {args.figs_dir}")

    rng = np.random.default_rng(20260722)
    data = load_dijet_tree(args.root_file, level=args.level)
    n_a = data["jet1_mult"].astype(int)
    n_b = data["jet2_mult"].astype(int)
    pt1 = data["jet1_pt"].astype(float)
    pt2 = data["jet2_pt"].astype(float)
    print(f"Loaded {len(n_a)} events.")

    # ---- whole sample ----
    res = same_vs_mixed(n_a, n_b, pt2, args.alphabet, args.mix_neighbors,
                         args.n_shuffles, args.n_boot, rng, tag="(whole sample)")
    plot_whole_sample_qa(res, args.out_prefix)

    # ---- pT-differential ----
    if args.pt_bins > 0:
        edges = make_pt_bins(pt1, args.pt_bins, 'quantile')
        bin_results = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i+1]
            mask = (pt1 >= lo) & (pt1 < hi) if i < len(edges)-2 else \
                   (pt1 >= lo) & (pt1 <= hi)
            if mask.sum() < args.min_bin_events:
                print(f"  skipping bin [{lo:.1f},{hi:.1f}): only {mask.sum()} events")
                continue
            r = same_vs_mixed(n_a[mask], n_b[mask], pt2[mask], args.alphabet,
                               args.mix_neighbors, args.n_shuffles, args.n_boot,
                               rng, tag=f"[{lo:.1f},{hi:.1f}) GeV")
            bin_results.append(dict(pt_mid=float(pt1[mask].mean()), res=r))
        if bin_results:
            plot_diff_vs_pt(bin_results, args.out_prefix)

    print("\nDone.")


if __name__ == "__main__":
    main()
