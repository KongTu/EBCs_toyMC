// analyze_dijet_entropy.C
//
// ROOT-native implementation of the same entropy / mutual-information /
// conditional-entropy analysis as analyze_dijet_entropy.py -- an
// independent cross-check using a completely separate numerical stack
// (ROOT/C++ instead of Python/scipy).
//
// All formulas (naive plug-in, Miller-Madow, shuffle-subtraction, NSB
// mean + posterior variance) are the same ones derived and validated
// throughout the toy-Monte-Carlo study. The one new numerical ingredient
// needed here -- digamma/trigamma, which scipy.special provided for free
// in Python -- is implemented below via the standard recurrence +
// Bernoulli-series asymptotic expansion (Abramowitz & Stegun 6.4.11/
// 6.4.12), and was validated against scipy.special.digamma/polygamma
// across x in [1e-6, 1000] to ~1e-10 (absolute) / ~1e-16 (relative)
// before being translated here -- see the chat history / companion note
// for that validation. No external dependency (GSL/MathMore) required.
//
// Usage (ACLiC-compiled -- strongly recommended for speed, since the
// NSB beta-integral + bootstrap loops are numerically heavy):
//
//   root -l -q 'analyze_dijet_entropy.C+("pythia_dijet_pp200.root","truth",80,200,4)'
//   root -l -q 'analyze_dijet_entropy.C+("pythia_dijet_pp200.root","det",60,200,4)'
//
// Arguments: (rootFile, level["truth"|"det"], alphabet, nBoot, ptBins)
//
// I have not been able to compile/run this against actual ROOT in the
// environment this was written in (no ROOT installed there) -- the
// special-function core was validated numerically in Python first (see
// above), and the ROOT/TTree/TH1/TGraph plumbing follows standard, common
// ROOT idioms, but a first compile pass on your end catching any
// version-specific API issues is expected. Send me the compiler output
// if anything doesn't build and I'll help fix it.

#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TProfile.h>
#include <TCanvas.h>
#include <TGraphErrors.h>
#include <TMultiGraph.h>
#include <TLegend.h>
#include <TLine.h>
#include <TRandom3.h>
#include <TMath.h>
#include <TString.h>
#include <TSystem.h>
#include <TStyle.h>

#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

// =============================================================================
// Special functions: digamma (psi0) and trigamma (psi1), self-contained.
// Validated against scipy.special to ~1e-10 absolute / ~1e-16 relative
// across x in [1e-6, 1000] before being used here.
// =============================================================================

double digamma(double x) {
    double result = 0.0;
    while (x < 6.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    double f = 1.0 / (x * x);
    result += std::log(x) - 0.5 / x
        - f * (1.0/12.0 - f * (1.0/120.0 - f * (1.0/252.0 - f * (1.0/240.0 - f * (1.0/132.0)))));
    return result;
}

double trigamma(double x) {
    double result = 0.0;
    while (x < 6.0) {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    double ix2 = 1.0 / (x * x);
    result += 1.0/x + 0.5*ix2 + ix2/x * (1.0/6.0 - ix2 * (1.0/30.0 - ix2 * (1.0/42.0 - ix2/30.0)));
    return result;
}

// =============================================================================
// Naive plug-in entropy / MI, from a vector of nonzero occupied-bin counts.
// =============================================================================

double shannonEntropy(const std::vector<int>& counts) {
    double N = 0;
    for (int c : counts) N += c;
    double S = 0.0;
    for (int c : counts) {
        if (c <= 0) continue;
        double p = c / N;
        S -= p * std::log(p);
    }
    return S;
}

// Build occupied-bin count vectors for the marginal(s) and joint distribution
// from raw (nA, nB) integer arrays.
struct Histograms {
    std::vector<int> countsA, countsB, countsAB;
    int K_A, K_B, K_AB;
    long long N;
};

Histograms buildHistograms(const std::vector<int>& nA, const std::vector<int>& nB) {
    std::map<int,int> mapA, mapB;
    std::map<std::pair<int,int>,int> mapAB;
    for (size_t i = 0; i < nA.size(); ++i) {
        mapA[nA[i]]++;
        mapB[nB[i]]++;
        mapAB[{nA[i], nB[i]}]++;
    }
    Histograms h;
    for (auto& kv : mapA) h.countsA.push_back(kv.second);
    for (auto& kv : mapB) h.countsB.push_back(kv.second);
    for (auto& kv : mapAB) h.countsAB.push_back(kv.second);
    h.K_A = h.countsA.size();
    h.K_B = h.countsB.size();
    h.K_AB = h.countsAB.size();
    h.N = nA.size();
    return h;
}

struct NaiveResult {
    double S_A, S_B, S_AB, I;
    int K_A, K_B, K_AB;
    long long N;
};

NaiveResult naiveMI(const std::vector<int>& nA, const std::vector<int>& nB) {
    Histograms h = buildHistograms(nA, nB);
    NaiveResult r;
    r.S_A = shannonEntropy(h.countsA);
    r.S_B = shannonEntropy(h.countsB);
    r.S_AB = shannonEntropy(h.countsAB);
    r.I = r.S_A + r.S_B - r.S_AB;
    r.K_A = h.K_A; r.K_B = h.K_B; r.K_AB = h.K_AB; r.N = h.N;
    return r;
}

// Exact leading-order Miller-Madow correction (uses OBSERVED K_AB, not K_A*K_B).
double millerMadowMI(const NaiveResult& r, double& biasOut) {
    biasOut = (r.K_AB - r.K_A - r.K_B + 1) / (2.0 * r.N);
    return r.I - biasOut;
}

// Shuffle-subtraction: permute B labels relative to A, many times, average.
double shuffleCorrectionMI(const std::vector<int>& nA, const std::vector<int>& nB,
                            int nShuffles, TRandom3& rng,
                            double& biasOut, double& biasErrOut) {
    NaiveResult naive = naiveMI(nA, nB);
    std::vector<int> nBshuffled = nB;
    std::vector<double> Ishuf(nShuffles);
    for (int s = 0; s < nShuffles; ++s) {
        // Fisher-Yates shuffle
        for (int i = (int)nBshuffled.size() - 1; i > 0; --i) {
            int j = rng.Integer(i + 1);
            std::swap(nBshuffled[i], nBshuffled[j]);
        }
        NaiveResult rs = naiveMI(nA, nBshuffled);
        Ishuf[s] = rs.I;
    }
    double mean = 0; for (double v : Ishuf) mean += v; mean /= nShuffles;
    double var = 0; for (double v : Ishuf) var += (v-mean)*(v-mean); var /= (nShuffles-1);
    biasOut = mean;
    biasErrOut = std::sqrt(var / nShuffles);
    return naive.I - mean;
}

// =============================================================================
// NSB: posterior mean AND variance of entropy, given occupied-bin counts and
// total alphabet size K. See toy-study notebook / companion note for the
// full derivation; validated there against brute-force Dirichlet sampling.
// =============================================================================

struct NSBResult { double mean; double var; };

std::vector<double> logspace(double lo, double hi, int n) {
    std::vector<double> v(n);
    double step = (hi - lo) / (n - 1);
    for (int i = 0; i < n; ++i) v[i] = std::pow(10.0, lo + i*step);
    return v;
}

NSBResult nsbEntropyAndVariance(const std::vector<int>& countsNonzero, long long K) {
    long long N = 0;
    for (int c : countsNonzero) N += c;
    long long K_obs = (long long)countsNonzero.size();
    long long K_zero = K - K_obs;
    if (K_zero < 0) {
        std::cerr << "ERROR: alphabet K=" << K << " smaller than observed occupied bins "
                  << K_obs << " -- increase alphabet.\n";
        return {0.0, 0.0};
    }

    static std::vector<double> betaGrid = logspace(-6.0, 3.0, 300);
    int nb = betaGrid.size();

    std::vector<double> means(nb), varis(nb), logP(nb), rho(nb);

    for (int ib = 0; ib < nb; ++ib) {
        double beta = betaGrid[ib];
        double A = N + K * beta;

        // ---- mean entropy given beta ----
        double sumTerm = 0.0;
        for (int c : countsNonzero) {
            double a = c + beta;
            sumTerm += a * digamma(a + 1.0);
        }
        sumTerm += K_zero * beta * digamma(beta + 1.0);
        double mean = digamma(A + 1.0) - sumTerm / A;

        // ---- variance given beta (Wolpert-Wolf 1994) ----
        double Ap2 = A + 2.0;
        double term1 = 0.0;
        double sum_a2 = 0.0, sum_ad = 0.0, sum_a2d2 = 0.0;

        auto accumulate = [&](double a, double weight) {
            double d = digamma(a + 1.0) - digamma(Ap2);
            term1 += weight * (a*(a+1.0)/(A*(A+1.0))) *
                     (trigamma(a+2.0) - trigamma(Ap2) + (digamma(a+2.0)-digamma(Ap2))*(digamma(a+2.0)-digamma(Ap2)));
            sum_a2  += weight * a*a;
            sum_ad  += weight * a*d;
            sum_a2d2+= weight * a*a*d*d;
        };
        for (int c : countsNonzero) accumulate(c + beta, 1.0);
        accumulate(beta, (double)K_zero);

        double term2 = (1.0/(A*(A+1.0))) * ( -trigamma(Ap2)*(A*A - sum_a2) + sum_ad*sum_ad - sum_a2d2 );

        double var = (term1 + term2) - mean*mean;

        means[ib] = mean;
        varis[ib] = var;

        // ---- log Dirichlet-multinomial evidence P(n|beta) ----
        double lp = TMath::LnGamma(K*beta) - TMath::LnGamma(N + K*beta) + TMath::LnGamma(N + 1.0);
        for (int c : countsNonzero) lp += TMath::LnGamma(c + beta) - TMath::LnGamma(beta);
        // (zero-count bins contribute LnGamma(beta)-LnGamma(beta)=0, omitted)
        for (int c : countsNonzero) lp -= TMath::LnGamma(c + 1.0);
        logP[ib] = lp;

        // ---- NSB hyperprior ----
        rho[ib] = K * trigamma(K*beta + 1.0) - trigamma(beta + 1.0);
    }

    // ---- combine: weight by rho*P(n|beta), integrate over log(beta) ----
    double maxLogW = -1e300;
    std::vector<double> logw(nb);
    for (int ib = 0; ib < nb; ++ib) {
        logw[ib] = logP[ib] + std::log(rho[ib]);
        if (logw[ib] > maxLogW) maxLogW = logw[ib];
    }
    std::vector<double> w(nb), t(nb), weightT(nb);
    for (int ib = 0; ib < nb; ++ib) {
        w[ib] = std::exp(logw[ib] - maxLogW);
        t[ib] = std::log(betaGrid[ib]);
        weightT[ib] = w[ib] * betaGrid[ib];
    }

    auto trapz = [&](const std::vector<double>& y) {
        double s = 0.0;
        for (int i = 0; i < nb - 1; ++i)
            s += 0.5 * (y[i] + y[i+1]) * (t[i+1] - t[i]);
        return s;
    };

    double Z = trapz(weightT);

    std::vector<double> wtMean(nb), wtVar(nb);
    for (int ib = 0; ib < nb; ++ib) wtMean[ib] = weightT[ib] * means[ib];
    double S_nsb = trapz(wtMean) / Z;

    for (int ib = 0; ib < nb; ++ib) wtVar[ib] = weightT[ib] * varis[ib];
    double E_var_given_beta = trapz(wtVar) / Z;

    std::vector<double> wtVarOfMean(nb);
    for (int ib = 0; ib < nb; ++ib) {
        double d = means[ib] - S_nsb;
        wtVarOfMean[ib] = weightT[ib] * d * d;
    }
    double var_of_mean_given_beta = trapz(wtVarOfMean) / Z;

    NSBResult out;
    out.mean = S_nsb;
    out.var = E_var_given_beta + var_of_mean_given_beta;
    return out;
}

struct NSBmiResult { double S_A, S_B, S_AB, I, S_A_err, S_B_err, S_AB_err, I_err, NMI, NMI_err; };

NSBmiResult nsbMI(const std::vector<int>& nA, const std::vector<int>& nB,
                   long long alphabetA, long long alphabetB) {
    Histograms h = buildHistograms(nA, nB);
    long long K_joint = alphabetA * alphabetB;

    NSBResult a = nsbEntropyAndVariance(h.countsA, alphabetA);
    NSBResult b = nsbEntropyAndVariance(h.countsB, alphabetB);
    NSBResult ab = nsbEntropyAndVariance(h.countsAB, K_joint);

    NSBmiResult r;
    r.S_A = a.mean; r.S_A_err = std::sqrt(a.var);
    r.S_B = b.mean; r.S_B_err = std::sqrt(b.var);
    r.S_AB = ab.mean; r.S_AB_err = std::sqrt(ab.var);
    r.I = a.mean + b.mean - ab.mean;
    // independence approximation (conservative; see toy-study caveats)
    r.I_err = std::sqrt(a.var + b.var + ab.var);

    // NMI = I / min(S_A,S_B): scale-comparable version of MI (classically
    // in [0,1]; NMI>1 is the same classical-bound violation as
    // S(B|A)<0). NMI_err via standard ratio error propagation (same
    // independence-approximation spirit as I_err above): treats I and
    // min(S_A,S_B) as uncorrelated, using whichever of S_A_err/S_B_err
    // corresponds to the min.
    double minS = std::min(r.S_A, r.S_B);
    double minS_err = (r.S_A < r.S_B) ? r.S_A_err : r.S_B_err;
    r.NMI = r.I / minS;
    r.NMI_err = std::sqrt((r.I_err/minS)*(r.I_err/minS)
                           + (r.I*minS_err/(minS*minS))*(r.I*minS_err/(minS*minS)));
    return r;
}

// S(B|A) = S_AB - S_A, with analytic (independence-approx) sigma.
void sbaNsb(const std::vector<int>& nA, const std::vector<int>& nB,
            long long alphabetA, long long alphabetB,
            double& SBA, double& sigmaAnalytic) {
    Histograms h = buildHistograms(nA, nB);
    long long K_joint = alphabetA * alphabetB;
    NSBResult a = nsbEntropyAndVariance(h.countsA, alphabetA);
    NSBResult ab = nsbEntropyAndVariance(h.countsAB, K_joint);
    SBA = ab.mean - a.mean;
    sigmaAnalytic = std::sqrt(ab.var + a.var);
}

// Bootstrap resampling of the actual dataset (with replacement) for a
// real-data noise floor on S(B|A) -- the real-data analogue of the toy
// study's Monte-Carlo-repetition noise floor.
void sbaBootstrap(const std::vector<int>& nA, const std::vector<int>& nB,
                   long long alphabetA, long long alphabetB,
                   int nBoot, TRandom3& rng,
                   double& meanOut, double& sigmaOut) {
    int N = (int)nA.size();
    std::vector<double> vals(nBoot);
    for (int b = 0; b < nBoot; ++b) {
        std::vector<int> rA(N), rB(N);
        for (int i = 0; i < N; ++i) {
            int idx = rng.Integer(N);
            rA[i] = nA[idx];
            rB[i] = nB[idx];
        }
        double sba, sig;
        sbaNsb(rA, rB, alphabetA, alphabetB, sba, sig);
        vals[b] = sba;
    }
    double mean = 0; for (double v : vals) mean += v; mean /= nBoot;
    double var = 0; for (double v : vals) var += (v-mean)*(v-mean); var /= (nBoot-1);
    meanOut = mean;
    sigmaOut = std::sqrt(var);
}

// =============================================================================
// Unified bootstrap across ALL estimators (naive/Miller-Madow/shuffle/NSB for
// I(A:B); naive/Miller-Madow/NSB for S(B|A)), from the SAME resampled indices
// each iteration -- so different methods' error bars come from one consistent
// procedure rather than mixed conventions. Mirrors
// bootstrap_all_estimators() in analyze_dijet_entropy.py exactly.
//
// No shuffle-subtraction analogue for S(B|A): shuffle-subtraction targets
// the joint-vs-marginal bias mechanism specific to I(A:B) (see toy-study
// Section 5); Miller-Madow's per-entropy correction, by contrast, applies
// directly to S_A and S_AB individually and generalizes to S(B|A) trivially.
// =============================================================================

struct AllEstimatorsResult {
    double I_naive, I_naive_err;
    double I_mm, I_mm_err;
    double I_shuffle, I_shuffle_err;
    double I_nsb, I_nsb_err;
    double SBA_naive, SBA_naive_err;
    double SBA_mm, SBA_mm_err;
    double SBA_nsb, SBA_nsb_err;
    double NMI_naive, NMI_naive_err;
    double NMI_mm, NMI_mm_err;
    double NMI_nsb, NMI_nsb_err;
};

AllEstimatorsResult bootstrapAllEstimators(const std::vector<int>& nA, const std::vector<int>& nB,
                                            long long alphabetA, long long alphabetB,
                                            int nBoot, int nShufflesInner, TRandom3& rng) {
    // Point estimates: computed ONCE, on the real (unresampled) data --
    // exactly matching naiveMI/millerMadowMI/shuffleCorrectionMI/nsbMI
    // used everywhere else in this macro. Bootstrap resampling below is
    // used ONLY for uncertainty (std across resamples), never to redefine
    // the central value. (Earlier version used the bootstrap MEAN as the
    // point estimate for every method, which silently made the "NSB"
    // value here disagree with the single-shot NSB value used in
    // plotEntropyVsPt/plotMIVsPt/plotSBAVsPt -- fixed for consistency.)
    int N = (int)nA.size();

    NaiveResult naivePoint = naiveMI(nA, nB);
    double mmBiasPoint;
    double I_mm_point = millerMadowMI(naivePoint, mmBiasPoint);
    double S_A_mm_point = naivePoint.S_A - (naivePoint.K_A - 1) / (2.0 * naivePoint.N);
    double S_AB_mm_point = naivePoint.S_AB - (naivePoint.K_AB - 1) / (2.0 * naivePoint.N);
    double shBiasPoint, shBiasErrPoint;
    double I_shuffle_point = shuffleCorrectionMI(nA, nB, nShufflesInner, rng, shBiasPoint, shBiasErrPoint);
    NSBmiResult nsbPoint = nsbMI(nA, nB, alphabetA, alphabetB);

    AllEstimatorsResult r;
    r.I_naive   = naivePoint.I;
    r.SBA_naive = naivePoint.S_AB - naivePoint.S_A;
    r.I_mm      = I_mm_point;
    r.SBA_mm    = S_AB_mm_point - S_A_mm_point;
    r.I_shuffle = I_shuffle_point;
    r.I_nsb     = nsbPoint.I;
    r.SBA_nsb   = nsbPoint.S_AB - nsbPoint.S_A;

    // NMI point estimates (all single-shot, on the real data -- same
    // convention as I_naive/I_mm/I_nsb above).
    r.NMI_naive = r.I_naive / std::min(naivePoint.S_A, naivePoint.S_B);
    r.NMI_mm    = I_mm_point / std::min(S_A_mm_point, naivePoint.S_B - (naivePoint.K_B-1)/(2.0*naivePoint.N));
    r.NMI_nsb   = nsbPoint.NMI;
    // NMI_nsb_err is NOT set here -- fixed below to come from the SAME
    // bootstrap loop as I_nsb_err/SBA_nsb_err (see bug note below).

    // ---- uncertainty: spread over bootstrap resamples ----
    std::vector<double> vI_naive(nBoot), vI_mm(nBoot), vI_shuffle(nBoot), vI_nsb(nBoot);
    std::vector<double> vSBA_naive(nBoot), vSBA_mm(nBoot), vSBA_nsb(nBoot);
    std::vector<double> vNMI_naive(nBoot), vNMI_mm(nBoot), vNMI_nsb(nBoot);

    for (int b = 0; b < nBoot; ++b) {
        std::vector<int> rA(N), rB(N);
        for (int i = 0; i < N; ++i) {
            int idx = rng.Integer(N);
            rA[i] = nA[idx];
            rB[i] = nB[idx];
        }

        NaiveResult naive = naiveMI(rA, rB);
        vI_naive[b] = naive.I;
        vSBA_naive[b] = naive.S_AB - naive.S_A;
        vNMI_naive[b] = naive.I / std::min(naive.S_A, naive.S_B);

        double mmBias;
        vI_mm[b] = millerMadowMI(naive, mmBias);
        double S_A_mm = naive.S_A - (naive.K_A - 1) / (2.0 * naive.N);
        double S_AB_mm = naive.S_AB - (naive.K_AB - 1) / (2.0 * naive.N);
        double S_B_mm = naive.S_B - (naive.K_B - 1) / (2.0 * naive.N);
        vSBA_mm[b] = S_AB_mm - S_A_mm;
        vNMI_mm[b] = vI_mm[b] / std::min(S_A_mm, S_B_mm);

        double shBias, shBiasErr;
        vI_shuffle[b] = shuffleCorrectionMI(rA, rB, nShufflesInner, rng, shBias, shBiasErr);

        NSBmiResult nsb = nsbMI(rA, rB, alphabetA, alphabetB);
        vI_nsb[b] = nsb.I;
        vSBA_nsb[b] = nsb.S_AB - nsb.S_A;
        // BUG FIX: this was previously missing -- NMI_nsb_err was being
        // taken from nsbMI()'s own analytic ratio-propagation formula
        // (a completely different, more conservative methodology than
        // the bootstrap used for I_nsb_err/SBA_nsb_err), which silently
        // made NSB's NMI error bar inconsistent with its own I/S(B|A)
        // error bars in the same comparison plot. Fixed to reuse this
        // resample's already-computed nsb.S_A/nsb.S_B (no extra cost)
        // and go through the same bootstrap-of-the-composite-quantity
        // procedure as everything else in this loop.
        vNMI_nsb[b] = nsb.I / std::min(nsb.S_A, nsb.S_B);
    }

    auto stdOnly = [](const std::vector<double>& v) {
        double mean = 0; for (double x : v) mean += x; mean /= v.size();
        double var = 0; for (double x : v) var += (x-mean)*(x-mean); var /= (v.size()-1);
        return std::sqrt(var);
    };

    r.I_naive_err   = stdOnly(vI_naive);
    r.I_mm_err      = stdOnly(vI_mm);
    r.I_shuffle_err = stdOnly(vI_shuffle);
    r.I_nsb_err     = stdOnly(vI_nsb);
    r.SBA_naive_err = stdOnly(vSBA_naive);
    r.SBA_mm_err    = stdOnly(vSBA_mm);
    r.SBA_nsb_err   = stdOnly(vSBA_nsb);
    r.NMI_naive_err = stdOnly(vNMI_naive);
    r.NMI_mm_err    = stdOnly(vNMI_mm);
    r.NMI_nsb_err   = stdOnly(vNMI_nsb);
    return r;
}

// =============================================================================
// NBD method-of-moments fit + pmf (for the multiplicity-distribution check)
// =============================================================================

bool fitNBDMoments(const std::vector<int>& counts, double& r, double& p,
                    double& meanOut, double& varOut) {
    double N = counts.size();
    double mean = 0; for (int c : counts) mean += c; mean /= N;
    double var = 0; for (int c : counts) var += (c-mean)*(c-mean); var /= (N-1);
    meanOut = mean; varOut = var;
    if (var <= mean) return false;   // under-dispersed; NBD moment fit degenerate
    p = mean / var;
    r = mean*mean / (var - mean);
    return true;
}

double nbdPmf(int n, double r, double p) {
    double logPmf = TMath::LnGamma(n + r) - TMath::LnGamma(r) - TMath::LnGamma(n + 1.0)
                    + r*std::log(p) + n*std::log(1.0 - p);
    return std::exp(logPmf);
}

// =============================================================================
// Data loading
// =============================================================================

struct DijetData {
    std::vector<int> nA, nB;
    std::vector<double> jetPt;   // jet1 (leading) pT -- used for pT-differential binning
    std::vector<double> jet2Pt;  // jet2 (subleading) pT -- used for the new pT-distribution plots
};

DijetData loadDijetTree(const char* rootFile, const TString& level) {
    TFile* f = TFile::Open(rootFile, "READ");
    if (!f || f->IsZombie()) {
        std::cerr << "ERROR: could not open " << rootFile << "\n";
        return {};
    }
    TTree* tree = (TTree*)f->Get("dijet");

    TString suffix = (level == "det") ? "_det" : "_truth";

    Int_t jet1_mult, jet2_mult;
    Double_t jet1_pt, jet2_pt;
    Int_t hasDetDijet = 1;  // default true for truth level (no filter needed)

    tree->SetBranchAddress(Form("jet1_mult%s", suffix.Data()), &jet1_mult);
    tree->SetBranchAddress(Form("jet2_mult%s", suffix.Data()), &jet2_mult);
    tree->SetBranchAddress(Form("jet1_pt%s", suffix.Data()), &jet1_pt);
    tree->SetBranchAddress(Form("jet2_pt%s", suffix.Data()), &jet2_pt);
    if (level == "det") tree->SetBranchAddress("hasDetDijet", &hasDetDijet);

    DijetData data;
    Long64_t nEntries = tree->GetEntries();
    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        if (level == "det" && hasDetDijet != 1) continue;
        data.nA.push_back(jet1_mult);
        data.nB.push_back(jet2_mult);
        data.jetPt.push_back(jet1_pt);
        data.jet2Pt.push_back(jet2_pt);
    }

    if (level == "det") {
        std::cout << "  detector-level filter (hasDetDijet==1): kept " << data.nA.size()
                  << "/" << nEntries << " ("
                  << 100.0*data.nA.size()/nEntries << "%) of truth-selected events\n";
    }

    f->Close();
    return data;
}

// =============================================================================
// Plotting
// =============================================================================

void plotMultiplicityDistributions(const std::vector<int>& nA, const std::vector<int>& nB,
                                    const TString& outPrefix) {
    TCanvas* c = new TCanvas("cMult", "Multiplicity distributions", 1200, 500);
    c->Divide(2, 1);

    struct JetSpec { const std::vector<int>* counts; const char* label; int color; };
    JetSpec specs[2] = {
        {&nA, "jet A (leading)", kBlue+1},
        {&nB, "jet B (subleading)", kRed+1},
    };

    for (int j = 0; j < 2; ++j) {
        c->cd(j+1);
        const std::vector<int>& counts = *specs[j].counts;
        int nmax = *std::max_element(counts.begin(), counts.end());
        int N = counts.size();

        TH1D* h = new TH1D(Form("hMult%d", j), Form("%s;charged multiplicity;probability", specs[j].label),
                            nmax+2, -0.5, nmax+1.5);
        for (int c : counts) h->Fill(c);
        h->Scale(1.0 / N);
        h->SetMarkerStyle(20);
        h->SetMarkerColor(specs[j].color);
        h->SetLineColor(specs[j].color);
        h->SetStats(0);
        h->Draw("E1");

        double r, p, meanV, varV;
        bool ok = fitNBDMoments(counts, r, p, meanV, varV);

        TLegend* leg = new TLegend(0.45, 0.65, 0.88, 0.88);
        leg->AddEntry(h, Form("data (N=%d)", N), "lep");

        if (ok) {
            TH1D* hNBD = new TH1D(Form("hNBD%d", j), "", nmax+2, -0.5, nmax+1.5);
            TH1D* hPoisson = new TH1D(Form("hPois%d", j), "", nmax+2, -0.5, nmax+1.5);
            for (int n = 0; n <= nmax+1; ++n) {
                hNBD->SetBinContent(n+1, nbdPmf(n, r, p));
                hPoisson->SetBinContent(n+1, TMath::Poisson(n, meanV));
            }
            hNBD->SetLineColor(kBlack);
            hNBD->SetLineWidth(2);
            hNBD->SetStats(0);
            hNBD->Draw("L SAME");
            hPoisson->SetLineColor(kGray+1);
            hPoisson->SetLineStyle(2);
            hPoisson->SetStats(0);
            hPoisson->Draw("L SAME");
            leg->AddEntry(hNBD, Form("NBD fit: r=%.2f p=%.3f (var/mean=%.2f)", r, p, varV/meanV), "l");
            leg->AddEntry(hPoisson, Form("Poisson, same mean (%.2f)", meanV), "l");
        } else {
            std::cout << "  [" << specs[j].label << "] var<=mean: under-dispersed, "
                      << "NBD moment fit undefined\n";
        }
        leg->Draw();
    }

    c->SaveAs(outPrefix + "_multiplicity_dists.png");
    std::cout << "  saved " << outPrefix << "_multiplicity_dists.png\n";
}

void plotJointMultiplicityDistribution(const std::vector<int>& nA, const std::vector<int>& nB,
                                        const TString& outPrefix) {
    int nmaxA = *std::max_element(nA.begin(), nA.end());
    int nmaxB = *std::max_element(nB.begin(), nB.end());

    TH2D* h2 = new TH2D("hJoint", "Joint distribution;N_{A};N_{B}",
                         nmaxA+2, -0.5, nmaxA+1.5, nmaxB+2, -0.5, nmaxB+1.5);
    for (size_t i = 0; i < nA.size(); ++i) h2->Fill(nA[i], nB[i]);

    int occupied = 0;
    for (int bx = 1; bx <= h2->GetNbinsX(); ++bx)
        for (int by = 1; by <= h2->GetNbinsY(); ++by)
            if (h2->GetBinContent(bx, by) > 0) ++occupied;
    long long fullGrid = (long long)(nmaxA+2) * (nmaxB+2);

    TCanvas* c = new TCanvas("cJoint", "Joint multiplicity distribution", 1200, 550);
    c->Divide(2, 1);

    c->cd(1);
    gPad->SetRightMargin(0.16);  // room for the COLZ z-axis palette
    h2->SetStats(0);
    h2->Draw("COLZ");

    c->cd(2);
    gPad->SetRightMargin(0.16);
    TH2D* h2sqrt = (TH2D*)h2->Clone("hJointSqrt");
    for (int bx = 1; bx <= h2sqrt->GetNbinsX(); ++bx)
        for (int by = 1; by <= h2sqrt->GetNbinsY(); ++by)
            h2sqrt->SetBinContent(bx, by, std::sqrt(h2sqrt->GetBinContent(bx, by)));
    h2sqrt->SetTitle("Joint distribution (sqrt scale);N_{A};N_{B}");
    h2sqrt->SetStats(0);
    h2sqrt->Draw("COLZ");

    c->SaveAs(outPrefix + "_joint_multiplicity_dist.png");
    std::cout << "  saved " << outPrefix << "_joint_multiplicity_dist.png\n";
    std::cout << "  N=" << nA.size() << ", K_AB (occupied) = " << occupied
              << ", full grid = " << (nmaxA+2) << "x" << (nmaxB+2) << " = " << fullGrid
              << ", filled fraction = " << (double)occupied/fullGrid << "\n";
    if ((double)occupied/fullGrid < 0.1)
        std::cout << "  NOTE: joint grid is <10% filled -- sparse regime, NSB matters most here.\n";
}

void plotMultiplicityVsPtProfile(const std::vector<double>& jet1Pt, const std::vector<double>& jet2Pt,
                                  const std::vector<int>& nA, const std::vector<int>& nB,
                                  const TString& outPrefix, int nBins = 15) {
    // DIAGNOSTIC: raw profile of mean multiplicity vs pT, bypassing all
    // NSB/entropy/quantile-binning machinery entirely (TProfile just
    // computes per-bin mean +/- standard error directly from the tree).
    // See analyze_dijet_entropy.py's plot_multiplicity_vs_pt_profile()
    // docstring for how to interpret this against the entropy-vs-pT plots.

    int N = jet1Pt.size();
    double sumPtA=0, sumPtA2=0, sumMultA=0, sumPtMultA=0;
    for (int i = 0; i < N; ++i) {
        sumPtA += jet1Pt[i]; sumPtA2 += jet1Pt[i]*jet1Pt[i];
        sumMultA += nA[i]; sumPtMultA += jet1Pt[i]*nA[i];
    }
    double meanPtA = sumPtA/N, meanMultA = sumMultA/N;
    double covA = sumPtMultA/N - meanPtA*meanMultA;
    double varPtA = sumPtA2/N - meanPtA*meanPtA;
    double sdMultA = 0; for (int i=0;i<N;++i) sdMultA += (nA[i]-meanMultA)*(nA[i]-meanMultA);
    sdMultA = std::sqrt(sdMultA/N);
    double rA = covA / (std::sqrt(varPtA) * sdMultA);

    int NB = jet2Pt.size();
    double sumPtB=0, sumPtB2=0, sumMultB=0, sumPtMultB=0;
    for (int i = 0; i < NB; ++i) {
        sumPtB += jet2Pt[i]; sumPtB2 += jet2Pt[i]*jet2Pt[i];
        sumMultB += nB[i]; sumPtMultB += jet2Pt[i]*nB[i];
    }
    double meanPtB = sumPtB/NB, meanMultB = sumMultB/NB;
    double covB = sumPtMultB/NB - meanPtB*meanMultB;
    double varPtB = sumPtB2/NB - meanPtB*meanPtB;
    double sdMultB = 0; for (int i=0;i<NB;++i) sdMultB += (nB[i]-meanMultB)*(nB[i]-meanMultB);
    sdMultB = std::sqrt(sdMultB/NB);
    double rB = covB / (std::sqrt(varPtB) * sdMultB);

    std::cout << "  Raw Pearson correlation, jet1_pt vs N_A: " << rA << "\n";
    std::cout << "  Raw Pearson correlation, jet2_pt vs N_B: " << rB << "\n";
    if (std::abs(rA) < 0.05)
        std::cout << "  NOTE: |corr| < 0.05 -- essentially no linear relationship between "
                  << "jet 1 pT and its own multiplicity in this dataset. If you physically "
                  << "expect one, this points to the generator/tree, not the pT-binning "
                  << "downstream.\n";

    double pt1Min = *std::min_element(jet1Pt.begin(), jet1Pt.end());
    double pt1Max = *std::max_element(jet1Pt.begin(), jet1Pt.end());
    double pt2Min = *std::min_element(jet2Pt.begin(), jet2Pt.end());
    double pt2Max = *std::max_element(jet2Pt.begin(), jet2Pt.end());

    TProfile* profA = new TProfile("profA", Form("jet 1: N_A vs p_{T} (r=%.3f);jet p_{T} (GeV);mean charged multiplicity", rA),
                                    nBins, pt1Min, pt1Max);
    for (int i = 0; i < N; ++i) profA->Fill(jet1Pt[i], nA[i]);

    TProfile* profB = new TProfile("profB", Form("jet 2: N_B vs p_{T} (r=%.3f);jet p_{T} (GeV);mean charged multiplicity", rB),
                                    nBins, pt2Min, pt2Max);
    for (int i = 0; i < NB; ++i) profB->Fill(jet2Pt[i], nB[i]);

    TCanvas* c = new TCanvas("cMultVsPt", "Multiplicity vs pT profile", 1200, 500);
    c->Divide(2, 1);
    c->cd(1);
    profA->SetMarkerStyle(20); profA->SetMarkerColor(kBlue+1); profA->SetLineColor(kBlue+1);
    profA->SetStats(0);
    profA->Draw("E1");
    c->cd(2);
    profB->SetMarkerStyle(20); profB->SetMarkerColor(kRed+1); profB->SetLineColor(kRed+1);
    profB->SetStats(0);
    profB->Draw("E1");

    c->SaveAs(outPrefix + "_mult_vs_pt_profile.png");
    std::cout << "  saved " << outPrefix << "_mult_vs_pt_profile.png\n";
}

void plotPtDistributions(const std::vector<double>& jet1Pt, const std::vector<double>& jet2Pt,
                          const TString& outPrefix, int nBins = 40) {
    TCanvas* c = new TCanvas("cPt", "Jet pT distributions", 1200, 500);
    c->Divide(2, 1);

    struct JetSpec { const std::vector<double>* pt; const char* label; int color; };
    JetSpec specs[2] = {
        {&jet1Pt, "jet 1 (leading)", kBlue+1},
        {&jet2Pt, "jet 2 (subleading)", kRed+1},
    };

    for (int j = 0; j < 2; ++j) {
        c->cd(j+1);
        gPad->SetLogy();  // steeply-falling QCD jet pT spectrum -- linear
                          // scale compresses the whole tail into a sliver
                          // near y=0, log-scale needed to see its shape
        const std::vector<double>& pt = *specs[j].pt;
        double ptMin = *std::min_element(pt.begin(), pt.end());
        double ptMax = *std::max_element(pt.begin(), pt.end());
        int N = pt.size();

        TH1D* h = new TH1D(Form("hPt%d", j),
                            Form("%s;jet p_{T} (GeV);probability density (GeV^{-1})", specs[j].label),
                            nBins, ptMin, ptMax);
        for (double p : pt) h->Fill(p);
        h->Scale(1.0 / (N * h->GetBinWidth(1)));
        h->SetMarkerStyle(20);
        h->SetMarkerColor(specs[j].color);
        h->SetLineColor(specs[j].color);
        h->SetStats(0);
        h->Draw("E1");

        TLegend* leg = new TLegend(0.55, 0.78, 0.88, 0.88);
        leg->AddEntry(h, Form("%s (N=%d)", specs[j].label, N), "lep");
        leg->Draw();
    }

    c->SaveAs(outPrefix + "_pt_dists.png");
    std::cout << "  saved " << outPrefix << "_pt_dists.png\n";
}

void plotJointPtDistribution(const std::vector<double>& jet1Pt, const std::vector<double>& jet2Pt,
                              const TString& outPrefix, int nBins = 40) {
    double pt1Min = *std::min_element(jet1Pt.begin(), jet1Pt.end());
    double pt1Max = *std::max_element(jet1Pt.begin(), jet1Pt.end());
    double pt2Min = *std::min_element(jet2Pt.begin(), jet2Pt.end());
    double pt2Max = *std::max_element(jet2Pt.begin(), jet2Pt.end());

    TH2D* h2 = new TH2D("hJointPt",
                         "Joint p_{T} distribution;jet 1 (leading) p_{T} (GeV);jet 2 (subleading) p_{T} (GeV)",
                         nBins, pt1Min, pt1Max, nBins, pt2Min, pt2Max);
    for (size_t i = 0; i < jet1Pt.size(); ++i) h2->Fill(jet1Pt[i], jet2Pt[i]);

    TCanvas* c = new TCanvas("cJointPt", "Joint pT distribution", 1200, 550);
    c->Divide(2, 1);

    c->cd(1);
    gPad->SetRightMargin(0.16);  // room for the COLZ z-axis palette
    h2->SetStats(0);
    h2->Draw("COLZ");

    c->cd(2);
    gPad->SetRightMargin(0.16);
    TH2D* h2sqrt = (TH2D*)h2->Clone("hJointPtSqrt");
    for (int bx = 1; bx <= h2sqrt->GetNbinsX(); ++bx)
        for (int by = 1; by <= h2sqrt->GetNbinsY(); ++by)
            h2sqrt->SetBinContent(bx, by, std::sqrt(h2sqrt->GetBinContent(bx, by)));
    h2sqrt->SetTitle("Joint p_{T} distribution (sqrt scale);jet 1 (leading) p_{T} (GeV);jet 2 (subleading) p_{T} (GeV)");
    h2sqrt->SetStats(0);
    h2sqrt->Draw("COLZ");

    c->SaveAs(outPrefix + "_joint_pt_dist.png");
    std::cout << "  saved " << outPrefix << "_joint_pt_dist.png\n";
}

std::vector<double> quantileBinEdges(std::vector<double> pt, int nBins) {
    std::sort(pt.begin(), pt.end());
    int N = pt.size();
    std::vector<double> edges(nBins + 1);
    for (int i = 0; i <= nBins; ++i) {
        int idx = std::min(N-1, (int)((double)i / nBins * N));
        edges[i] = pt[idx];
    }
    edges[0] = pt.front();
    edges[nBins] = pt.back();
    return edges;
}

void plotMethodComparisonVsPt(const std::vector<double>& ptMid,
                               const std::vector<double>& I_naive, const std::vector<double>& I_naive_e,
                               const std::vector<double>& I_mm, const std::vector<double>& I_mm_e,
                               const std::vector<double>& I_shuffle, const std::vector<double>& I_shuffle_e,
                               const std::vector<double>& I_nsb, const std::vector<double>& I_nsb_e,
                               const std::vector<double>& SBA_naive, const std::vector<double>& SBA_naive_e,
                               const std::vector<double>& SBA_mm, const std::vector<double>& SBA_mm_e,
                               const std::vector<double>& SBA_nsb, const std::vector<double>& SBA_nsb_e,
                               const TString& outPrefix) {
    int n = ptMid.size();
    std::vector<double> zeros(n, 0.0);

    // small horizontal offsets (fraction of typical bin spacing) so
    // overlapping error bars from different methods stay legible
    double span = (n > 1) ? (ptMid.back() - ptMid.front()) / n : 1.0;
    std::vector<double> pt_naive(n), pt_mm(n), pt_shuffle(n), pt_nsb(n);
    std::vector<double> pt_naive2(n), pt_mm2(n), pt_nsb2(n);
    for (int i = 0; i < n; ++i) {
        pt_naive[i]  = ptMid[i] - 0.15*span;
        pt_mm[i]     = ptMid[i] - 0.05*span;
        pt_shuffle[i]= ptMid[i] + 0.05*span;
        pt_nsb[i]    = ptMid[i] + 0.15*span;
        pt_naive2[i] = ptMid[i] - 0.12*span;
        pt_mm2[i]    = ptMid[i];
        pt_nsb2[i]   = ptMid[i] + 0.12*span;
    }

    // --- MI comparison ---
    TCanvas* c1 = new TCanvas("cMICompare", "MI method comparison vs pT", 800, 550);
    TGraphErrors* gN = new TGraphErrors(n, pt_naive.data(), I_naive.data(), zeros.data(), I_naive_e.data());
    TGraphErrors* gM = new TGraphErrors(n, pt_mm.data(), I_mm.data(), zeros.data(), I_mm_e.data());
    TGraphErrors* gS = new TGraphErrors(n, pt_shuffle.data(), I_shuffle.data(), zeros.data(), I_shuffle_e.data());
    TGraphErrors* gB = new TGraphErrors(n, pt_nsb.data(), I_nsb.data(), zeros.data(), I_nsb_e.data());
    gN->SetMarkerStyle(20); gN->SetMarkerColor(kRed+1);   gN->SetLineColor(kRed+1);
    gM->SetMarkerStyle(21); gM->SetMarkerColor(kOrange+1);gM->SetLineColor(kOrange+1);
    gS->SetMarkerStyle(22); gS->SetMarkerColor(kBlue+1);  gS->SetLineColor(kBlue+1);
    gB->SetMarkerStyle(23); gB->SetMarkerColor(kViolet+1);gB->SetLineColor(kViolet+1);

    TMultiGraph* mg1 = new TMultiGraph();
    mg1->Add(gN, "P"); mg1->Add(gM, "P"); mg1->Add(gS, "P"); mg1->Add(gB, "P");
    mg1->SetTitle("Method comparison: MI vs. p_{T} (bootstrap error bars);leading jet p_{T} (GeV);I(A:B) (nats)");

    // Expand y-range with headroom above the highest point+error so the
    // top-right legend has guaranteed clear space, regardless of how far
    // up the error bars reach (this was previously overlapping data,
    // e.g. the widest-error highest-pT bin).
    {
        double yMax = -1e300, yMin = 1e300;
        for (int i = 0; i < n; ++i) {
            yMax = std::max({yMax, I_naive[i]+I_naive_e[i], I_mm[i]+I_mm_e[i],
                              I_shuffle[i]+I_shuffle_e[i], I_nsb[i]+I_nsb_e[i]});
            yMin = std::min({yMin, I_naive[i]-I_naive_e[i], I_mm[i]-I_mm_e[i],
                              I_shuffle[i]-I_shuffle_e[i], I_nsb[i]-I_nsb_e[i]});
        }
        double range = yMax - yMin;
        mg1->SetMaximum(yMax + 0.35*range);
        mg1->SetMinimum(yMin - 0.05*range);
    }

    mg1->Draw("A");
    TLegend* leg1 = new TLegend(0.65, 0.68, 0.88, 0.88);
    leg1->AddEntry(gN, "naive", "lep");
    leg1->AddEntry(gM, "Miller-Madow", "lep");
    leg1->AddEntry(gS, "shuffle", "lep");
    leg1->AddEntry(gB, "NSB", "lep");
    leg1->Draw();
    // Use the ACTUAL rendered x-axis range (queried after Draw), not a
    // manually-estimated span -- guarantees the line spans exactly
    // edge-to-edge with no overshoot or gap, regardless of how
    // TMultiGraph auto-computed its x-range.
    TLine* zeroLine1 = new TLine(gPad->GetUxmin(), 0, gPad->GetUxmax(), 0);
    zeroLine1->SetLineColor(kGray+1);
    zeroLine1->Draw();
    c1->SaveAs(outPrefix + "_method_comparison_MI_vs_pt.png");
    std::cout << "  saved " << outPrefix << "_method_comparison_MI_vs_pt.png\n";

    // --- S(B|A) comparison (no shuffle -- see bootstrapAllEstimators docs) ---
    TCanvas* c2 = new TCanvas("cSBACompare", "S(B|A) method comparison vs pT", 800, 550);
    TGraphErrors* gN2 = new TGraphErrors(n, pt_naive2.data(), SBA_naive.data(), zeros.data(), SBA_naive_e.data());
    TGraphErrors* gM2 = new TGraphErrors(n, pt_mm2.data(), SBA_mm.data(), zeros.data(), SBA_mm_e.data());
    TGraphErrors* gB2 = new TGraphErrors(n, pt_nsb2.data(), SBA_nsb.data(), zeros.data(), SBA_nsb_e.data());
    gN2->SetMarkerStyle(20); gN2->SetMarkerColor(kRed+1);    gN2->SetLineColor(kRed+1);
    gM2->SetMarkerStyle(21); gM2->SetMarkerColor(kOrange+1); gM2->SetLineColor(kOrange+1);
    gB2->SetMarkerStyle(23); gB2->SetMarkerColor(kViolet+1); gB2->SetLineColor(kViolet+1);

    TMultiGraph* mg2 = new TMultiGraph();
    mg2->Add(gN2, "P"); mg2->Add(gM2, "P"); mg2->Add(gB2, "P");
    mg2->SetTitle("Method comparison: S(B|A) vs. p_{T} (bootstrap error bars);leading jet p_{T} (GeV);S(B|A) (nats)");

    {
        double yMax = -1e300, yMin = 1e300;
        for (int i = 0; i < n; ++i) {
            yMax = std::max({yMax, SBA_naive[i]+SBA_naive_e[i], SBA_mm[i]+SBA_mm_e[i],
                              SBA_nsb[i]+SBA_nsb_e[i]});
            yMin = std::min({yMin, SBA_naive[i]-SBA_naive_e[i], SBA_mm[i]-SBA_mm_e[i],
                              SBA_nsb[i]-SBA_nsb_e[i]});
        }
        double range = yMax - yMin;
        mg2->SetMaximum(yMax + 0.35*range);
        mg2->SetMinimum(yMin - 0.05*range);
    }

    mg2->Draw("A");
    TLegend* leg2 = new TLegend(0.65, 0.72, 0.88, 0.88);
    leg2->AddEntry(gN2, "naive", "lep");
    leg2->AddEntry(gM2, "Miller-Madow", "lep");
    leg2->AddEntry(gB2, "NSB", "lep");
    leg2->Draw();
    TLine* zeroLine2 = new TLine(gPad->GetUxmin(), 0, gPad->GetUxmax(), 0);
    zeroLine2->SetLineColor(kRed);
    zeroLine2->SetLineStyle(3);
    zeroLine2->Draw();
    c2->SaveAs(outPrefix + "_method_comparison_SBA_vs_pt.png");
    std::cout << "  saved " << outPrefix << "_method_comparison_SBA_vs_pt.png\n";
}

void plotMethodComparisonNMIVsPt(const std::vector<double>& ptMid,
                                  const std::vector<double>& NMI_naive, const std::vector<double>& NMI_naive_e,
                                  const std::vector<double>& NMI_mm, const std::vector<double>& NMI_mm_e,
                                  const std::vector<double>& NMI_nsb, const std::vector<double>& NMI_nsb_e,
                                  const TString& outPrefix) {
    // NEW: compare naive/Miller-Madow/NSB NMI = I/min(S_A,S_B) vs pT (no
    // shuffle, same reason as the S(B|A) comparison). Separate new plot
    // file; plotMethodComparisonVsPt above is untouched.
    int n = ptMid.size();
    std::vector<double> zeros(n, 0.0);
    double span = (n > 1) ? (ptMid.back() - ptMid.front()) / n : 1.0;
    std::vector<double> ptN(n), ptM(n), ptB(n);
    for (int i = 0; i < n; ++i) {
        ptN[i] = ptMid[i] - 0.12*span;
        ptM[i] = ptMid[i];
        ptB[i] = ptMid[i] + 0.12*span;
    }

    TCanvas* c = new TCanvas("cNMICompare", "NMI method comparison vs pT", 800, 550);
    TGraphErrors* gN = new TGraphErrors(n, ptN.data(), NMI_naive.data(), zeros.data(), NMI_naive_e.data());
    TGraphErrors* gM = new TGraphErrors(n, ptM.data(), NMI_mm.data(), zeros.data(), NMI_mm_e.data());
    TGraphErrors* gB = new TGraphErrors(n, ptB.data(), NMI_nsb.data(), zeros.data(), NMI_nsb_e.data());
    gN->SetMarkerStyle(20); gN->SetMarkerColor(kRed+1);    gN->SetLineColor(kRed+1);
    gM->SetMarkerStyle(21); gM->SetMarkerColor(kOrange+1); gM->SetLineColor(kOrange+1);
    gB->SetMarkerStyle(23); gB->SetMarkerColor(kViolet+1); gB->SetLineColor(kViolet+1);

    TMultiGraph* mg = new TMultiGraph();
    mg->Add(gN, "P"); mg->Add(gM, "P"); mg->Add(gB, "P");
    mg->SetTitle("Method comparison: NMI vs. p_{T} (bootstrap error bars);leading jet p_{T} (GeV);I(A:B)/min(S_{A},S_{B})");

    {
        double yMax = -1e300, yMin = 1e300;
        for (int i = 0; i < n; ++i) {
            yMax = std::max({yMax, NMI_naive[i]+NMI_naive_e[i], NMI_mm[i]+NMI_mm_e[i],
                              NMI_nsb[i]+NMI_nsb_e[i]});
            yMin = std::min({yMin, NMI_naive[i]-NMI_naive_e[i], NMI_mm[i]-NMI_mm_e[i],
                              NMI_nsb[i]-NMI_nsb_e[i]});
        }
        double range = yMax - yMin;
        mg->SetMaximum(yMax + 0.35*range);
        mg->SetMinimum(yMin - 0.05*range);
    }

    mg->Draw("A");
    TLegend* leg = new TLegend(0.65, 0.68, 0.88, 0.88);
    leg->AddEntry(gN, "naive", "lep");
    leg->AddEntry(gM, "Miller-Madow", "lep");
    leg->AddEntry(gB, "NSB", "lep");
    leg->Draw();
    TLine* oneLine = new TLine(gPad->GetUxmin(), 1, gPad->GetUxmax(), 1);
    oneLine->SetLineColor(kRed);
    oneLine->SetLineStyle(3);
    oneLine->Draw();
    c->SaveAs(outPrefix + "_method_comparison_NMI_vs_pt.png");
    std::cout << "  saved " << outPrefix << "_method_comparison_NMI_vs_pt.png\n";
}

void ptDifferentialAnalysis(const std::vector<int>& nA, const std::vector<int>& nB,
                             const std::vector<double>& jetPt,
                             long long alphabetA, long long alphabetB,
                             int nBins, const TString& outPrefix,
                             bool compareMethods = false, int nBootCompare = 50,
                             int nShufflesInner = 30, TRandom3* rngPtr = nullptr) {
    std::vector<double> edges = quantileBinEdges(jetPt, nBins);

    std::vector<double> ptMid, S_A, S_A_e, S_B, S_B_e, S_AB, S_AB_e, I, I_e, SBA, SBA_e, NMI, NMI_e;

    // multi-method comparison accumulators (only filled if compareMethods)
    std::vector<double> cI_naive, cI_naive_e, cI_mm, cI_mm_e, cI_shuffle, cI_shuffle_e, cI_nsb, cI_nsb_e;
    std::vector<double> cSBA_naive, cSBA_naive_e, cSBA_mm, cSBA_mm_e, cSBA_nsb, cSBA_nsb_e;
    std::vector<double> cNMI_naive, cNMI_naive_e, cNMI_mm, cNMI_mm_e, cNMI_nsb, cNMI_nsb_e;

    std::cout << "\n  pT range        |     N |     S_A      |     S_B      |    S_AB      |    I(A:B)    |    S(B|A)\n";
    std::cout << "  --------------------------------------------------------------------------------------------------\n";

    for (int ib = 0; ib < nBins; ++ib) {
        double lo = edges[ib], hi = edges[ib+1];
        std::vector<int> binA, binB;
        double ptSum = 0;
        for (size_t i = 0; i < nA.size(); ++i) {
            bool inBin = (ib < nBins-1) ? (jetPt[i] >= lo && jetPt[i] < hi)
                                         : (jetPt[i] >= lo && jetPt[i] <= hi);
            if (inBin) {
                binA.push_back(nA[i]);
                binB.push_back(nB[i]);
                ptSum += jetPt[i];
            }
        }
        int n = binA.size();
        if (n < 5) continue;
        if (n < 30) std::cout << "  WARNING: bin [" << lo << "," << hi << ") has only " << n
                               << " events -- results will be noisy.\n";

        NSBmiResult r = nsbMI(binA, binB, alphabetA, alphabetB);
        double sba, sbaErr;
        sbaNsb(binA, binB, alphabetA, alphabetB, sba, sbaErr);

        ptMid.push_back(ptSum / n);
        S_A.push_back(r.S_A); S_A_e.push_back(r.S_A_err);
        S_B.push_back(r.S_B); S_B_e.push_back(r.S_B_err);
        S_AB.push_back(r.S_AB); S_AB_e.push_back(r.S_AB_err);
        I.push_back(r.I); I_e.push_back(r.I_err);
        SBA.push_back(sba); SBA_e.push_back(sbaErr);
        NMI.push_back(r.NMI); NMI_e.push_back(r.NMI_err);

        printf("  [%5.1f,%5.1f) | %5d | %6.3f+/-%5.3f | %6.3f+/-%5.3f | %6.3f+/-%5.3f | %6.3f+/-%5.3f | %6.3f+/-%5.3f\n",
               lo, hi, n, r.S_A, r.S_A_err, r.S_B, r.S_B_err, r.S_AB, r.S_AB_err, r.I, r.I_err, sba, sbaErr);

        if (compareMethods && rngPtr) {
            AllEstimatorsResult boot = bootstrapAllEstimators(binA, binB, alphabetA, alphabetB,
                                                               nBootCompare, nShufflesInner, *rngPtr);
            cI_naive.push_back(boot.I_naive); cI_naive_e.push_back(boot.I_naive_err);
            cI_mm.push_back(boot.I_mm); cI_mm_e.push_back(boot.I_mm_err);
            cI_shuffle.push_back(boot.I_shuffle); cI_shuffle_e.push_back(boot.I_shuffle_err);
            cI_nsb.push_back(boot.I_nsb); cI_nsb_e.push_back(boot.I_nsb_err);
            cSBA_naive.push_back(boot.SBA_naive); cSBA_naive_e.push_back(boot.SBA_naive_err);
            cSBA_mm.push_back(boot.SBA_mm); cSBA_mm_e.push_back(boot.SBA_mm_err);
            cSBA_nsb.push_back(boot.SBA_nsb); cSBA_nsb_e.push_back(boot.SBA_nsb_err);
            cNMI_naive.push_back(boot.NMI_naive); cNMI_naive_e.push_back(boot.NMI_naive_err);
            cNMI_mm.push_back(boot.NMI_mm); cNMI_mm_e.push_back(boot.NMI_mm_err);
            cNMI_nsb.push_back(boot.NMI_nsb); cNMI_nsb_e.push_back(boot.NMI_nsb_err);
        }
    }

    int n = ptMid.size();
    if (n == 0) {
        std::cout << "  No pT bins had enough events -- skipping pT-differential plots.\n";
        return;
    }
    std::vector<double> zeros(n, 0.0);

    // --- entropy vs pT ---
    TCanvas* c1 = new TCanvas("cEntropyPt", "Entropy vs pT", 800, 550);
    TGraphErrors* gA = new TGraphErrors(n, ptMid.data(), S_A.data(), zeros.data(), S_A_e.data());
    TGraphErrors* gB = new TGraphErrors(n, ptMid.data(), S_B.data(), zeros.data(), S_B_e.data());
    TGraphErrors* gAB = new TGraphErrors(n, ptMid.data(), S_AB.data(), zeros.data(), S_AB_e.data());
    gA->SetLineColor(kBlue+1); gA->SetMarkerColor(kBlue+1); gA->SetMarkerStyle(20);
    gB->SetLineColor(kRed+1); gB->SetMarkerColor(kRed+1); gB->SetMarkerStyle(21);
    gAB->SetLineColor(kGreen+2); gAB->SetMarkerColor(kGreen+2); gAB->SetMarkerStyle(22);
    TMultiGraph* mg = new TMultiGraph();
    mg->Add(gA); mg->Add(gB); mg->Add(gAB);
    mg->SetTitle("NSB entropy vs. leading-jet p_{T};leading jet p_{T} (GeV);entropy (nats)");

    {
        double yMax = -1e300, yMin = 1e300;
        for (int i = 0; i < n; ++i) {
            yMax = std::max({yMax, S_A[i]+S_A_e[i], S_B[i]+S_B_e[i], S_AB[i]+S_AB_e[i]});
            yMin = std::min({yMin, S_A[i]-S_A_e[i], S_B[i]-S_B_e[i], S_AB[i]-S_AB_e[i]});
        }
        double range = yMax - yMin;
        // Modest headroom only -- S_AB sits well above S_A/S_B across the
        // whole pT range, leaving a large natural gap between them, so
        // the legend goes there instead of needing extra top margin
        // (previously 30%, which left roughly half the plot empty above
        // the data for no real reason).
        mg->SetMaximum(yMax + 0.08*range);
        mg->SetMinimum(yMin - 0.05*range);
    }

    mg->Draw("APL");
    // Positioned in the empty band between the S_AB curve (high) and the
    // S_A/S_B curves (low), rather than above everything -- avoids
    // wasting vertical range on headroom purely for legend space.
    TLegend* leg1 = new TLegend(0.38, 0.42, 0.65, 0.58);
    leg1->AddEntry(gA, "S_{A} (leading)", "lep");
    leg1->AddEntry(gB, "S_{B} (subleading)", "lep");
    leg1->AddEntry(gAB, "S_{AB} (joint)", "lep");
    leg1->Draw();
    c1->SaveAs(outPrefix + "_entropy_vs_pt.png");
    std::cout << "  saved " << outPrefix << "_entropy_vs_pt.png\n";

    // --- MI vs pT ---
    TCanvas* c2 = new TCanvas("cMIPt", "MI vs pT", 800, 550);
    TGraphErrors* gI = new TGraphErrors(n, ptMid.data(), I.data(), zeros.data(), I_e.data());
    gI->SetLineColor(kViolet+1); gI->SetMarkerColor(kViolet+1); gI->SetMarkerStyle(20);
    gI->SetTitle("Mutual information vs. leading-jet p_{T};leading jet p_{T} (GeV);I(A:B) (nats), NSB");
    gI->Draw("APL");
    TLine* zeroLineMI = new TLine(gPad->GetUxmin(), 0, gPad->GetUxmax(), 0);
    zeroLineMI->SetLineColor(kGray+1);
    zeroLineMI->Draw();
    c2->SaveAs(outPrefix + "_MI_vs_pt.png");
    std::cout << "  saved " << outPrefix << "_MI_vs_pt.png\n";

    // --- S(B|A) vs pT ---
    TCanvas* c3 = new TCanvas("cSBAPt", "S(B|A) vs pT", 800, 550);
    TGraphErrors* gSBA = new TGraphErrors(n, ptMid.data(), SBA.data(), zeros.data(), SBA_e.data());
    gSBA->SetLineColor(kMagenta+2); gSBA->SetMarkerColor(kMagenta+2); gSBA->SetMarkerStyle(20);
    gSBA->SetTitle("Conditional entropy vs. leading-jet p_{T};leading jet p_{T} (GeV);S(B|A) (nats), NSB");
    gSBA->Draw("APL");
    TLine* zeroLine = new TLine(gPad->GetUxmin(), 0, gPad->GetUxmax(), 0);
    zeroLine->SetLineColor(kRed);
    zeroLine->SetLineStyle(3);
    zeroLine->Draw();
    c3->SaveAs(outPrefix + "_SBA_vs_pt.png");
    std::cout << "  saved " << outPrefix << "_SBA_vs_pt.png\n";

    // --- NMI = I/min(S_A,S_B) vs pT (NEW; separate file, c1/c2/c3 untouched) ---
    TCanvas* c4 = new TCanvas("cNMIPt", "NMI vs pT", 800, 550);
    TGraphErrors* gNMI = new TGraphErrors(n, ptMid.data(), NMI.data(), zeros.data(), NMI_e.data());
    gNMI->SetLineColor(kGreen+2); gNMI->SetMarkerColor(kGreen+2); gNMI->SetMarkerStyle(20);
    gNMI->SetTitle("Normalized mutual information vs. leading-jet p_{T};leading jet p_{T} (GeV);I(A:B)/min(S_{A},S_{B}), NSB");
    gNMI->Draw("APL");
    TLine* zeroLineNMI = new TLine(gPad->GetUxmin(), 0, gPad->GetUxmax(), 0);
    zeroLineNMI->SetLineColor(kGray+1);
    zeroLineNMI->Draw();
    // No legend here -- consistent with the standalone MI_vs_pt/SBA_vs_pt
    // plots above, which likewise don't label their reference line. The
    // classical bound (NMI=1) line itself is only drawn/labeled in the
    // multi-method comparison plot, where a legend already exists for
    // the data series and the bound fits naturally alongside it.
    c4->SaveAs(outPrefix + "_NMI_vs_pt.png");
    std::cout << "  saved " << outPrefix << "_NMI_vs_pt.png\n";

    if (compareMethods) {
        plotMethodComparisonVsPt(ptMid,
                                  cI_naive, cI_naive_e, cI_mm, cI_mm_e,
                                  cI_shuffle, cI_shuffle_e, cI_nsb, cI_nsb_e,
                                  cSBA_naive, cSBA_naive_e, cSBA_mm, cSBA_mm_e,
                                  cSBA_nsb, cSBA_nsb_e, outPrefix);
        plotMethodComparisonNMIVsPt(ptMid, cNMI_naive, cNMI_naive_e,
                                     cNMI_mm, cNMI_mm_e, cNMI_nsb, cNMI_nsb_e, outPrefix);
    }
}

void plotSummaryBarChart(double I_naive, double I_naive_err,
                          double I_mm, double I_mm_err,
                          double I_sh, double I_sh_err,
                          double I_nsb, double I_nsb_err,
                          double SBA_nsb, double sigmaBoot,
                          const TString& outPrefix) {
    TCanvas* c = new TCanvas("cSummary", "Summary", 1100, 450);
    c->Divide(2, 1);

    c->cd(1);
    TH1D* h = new TH1D("hSummary", "MI estimators, this dataset;;I(A:B) (nats)", 4, 0, 4);
    h->SetBinContent(1, I_naive); h->SetBinError(1, I_naive_err);
    h->SetBinContent(2, I_mm);    h->SetBinError(2, I_mm_err);
    h->SetBinContent(3, I_sh);    h->SetBinError(3, I_sh_err);
    h->SetBinContent(4, I_nsb);   h->SetBinError(4, I_nsb_err);
    h->GetXaxis()->SetBinLabel(1, "naive");
    h->GetXaxis()->SetBinLabel(2, "Miller-Madow");
    h->GetXaxis()->SetBinLabel(3, "shuffle");
    h->GetXaxis()->SetBinLabel(4, "NSB");
    h->SetFillColorAlpha(kAzure+1, 0.6);
    h->SetLineColor(kBlack);
    h->SetStats(0);
    h->Draw("E1 HIST");

    c->cd(2);
    TGraphErrors* g = new TGraphErrors(1);
    g->SetPoint(0, 0, SBA_nsb);
    g->SetPointError(0, 0, sigmaBoot);
    g->SetMarkerStyle(20);
    g->SetMarkerColor(kMagenta+2);
    g->SetMarkerSize(1.5);
    g->SetLineColor(kMagenta+2);
    g->SetTitle("Conditional entropy (quantum witness);;S(B|A) (nats)");
    g->GetXaxis()->SetLimits(-1, 1);
    g->Draw("AP");
    TLine* zeroLine = new TLine(-1, 0, 1, 0);
    zeroLine->SetLineColor(kRed);
    zeroLine->SetLineStyle(3);
    zeroLine->Draw();

    c->SaveAs(outPrefix + "_summary.png");
    std::cout << "  saved " << outPrefix << "_summary.png\n";
}

// =============================================================================
// Main entry point
// =============================================================================

void analyze_dijet_entropy(const char* rootFile, const char* levelIn = "truth",
                            long long alphabet = 80, int nBoot = 200, int ptBins = 4,
                            bool compareMethods = true, int nBootCompare = 50,
                            int nShufflesInner = 30,
                            const char* figsDir = "/Users/zhoudunmingtu/bnl_work/Work/MODELS/EBCs_toyMC/analysis/figs") {
    TString level(levelIn);

    // ---- global plotting style: larger margins + axis title offsets so
    // longer axis titles (e.g. "NMI = I(A:B)/min(S_A,S_B), NSB",
    // "probability density (GeV^{-1})") aren't clipped by the canvas
    // edge -- ROOT's default margins are too tight for these. Set once,
    // before any canvas is created; applies to all pads/histograms/
    // graphs created afterward (including sub-pads from ->Divide()).
    // The two COLZ (2D) plotting functions override the right margin
    // locally to leave room for the z-axis palette.
    gStyle->SetPadLeftMargin(0.15);
    gStyle->SetPadRightMargin(0.06);
    gStyle->SetPadBottomMargin(0.13);
    gStyle->SetPadTopMargin(0.09);
    gStyle->SetTitleOffset(1.5, "Y");
    gStyle->SetTitleOffset(1.1, "X");
    gStyle->SetTitleSize(0.045, "XYZ");
    gStyle->SetLabelSize(0.035, "XYZ");
    gStyle->SetTitleSize(0.05, "T");   // canvas/pad title text

    TString figsDirStr(figsDir);
    gSystem->mkdir(figsDirStr, kTRUE);  // kTRUE = create parent dirs too if needed; no-op if it exists
    TString outPrefix = figsDirStr + TString::Format("/dijet_entropy_result_root_%s", levelIn);
    std::cout << "Figures will be saved under: " << figsDirStr << "\n";

    TRandom3 rng(12345);

    std::cout << "Loading " << rootFile << " (level='" << level << "') ...\n";
    DijetData data = loadDijetTree(rootFile, level);
    int N = data.nA.size();
    std::cout << "Loaded " << N << " back-to-back dijet events.\n";

    double meanA = 0, meanB = 0;
    for (int i = 0; i < N; ++i) { meanA += data.nA[i]; meanB += data.nB[i]; }
    meanA /= N; meanB /= N;
    std::cout << "  <N_A> = " << meanA << "   <N_B> = " << meanB << "\n";

    int maxA = *std::max_element(data.nA.begin(), data.nA.end());
    int maxB = *std::max_element(data.nB.begin(), data.nB.end());
    std::cout << "  max observed multiplicity: A=" << maxA << "  B=" << maxB
              << "  (alphabet K=" << alphabet << " chosen ahead of time)\n";
    if (maxA >= alphabet || maxB >= alphabet)
        std::cout << "  WARNING: observed multiplicity reaches the chosen alphabet bound "
                  << "-- increase alphabet and rerun.\n";

    // ---- point estimates ----
    std::cout << "\n--- Mutual information I(A:B) ---\n";
    NaiveResult naive = naiveMI(data.nA, data.nB);
    double mmBias;
    double I_mm = millerMadowMI(naive, mmBias);
    double shBias, shBiasErr;
    double I_sh = shuffleCorrectionMI(data.nA, data.nB, 200, rng, shBias, shBiasErr);
    NSBmiResult nsb = nsbMI(data.nA, data.nB, alphabet, alphabet);

    std::cout << "  K_A=" << naive.K_A << "  K_B=" << naive.K_B << "  K_AB=" << naive.K_AB
              << "  N=" << naive.N << "\n";
    printf("  naive          : %.4f nats\n", naive.I);
    printf("  Miller-Madow   : %.4f nats  (bias correction: %.4f)\n", I_mm, mmBias);
    printf("  shuffle-corr.  : %.4f nats  (bias estimate: %.4f +/- %.4f)\n", I_sh, shBias, shBiasErr);
    printf("  NSB            : %.4f +/- %.4f nats   <-- recommended\n", nsb.I, nsb.I_err);

    std::cout << "\n--- NMI = I(A:B) / min(S_A,S_B) (scale-comparable; classical bound: NMI<=1) ---\n";
    printf("  NSB            : %.4f +/- %.4f   (S_A=%.4f, S_B=%.4f, min=%.4f)\n",
           nsb.NMI, nsb.NMI_err, nsb.S_A, nsb.S_B, std::min(nsb.S_A, nsb.S_B));

    // ---- conditional entropy ----
    std::cout << "\n--- Conditional entropy S(B|A) = S_AB - S_A (quantum witness) ---\n";
    double SBA_nsb, sigmaAnalytic;
    sbaNsb(data.nA, data.nB, alphabet, alphabet, SBA_nsb, sigmaAnalytic);
    double bootMean, bootSigma;
    sbaBootstrap(data.nA, data.nB, alphabet, alphabet, nBoot, rng, bootMean, bootSigma);

    printf("  S(B|A)_NSB                       = %.4f nats\n", SBA_nsb);
    printf("  sigma (analytic, single dataset)  = %.4f nats\n", sigmaAnalytic);
    printf("  sigma (bootstrap, %d resamples)  = %.4f nats\n", nBoot, bootSigma);
    double z = (bootSigma > 0) ? SBA_nsb / bootSigma : 0.0/0.0;
    printf("\n  S(B|A) is %.2f sigma from zero (bootstrap sigma).\n", z);
    if (SBA_nsb < 0)
        std::cout << "  NEGATIVE conditional entropy -- classically impossible; significance = "
                  << std::abs(z) << " sigma.\n";
    else
        std::cout << "  Positive, consistent with a classical description at this precision.\n";

    // ---- (ii) summary bar chart with bootstrap error bars on every method ----
    std::cout << "\n--- Bootstrapping all estimators on the full sample (for error bars) ---\n";
    AllEstimatorsResult fullBoot = bootstrapAllEstimators(data.nA, data.nB, alphabet, alphabet,
                                                           nBootCompare, nShufflesInner, rng);
    plotSummaryBarChart(naive.I, fullBoot.I_naive_err, I_mm, fullBoot.I_mm_err,
                         I_sh, fullBoot.I_shuffle_err, nsb.I, fullBoot.I_nsb_err,
                         SBA_nsb, bootSigma, outPrefix);

    // ---- multiplicity distribution diagnostics ----
    std::cout << "\n--- i) Multiplicity distribution shape (vs. NBD assumption) ---\n";
    plotMultiplicityDistributions(data.nA, data.nB, outPrefix);
    plotJointMultiplicityDistribution(data.nA, data.nB, outPrefix);

    // ---- i) jet pT distributions ----
    std::cout << "\n--- i) Jet pT distributions ---\n";
    plotPtDistributions(data.jetPt, data.jet2Pt, outPrefix);
    plotJointPtDistribution(data.jetPt, data.jet2Pt, outPrefix);

    // ---- DIAGNOSTIC: raw multiplicity-vs-pT correlation, bypassing all
    // entropy/NSB/binning machinery. Look at this BEFORE trusting anything
    // downstream that bins by pT.
    std::cout << "\n--- DIAGNOSTIC: raw multiplicity vs. pT profile (bypasses NSB/binning) ---\n";
    plotMultiplicityVsPtProfile(data.jetPt, data.jet2Pt, data.nA, data.nB, outPrefix);

    // ---- pT-differential ----
    std::cout << "\n--- ii)-iv) Differential vs. leading-jet pT (" << ptBins << " quantile bins) ---\n";
    if (compareMethods)
        std::cout << "  (multi-method comparison enabled: " << ptBins << " bins x "
                  << nBootCompare << " bootstrap resamples x 4 methods -- "
                  << "this is the slowest part of the analysis)\n";
    ptDifferentialAnalysis(data.nA, data.nB, data.jetPt, alphabet, alphabet, ptBins, outPrefix,
                            compareMethods, nBootCompare, nShufflesInner, &rng);

    std::cout << "\nDone.\n";
}
