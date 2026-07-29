// closure_test_mixed_pt.C
//
// pT-DIFFERENTIAL closure test for the MIXED-EVENT SUBTRACTION method
// (ROOT version) -- see closure_test_mixed_pt.py for the full design
// rationale. Summary: for each leading-jet-pT bin, and for BOTH
// scenarios (null s=0, injected s=signalStrength), fit a BIN-SPECIFIC
// lambda_j(pT)=a_j+b_j ln(pT) to that bin's own (n,pT) pairs, resample
// kinematics from that bin's own real (pT1,pT2) pairs (not the whole
// sample's), build a bin-specific pseudo-truth from a large reference
// sample, and run nRepeats closure repeats at that bin's actual N, each
// with the full joint-bootstrap machinery. Reports and plots the
// NSB-difference pull mean/std vs. pT, one column per scenario.
//
// Self-contained, all symbols "cmp"-prefixed (loadable alongside
// closure_test_mixed.C or any other macro in this project without
// collisions).
//
// Usage (ACLiC):
//   root -l -q 'closure_test_mixed_pt.C+("file.root","truth",80,10,0.15,4,"quantile",300,20,20,15,500000,"figs")'
// Arguments: (rootFile, level, alphabet, mixNeighbors, signalStrength,
//             ptBins, ptBinMode, minEvents, nRepeats, nBoot, nShuffles,
//             refN, figsDir)
//
// Same honest caveat as always: written against documented ROOT APIs,
// not compiled here (no ROOT in this environment) -- send compiler
// output if anything doesn't build.

#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TH1D.h>
#include <TGraphErrors.h>
#include <TGraph.h>
#include <TMultiGraph.h>
#include <TF1.h>
#include <TLegend.h>
#include <TLine.h>
#include <TRandom3.h>
#include <TMath.h>
#include <TMinuit.h>
#include <TString.h>
#include <TSystem.h>
#include <TStyle.h>
#include <Math/DistFunc.h>

#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

// =============================================================================
// Special functions (identical to analyze_dijet_entropy.C; validated
// against scipy.special there before use).
// =============================================================================

double cmpDigamma(double x) {
    double result = 0.0;
    while (x < 6.0) { result -= 1.0/x; x += 1.0; }
    double f = 1.0/(x*x);
    result += std::log(x) - 0.5/x
        - f*(1.0/12.0 - f*(1.0/120.0 - f*(1.0/252.0 - f*(1.0/240.0 - f*(1.0/132.0)))));
    return result;
}

double cmpTrigamma(double x) {
    double result = 0.0;
    while (x < 6.0) { result += 1.0/(x*x); x += 1.0; }
    double ix2 = 1.0/(x*x);
    result += 1.0/x + 0.5*ix2 + ix2/x*(1.0/6.0 - ix2*(1.0/30.0 - ix2*(1.0/42.0 - ix2/30.0)));
    return result;
}

// =============================================================================
// Entropy / MI estimators (subset needed here; see analyze_dijet_entropy.C
// for the full, documented versions -- kept terse here to keep this file
// a manageable size).
// =============================================================================

double cmpShannonEntropy(const std::vector<int>& counts) {
    double N = 0; for (int c : counts) N += c;
    double S = 0.0;
    for (int c : counts) { if (c<=0) continue; double p=c/N; S -= p*std::log(p); }
    return S;
}

struct CmpHist { std::vector<int> ca, cb, cab; int KA,KB,KAB; long long N; };

CmpHist cmpBuildHist(const std::vector<int>& nA, const std::vector<int>& nB) {
    std::map<int,int> mA, mB; std::map<std::pair<int,int>,int> mAB;
    for (size_t i=0;i<nA.size();++i){ mA[nA[i]]++; mB[nB[i]]++; mAB[{nA[i],nB[i]}]++; }
    CmpHist h;
    for (auto& kv: mA) h.ca.push_back(kv.second);
    for (auto& kv: mB) h.cb.push_back(kv.second);
    for (auto& kv: mAB) h.cab.push_back(kv.second);
    h.KA=h.ca.size(); h.KB=h.cb.size(); h.KAB=h.cab.size(); h.N=nA.size();
    return h;
}

double cmpNaiveI(const std::vector<int>& nA, const std::vector<int>& nB) {
    CmpHist h = cmpBuildHist(nA, nB);
    return cmpShannonEntropy(h.ca) + cmpShannonEntropy(h.cb) - cmpShannonEntropy(h.cab);
}

double cmpMillerMadowI(const std::vector<int>& nA, const std::vector<int>& nB) {
    CmpHist h = cmpBuildHist(nA, nB);
    double I = cmpShannonEntropy(h.ca) + cmpShannonEntropy(h.cb) - cmpShannonEntropy(h.cab);
    double bias = (h.KAB - h.KA - h.KB + 1) / (2.0*h.N);
    return I - bias;
}

double cmpShuffleI(const std::vector<int>& nA, const std::vector<int>& nB, int nShuffles, TRandom3& rng) {
    double I0 = cmpNaiveI(nA, nB);
    std::vector<int> nBs = nB;
    double sum = 0;
    for (int s=0; s<nShuffles; ++s) {
        for (int i=(int)nBs.size()-1;i>0;--i) { int j=rng.Integer(i+1); std::swap(nBs[i],nBs[j]); }
        sum += cmpNaiveI(nA, nBs);
    }
    return I0 - sum/nShuffles;
}

// NSB mean-only entropy (fast: skips the Wolpert-Wolf variance terms,
// since the closure test gets its uncertainty from bootstrapping the
// composite I/S(B|A) directly -- see cmpNsbAll below).
double cmpNsbEntropyMean(const std::vector<int>& countsNonzero, long long K) {
    long long N=0; for (int c: countsNonzero) N += c;
    long long Kobs = countsNonzero.size();
    long long Kzero = K - Kobs;
    if (Kzero < 0) { std::cerr << "ERROR: alphabet too small\n"; return 0.0; }

    static std::vector<double> betaGrid;
    if (betaGrid.empty()) {
        int nb=300; double lo=-6.0, hi=3.0;
        betaGrid.resize(nb);
        for (int i=0;i<nb;++i) betaGrid[i] = std::pow(10.0, lo + i*(hi-lo)/(nb-1));
    }
    int nb = betaGrid.size();
    std::vector<double> means(nb), logP(nb), rho(nb);
    for (int ib=0; ib<nb; ++ib) {
        double beta = betaGrid[ib];
        double A = N + K*beta;
        double sumTerm = 0.0;
        for (int c : countsNonzero) { double a=c+beta; sumTerm += a*cmpDigamma(a+1.0); }
        sumTerm += Kzero*beta*cmpDigamma(beta+1.0);
        means[ib] = cmpDigamma(A+1.0) - sumTerm/A;

        double lp = TMath::LnGamma(K*beta) - TMath::LnGamma(N+K*beta) + TMath::LnGamma(N+1.0);
        for (int c: countsNonzero) lp += TMath::LnGamma(c+beta) - TMath::LnGamma(beta);
        for (int c: countsNonzero) lp -= TMath::LnGamma(c+1.0);
        logP[ib] = lp;
        rho[ib] = K*cmpTrigamma(K*beta+1.0) - cmpTrigamma(beta+1.0);
    }
    double maxLogW=-1e300; std::vector<double> logw(nb);
    for (int ib=0; ib<nb; ++ib) { logw[ib]=logP[ib]+std::log(rho[ib]); if (logw[ib]>maxLogW) maxLogW=logw[ib]; }
    std::vector<double> w(nb), t(nb), wt(nb);
    for (int ib=0; ib<nb; ++ib) { w[ib]=std::exp(logw[ib]-maxLogW); t[ib]=std::log(betaGrid[ib]); wt[ib]=w[ib]*betaGrid[ib]; }
    auto trapz=[&](const std::vector<double>& y){ double s=0; for(int i=0;i<nb-1;++i) s+=0.5*(y[i]+y[i+1])*(t[i+1]-t[i]); return s; };
    double Z = trapz(wt);
    std::vector<double> wtMean(nb); for(int ib=0;ib<nb;++ib) wtMean[ib]=wt[ib]*means[ib];
    return trapz(wtMean)/Z;
}

struct CmpNsbAllResult { double SA,SB,SAB,I,I_err,SBA,SBA_err,NMI,NMI_err; };

CmpNsbAllResult cmpNsbAll(const std::vector<int>& nA, const std::vector<int>& nB,
                     long long alphaA, long long alphaB, int nBoot, TRandom3& rng) {
    CmpHist h0 = cmpBuildHist(nA, nB);
    long long Kj = alphaA*alphaB;
    double SA = cmpNsbEntropyMean(h0.ca, alphaA);
    double SB = cmpNsbEntropyMean(h0.cb, alphaB);
    double SAB = cmpNsbEntropyMean(h0.cab, Kj);

    int N = nA.size();
    std::vector<double> Iboot(nBoot), SBAboot(nBoot), NMIboot(nBoot);
    for (int b=0; b<nBoot; ++b) {
        std::vector<int> rA(N), rB(N);
        for (int i=0;i<N;++i){ int idx=rng.Integer(N); rA[i]=nA[idx]; rB[i]=nB[idx]; }
        CmpHist hb = cmpBuildHist(rA, rB);
        double sA = cmpNsbEntropyMean(hb.ca, alphaA);
        double sB = cmpNsbEntropyMean(hb.cb, alphaB);
        double sAB = cmpNsbEntropyMean(hb.cab, Kj);
        Iboot[b] = sA+sB-sAB;
        SBAboot[b] = sAB - sA;
        NMIboot[b] = Iboot[b] / std::min(sA, sB);
    }
    auto stdv=[&](std::vector<double>& v){ double m=0; for(double x:v) m+=x; m/=v.size();
        double s=0; for(double x:v) s+=(x-m)*(x-m); s/=(v.size()-1); return std::sqrt(s); };

    CmpNsbAllResult r;
    r.SA=SA; r.SB=SB; r.SAB=SAB;
    r.I = SA+SB-SAB; r.I_err = stdv(Iboot);
    r.SBA = SAB-SA; r.SBA_err = stdv(SBAboot);
    // NMI = I / min(S_A,S_B): scale-comparable version of MI (classically
    // in [0,1]; NMI>1 is the same classical-bound violation as
    // S(B|A)<0). Point estimate and bootstrap uncertainty both computed
    // directly on the ratio (reusing the same resamples as I/SBA above),
    // same reasoning as the Python version.
    r.NMI = r.I / std::min(SA, SB); r.NMI_err = stdv(NMIboot);
    return r;
}

// =============================================================================
// Point estimates for all four methods on one pair sample
// =============================================================================

struct CmpAllPoints { double naive, mm, shuffle, nsb; };

CmpAllPoints cmpAllPointEstimates(const std::vector<int>& nA, const std::vector<int>& nB,
                                 long long alphabet, int nShuffles, TRandom3& rng) {
    CmpHist h = cmpBuildHist(nA, nB);
    double S_A = cmpShannonEntropy(h.ca), S_B = cmpShannonEntropy(h.cb), S_AB = cmpShannonEntropy(h.cab);
    CmpAllPoints r;
    r.naive = S_A + S_B - S_AB;
    r.mm = r.naive - (h.KAB - h.KA - h.KB + 1) / (2.0 * h.N);
    r.shuffle = cmpShuffleI(nA, nB, nShuffles, rng);
    long long Kj = alphabet * alphabet;
    r.nsb = cmpNsbEntropyMean(h.ca, alphabet) + cmpNsbEntropyMean(h.cb, alphabet)
          - cmpNsbEntropyMean(h.cab, Kj);
    return r;
}

// =============================================================================
// Mixed-partner construction: k-nearest-neighbor matching in pT2
// =============================================================================

// Returns idx such that mixed pairs are (nA[i], nB[idx[i]]). Also fills
// meanDpt with the achieved mean |pT2_i - pT2_idx[i]| (matching quality).
std::vector<int> cmpBuildPartnerIdx(const std::vector<double>& pt2, int kNeighbors,
                                    TRandom3& rng, double& meanDpt) {
    int n = pt2.size();
    std::vector<int> order(n), rank(n), idx(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return pt2[a] < pt2[b]; });
    for (int r = 0; r < n; ++r) rank[order[r]] = r;
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        int off = (int)rng.Integer(kNeighbors) + 1;
        if (rng.Uniform() < 0.5) off = -off;
        int pr = rank[i] + off;
        if (pr < 0) pr = 0;
        if (pr > n-1) pr = n-1;
        if (pr == rank[i]) pr += (rank[i] < n/2) ? 1 : -1;
        idx[i] = order[pr];
        sum += std::abs(pt2[i] - pt2[idx[i]]);
    }
    meanDpt = sum / n;
    return idx;
}

// =============================================================================
// Calibrated generative model
// =============================================================================

// Least-squares fit of <N> = a + b ln(pT) on the raw event list.
void cmpFitLambdaVsPt(const std::vector<int>& n, const std::vector<double>& pt,
                      double& a, double& b) {
    int N = n.size();
    double sx=0, sy=0, sxx=0, sxy=0;
    for (int i = 0; i < N; ++i) {
        double x = std::log(pt[i]), y = n[i];
        sx += x; sy += y; sxx += x*x; sxy += x*y;
    }
    b = (N*sxy - sx*sy) / (N*sxx - sx*sx);
    a = (sy - b*sx) / N;
}

// N_A ~ Poisson(lam1*g), N_B ~ Poisson(lam2*g), g ~ Gamma(1/s^2, s^2)
// shared (mean 1, var s^2); s=0 -> g=1 (conditional independence).
// Gamma sampling via inverse CDF (ROOT::Math::gamma_quantile, MathCore).
void cmpGenerate(const std::vector<double>& pt1, const std::vector<double>& pt2,
                double a1, double b1, double a2, double b2, double s,
                TRandom3& rng, std::vector<int>& nA, std::vector<int>& nB) {
    int N = pt1.size();
    nA.resize(N); nB.resize(N);
    double k = (s > 0) ? 1.0/(s*s) : 0.0, th = (s > 0) ? s*s : 0.0;
    for (int i = 0; i < N; ++i) {
        double g = 1.0;
        if (s > 0) {
            double u = rng.Uniform();
            if (u < 1e-12) u = 1e-12;
            if (u > 1.0-1e-12) u = 1.0-1e-12;
            g = ROOT::Math::gamma_quantile(u, k, th);
        }
        double l1 = std::max(a1 + b1*std::log(pt1[i]), 0.1) * g;
        double l2 = std::max(a2 + b2*std::log(pt2[i]), 0.1) * g;
        nA[i] = rng.Poisson(l1);
        nB[i] = rng.Poisson(l2);
    }
}

// =============================================================================
// One repeat: difference with joint bootstrap (quiet)
// =============================================================================

void cmpDiffJoint(const std::vector<int>& nA, const std::vector<int>& nB,
                 const std::vector<double>& pt2, long long alphabet,
                 int kNeighbors, int nShuffles, int nBoot, TRandom3& rng,
                 CmpAllPoints& diff, CmpAllPoints& err) {
    int N = nA.size();
    double dd;
    std::vector<int> idx = cmpBuildPartnerIdx(pt2, kNeighbors, rng, dd);
    std::vector<int> nBm(N);
    for (int i = 0; i < N; ++i) nBm[i] = nB[idx[i]];
    CmpAllPoints s1 = cmpAllPointEstimates(nA, nB, alphabet, nShuffles, rng);
    CmpAllPoints s2 = cmpAllPointEstimates(nA, nBm, alphabet, nShuffles, rng);
    diff.naive = s1.naive - s2.naive; diff.mm = s1.mm - s2.mm;
    diff.shuffle = s1.shuffle - s2.shuffle; diff.nsb = s1.nsb - s2.nsb;

    std::vector<double> bN(nBoot), bM(nBoot), bS(nBoot), bB(nBoot);
    for (int b = 0; b < nBoot; ++b) {
        std::vector<int> ra(N), rb(N); std::vector<double> rp(N);
        for (int i = 0; i < N; ++i) {
            int q = rng.Integer(N); ra[i] = nA[q]; rb[i] = nB[q]; rp[i] = pt2[q];
        }
        std::vector<int> ridx = cmpBuildPartnerIdx(rp, kNeighbors, rng, dd);
        std::vector<int> rbm(N);
        for (int i = 0; i < N; ++i) rbm[i] = rb[ridx[i]];
        CmpAllPoints x1 = cmpAllPointEstimates(ra, rb, alphabet, nShuffles, rng);
        CmpAllPoints x2 = cmpAllPointEstimates(ra, rbm, alphabet, nShuffles, rng);
        bN[b] = x1.naive - x2.naive; bM[b] = x1.mm - x2.mm;
        bS[b] = x1.shuffle - x2.shuffle; bB[b] = x1.nsb - x2.nsb;
    }
    auto stdv = [](std::vector<double>& v){ double m=0; for(double x:v) m+=x; m/=v.size();
        double sq=0; for(double x:v) sq+=(x-m)*(x-m); return std::sqrt(sq/(v.size()-1)); };
    err.naive = stdv(bN); err.mm = stdv(bM); err.shuffle = stdv(bS); err.nsb = stdv(bB);
}

// =============================================================================
// Quantile pT bin edges (same convention as elsewhere in this project)
// =============================================================================

std::vector<double> cmpQuantileBinEdgesPt(std::vector<double> pt, int nBins) {
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

// =============================================================================
// One (bin, scenario): bin-specific fit + bin-specific truth + closure
// =============================================================================

struct CmpBinResult {
    double ptLo, ptHi, ptMid; int N; TString scenario;
    double diffTrue, pullMean, pullStd;
};

CmpBinResult cmpAnalyzeOneBinScenario(double lo, double hi,
        const std::vector<int>& naBin, const std::vector<int>& nbBin,
        const std::vector<double>& pt1Bin, const std::vector<double>& pt2Bin,
        double s, long long alphabet, int kNeighbors, int nShuffles, int nBoot,
        int nRepeats, int refN, TRandom3& rng, const TString& tag) {
    int nBin = naBin.size();
    printf("\n%s\npT bin [%.1f,%.1f) GeV, N=%d, scenario=%s\n%s\n",
           std::string(70,'-').c_str(), lo, hi, nBin, tag.Data(), std::string(70,'-').c_str());

    printf("  fitting bin-specific lambda(pT) = a + b ln(pT) ...\n");
    double a1, b1, a2, b2;
    cmpFitLambdaVsPt(naBin, pt1Bin, a1, b1);
    cmpFitLambdaVsPt(nbBin, pt2Bin, a2, b2);
    printf("    jet A: a=%.3f b=%.3f   jet B: a=%.3f b=%.3f\n", a1, b1, a2, b2);

    // ---- bin-specific pseudo-truth ----
    int refNBin = std::min(refN, std::max(50000, nBin * 20));
    std::vector<double> p1r(refNBin), p2r(refNBin);
    for (int i = 0; i < refNBin; ++i) {
        int q = rng.Integer(nBin); p1r[i] = pt1Bin[q]; p2r[i] = pt2Bin[q];
    }
    std::vector<int> naR, nbR;
    cmpGenerate(p1r, p2r, a1, b1, a2, b2, s, rng, naR, nbR);
    double ISameRef = cmpNaiveI(naR, nbR);
    double dd;
    std::vector<int> idxR = cmpBuildPartnerIdx(p2r, kNeighbors, rng, dd);
    std::vector<int> nbRm(refNBin);
    for (int i = 0; i < refNBin; ++i) nbRm[i] = nbR[idxR[i]];
    double IMixRef = cmpNaiveI(naR, nbRm);
    double diffTrue = ISameRef - IMixRef;
    printf("    pseudo-truth (ref N=%d): diff_true = %+.5f nats\n", refNBin, diffTrue);

    // ---- closure loop at this bin's actual N ----
    std::vector<double> diffs(nRepeats), errs(nRepeats);
    for (int rep = 0; rep < nRepeats; ++rep) {
        std::vector<double> p1(nBin), p2(nBin);
        for (int i = 0; i < nBin; ++i) {
            int q = rng.Integer(nBin); p1[i] = pt1Bin[q]; p2[i] = pt2Bin[q];
        }
        std::vector<int> na, nb;
        cmpGenerate(p1, p2, a1, b1, a2, b2, s, rng, na, nb);
        CmpAllPoints diff, err;
        cmpDiffJoint(na, nb, p2, alphabet, kNeighbors, nShuffles, nBoot, rng, diff, err);
        diffs[rep] = diff.nsb; errs[rep] = err.nsb;
    }

    std::vector<double> pulls(nRepeats);
    for (int r = 0; r < nRepeats; ++r) pulls[r] = (diffs[r] - diffTrue) / errs[r];
    double pm = 0; for (double x : pulls) pm += x; pm /= nRepeats;
    double ps = 0; for (double x : pulls) ps += (x-pm)*(x-pm); ps = std::sqrt(ps/(nRepeats-1));

    printf("    NSB-diff pull: mean=%+.3f  std=%.3f\n", pm, ps);
    if (std::abs(ps - 1.0) > 0.3) {
        const char* d = (ps > 1.3) ? "UNDER-covering" : "OVER-covering";
        printf("    NOTE: pull std deviates from 1 by >0.3 -> %s in this bin/scenario.\n", d);
    }

    CmpBinResult out;
    out.ptLo = lo; out.ptHi = hi;
    double ptSum = 0; for (double x : pt1Bin) ptSum += x;
    out.ptMid = ptSum / nBin; out.N = nBin; out.scenario = tag;
    out.diffTrue = diffTrue; out.pullMean = pm; out.pullStd = ps;
    return out;
}

// =============================================================================
// Summary table + plot
// =============================================================================

void cmpPrintSummaryTable(const std::vector<CmpBinResult>& results) {
    printf("\n%s\npT-DIFFERENTIAL CLOSURE TEST SUMMARY -- mixed-event difference (NSB)\n%s\n",
           std::string(92,'=').c_str(), std::string(92,'=').c_str());
    printf("%16s | %7s | %8s | %10s | %10s | %9s\n",
           "pT range", "N", "scenario", "diff_true", "pull mean", "pull std");
    std::cout << std::string(92,'-') << "\n";
    for (const auto& r : results) {
        printf("[%6.1f,%6.1f) | %7d | %8s | %+10.5f | %+10.3f | %9.3f\n",
               r.ptLo, r.ptHi, r.N, r.scenario.Data(), r.diffTrue, r.pullMean, r.pullStd);
    }
}

void cmpPlotSummary(const std::vector<CmpBinResult>& results, const TString& outPrefix) {
    TCanvas* c = new TCanvas("cCmpSummary", "pT-differential mixed closure", 1100, 800);
    c->Divide(2, 2);

    TString scens[2] = {"null", "signal"};
    TString labels[2] = {"NULL (s=0)", "INJECTED"};
    int colors[2] = {kViolet+1, kRed+1};

    for (int col = 0; col < 2; ++col) {
        std::vector<double> pt, mean, std_;
        for (const auto& r : results) {
            if (r.scenario == scens[col]) {
                pt.push_back(r.ptMid); mean.push_back(r.pullMean); std_.push_back(r.pullStd);
            }
        }
        if (pt.empty()) continue;
        int n = pt.size();
        std::vector<double> zero(n, 0.0);

        c->cd(col + 1);
        TGraph* gMean = new TGraph(n, pt.data(), mean.data());
        gMean->SetMarkerStyle(20); gMean->SetMarkerColor(colors[col]); gMean->SetLineColor(colors[col]);
        gMean->SetTitle(Form("%s: pull mean vs. p_{T} (should be ~0);leading jet p_{T} (GeV);NSB-diff pull mean", labels[col].Data()));
        gMean->Draw("APL");
        TLine* l0 = new TLine(gPad->GetUxmin(), 0, gPad->GetUxmax(), 0);
        l0->SetLineColor(kGray+1); l0->Draw();

        c->cd(col + 3);
        TGraph* gStd = new TGraph(n, pt.data(), std_.data());
        gStd->SetMarkerStyle(20); gStd->SetMarkerColor(colors[col]); gStd->SetLineColor(colors[col]);
        gStd->SetTitle(Form("%s: pull std vs. p_{T} (>1 under-cov., <1 over-cov.);leading jet p_{T} (GeV);NSB-diff pull std", labels[col].Data()));
        gStd->Draw("APL");
        TLine* l1 = new TLine(gPad->GetUxmin(), 1, gPad->GetUxmax(), 1);
        l1->SetLineColor(kRed+1); l1->SetLineStyle(2); l1->SetLineWidth(2); l1->Draw();
    }

    c->SaveAs(outPrefix + "_closure_mixed_pt_summary.png");
    std::cout << "\n  saved " << outPrefix << "_closure_mixed_pt_summary.png\n";
}

// =============================================================================
// Data loading + main
// =============================================================================

void cmpLoadTree(const char* rootFile, const TString& level,
                 std::vector<int>& nA, std::vector<int>& nB,
                 std::vector<double>& pt1, std::vector<double>& pt2) {
    TFile* f = TFile::Open(rootFile, "READ");
    if (!f || f->IsZombie()) { std::cerr << "ERROR: could not open " << rootFile << "\n"; return; }
    TTree* tree = (TTree*)f->Get("dijet");
    TString suffix = (level == "det") ? "_det" : "_truth";
    Int_t j1m, j2m, hasDet = 1; Double_t j1pt, j2pt;
    tree->SetBranchAddress(Form("jet1_mult%s", suffix.Data()), &j1m);
    tree->SetBranchAddress(Form("jet2_mult%s", suffix.Data()), &j2m);
    tree->SetBranchAddress(Form("jet1_pt%s", suffix.Data()), &j1pt);
    tree->SetBranchAddress(Form("jet2_pt%s", suffix.Data()), &j2pt);
    if (level == "det") tree->SetBranchAddress("hasDetDijet", &hasDet);
    for (Long64_t i = 0; i < tree->GetEntries(); ++i) {
        tree->GetEntry(i);
        if (level == "det" && hasDet != 1) continue;
        nA.push_back(j1m); nB.push_back(j2m); pt1.push_back(j1pt); pt2.push_back(j2pt);
    }
    f->Close();
}

void closure_test_mixed_pt(const char* rootFile, const char* levelIn = "truth",
        long long alphabet = 80, int mixNeighbors = 10, double signalStrength = 0.15,
        int ptBins = 4, const char* ptBinMode = "quantile", int minEvents = 300,
        int nRepeats = 20, int nBoot = 20, int nShuffles = 15, int refN = 500000,
        const char* figsDir = "/Users/zhoudunmingtu/bnl_work/Work/MODELS/EBCs_toyMC/analysis/figs") {
    TString level(levelIn);
    gStyle->SetPadLeftMargin(0.15); gStyle->SetPadRightMargin(0.06);
    gStyle->SetPadBottomMargin(0.13); gStyle->SetPadTopMargin(0.1);
    gStyle->SetTitleOffset(1.5, "Y"); gStyle->SetTitleOffset(1.1, "X");
    gStyle->SetTitleSize(0.045, "XYZ"); gStyle->SetLabelSize(0.035, "XYZ");

    TString figsDirStr(figsDir);
    gSystem->mkdir(figsDirStr, kTRUE);
    TString outPrefix = figsDirStr + TString::Format("/closure_mixed_pt_root_%s", levelIn);
    std::cout << "Figures will be saved under: " << figsDirStr << "\n";

    TRandom3 rng(20260728);
    std::vector<int> nA, nB; std::vector<double> pt1, pt2;
    cmpLoadTree(rootFile, level, nA, nB, pt1, pt2);
    std::cout << "Loaded " << nA.size() << " events.\n";

    std::vector<double> edges = cmpQuantileBinEdgesPt(pt1, ptBins);
    std::cout << "pT bin edges (quantile): ";
    for (double e : edges) std::cout << e << " ";
    std::cout << "\n";

    TString scenTags[2] = {"null", "signal"};
    double scenS[2] = {0.0, signalStrength};

    std::vector<CmpBinResult> results;
    for (int i = 0; i < ptBins; ++i) {
        double lo = edges[i], hi = edges[i+1];
        std::vector<int> ba, bb; std::vector<double> bp1, bp2;
        for (size_t j = 0; j < nA.size(); ++j) {
            bool in = (i < ptBins-1) ? (pt1[j] >= lo && pt1[j] < hi)
                                      : (pt1[j] >= lo && pt1[j] <= hi);
            if (in) { ba.push_back(nA[j]); bb.push_back(nB[j]); bp1.push_back(pt1[j]); bp2.push_back(pt2[j]); }
        }
        if ((int)ba.size() < minEvents) {
            printf("\nSkipping bin [%.1f,%.1f): only %d events (< minEvents=%d)\n",
                   lo, hi, (int)ba.size(), minEvents);
            continue;
        }
        for (int sc = 0; sc < 2; ++sc) {
            CmpBinResult r = cmpAnalyzeOneBinScenario(
                lo, hi, ba, bb, bp1, bp2, scenS[sc], alphabet, mixNeighbors,
                nShuffles, nBoot, nRepeats, refN, rng, scenTags[sc]);
            results.push_back(r);
        }
    }

    if (results.empty()) {
        std::cout << "\nNo bins had enough events -- nothing to summarize.\n";
        return;
    }

    cmpPrintSummaryTable(results);
    cmpPlotSummary(results, outPrefix);
    std::cout << "\nDone.\n";
}
