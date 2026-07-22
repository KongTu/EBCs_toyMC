// closure_test_mixed.C
//
// Closure test for the MIXED-EVENT SUBTRACTION method (ROOT version) --
// see closure_test_mixed.py for the full design rationale. Summary:
// kinematics (pT1,pT2) resampled directly from the real data; per-jet
// lambda_j(pT) = a_j + b_j ln(pT) fitted to the real profile; genuine
// beyond-kinematics signal injected via a SHARED latent
// g ~ Gamma(1/s^2, s^2) (s=0 -> null, true signal exactly zero).
// Pseudo-truth = huge-N reference sample analyzed with the same
// k-NN-matched difference (naive estimator, bias negligible there).
// Closure = nRepeats fresh draws at the real N, each analyzed with all
// four methods per term + JOINT bootstrap of the difference; pulls
// reported per method. Runs the NULL scenario and one injected-signal
// scenario. Purely diagnostic.
//
// Self-contained, all symbols "cm"-prefixed (loadable alongside the
// other macros without collisions).
//
// Usage (ACLiC):
//   root -l -q 'closure_test_mixed.C+("file.root","truth",80,10,0.15,30,30,15,1000000,-1)'
// Arguments: (rootFile, level, alphabet, mixNeighbors, signalStrength,
//             nRepeats, nBoot, nShuffles, refN, closureN, figsDir)
//   closureN = -1 -> use the real dataset's N per repeat.
//
// Same honest caveat as always: written against documented ROOT APIs,
// not compiled here (no ROOT in this environment) -- send compiler
// output if anything doesn't build.

#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TH1D.h>
#include <TGraphErrors.h>
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

double cmDigamma(double x) {
    double result = 0.0;
    while (x < 6.0) { result -= 1.0/x; x += 1.0; }
    double f = 1.0/(x*x);
    result += std::log(x) - 0.5/x
        - f*(1.0/12.0 - f*(1.0/120.0 - f*(1.0/252.0 - f*(1.0/240.0 - f*(1.0/132.0)))));
    return result;
}

double cmTrigamma(double x) {
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

double cmShannonEntropy(const std::vector<int>& counts) {
    double N = 0; for (int c : counts) N += c;
    double S = 0.0;
    for (int c : counts) { if (c<=0) continue; double p=c/N; S -= p*std::log(p); }
    return S;
}

struct CmHist { std::vector<int> ca, cb, cab; int KA,KB,KAB; long long N; };

CmHist cmBuildHist(const std::vector<int>& nA, const std::vector<int>& nB) {
    std::map<int,int> mA, mB; std::map<std::pair<int,int>,int> mAB;
    for (size_t i=0;i<nA.size();++i){ mA[nA[i]]++; mB[nB[i]]++; mAB[{nA[i],nB[i]}]++; }
    CmHist h;
    for (auto& kv: mA) h.ca.push_back(kv.second);
    for (auto& kv: mB) h.cb.push_back(kv.second);
    for (auto& kv: mAB) h.cab.push_back(kv.second);
    h.KA=h.ca.size(); h.KB=h.cb.size(); h.KAB=h.cab.size(); h.N=nA.size();
    return h;
}

double cmNaiveI(const std::vector<int>& nA, const std::vector<int>& nB) {
    CmHist h = cmBuildHist(nA, nB);
    return cmShannonEntropy(h.ca) + cmShannonEntropy(h.cb) - cmShannonEntropy(h.cab);
}

double cmMillerMadowI(const std::vector<int>& nA, const std::vector<int>& nB) {
    CmHist h = cmBuildHist(nA, nB);
    double I = cmShannonEntropy(h.ca) + cmShannonEntropy(h.cb) - cmShannonEntropy(h.cab);
    double bias = (h.KAB - h.KA - h.KB + 1) / (2.0*h.N);
    return I - bias;
}

double cmShuffleI(const std::vector<int>& nA, const std::vector<int>& nB, int nShuffles, TRandom3& rng) {
    double I0 = cmNaiveI(nA, nB);
    std::vector<int> nBs = nB;
    double sum = 0;
    for (int s=0; s<nShuffles; ++s) {
        for (int i=(int)nBs.size()-1;i>0;--i) { int j=rng.Integer(i+1); std::swap(nBs[i],nBs[j]); }
        sum += cmNaiveI(nA, nBs);
    }
    return I0 - sum/nShuffles;
}

// NSB mean-only entropy (fast: skips the Wolpert-Wolf variance terms,
// since the closure test gets its uncertainty from bootstrapping the
// composite I/S(B|A) directly -- see cmNsbAll below).
double cmNsbEntropyMean(const std::vector<int>& countsNonzero, long long K) {
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
        for (int c : countsNonzero) { double a=c+beta; sumTerm += a*cmDigamma(a+1.0); }
        sumTerm += Kzero*beta*cmDigamma(beta+1.0);
        means[ib] = cmDigamma(A+1.0) - sumTerm/A;

        double lp = TMath::LnGamma(K*beta) - TMath::LnGamma(N+K*beta) + TMath::LnGamma(N+1.0);
        for (int c: countsNonzero) lp += TMath::LnGamma(c+beta) - TMath::LnGamma(beta);
        for (int c: countsNonzero) lp -= TMath::LnGamma(c+1.0);
        logP[ib] = lp;
        rho[ib] = K*cmTrigamma(K*beta+1.0) - cmTrigamma(beta+1.0);
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

struct CmNsbAllResult { double SA,SB,SAB,I,I_err,SBA,SBA_err,NMI,NMI_err; };

CmNsbAllResult cmNsbAll(const std::vector<int>& nA, const std::vector<int>& nB,
                     long long alphaA, long long alphaB, int nBoot, TRandom3& rng) {
    CmHist h0 = cmBuildHist(nA, nB);
    long long Kj = alphaA*alphaB;
    double SA = cmNsbEntropyMean(h0.ca, alphaA);
    double SB = cmNsbEntropyMean(h0.cb, alphaB);
    double SAB = cmNsbEntropyMean(h0.cab, Kj);

    int N = nA.size();
    std::vector<double> Iboot(nBoot), SBAboot(nBoot), NMIboot(nBoot);
    for (int b=0; b<nBoot; ++b) {
        std::vector<int> rA(N), rB(N);
        for (int i=0;i<N;++i){ int idx=rng.Integer(N); rA[i]=nA[idx]; rB[i]=nB[idx]; }
        CmHist hb = cmBuildHist(rA, rB);
        double sA = cmNsbEntropyMean(hb.ca, alphaA);
        double sB = cmNsbEntropyMean(hb.cb, alphaB);
        double sAB = cmNsbEntropyMean(hb.cab, Kj);
        Iboot[b] = sA+sB-sAB;
        SBAboot[b] = sAB - sA;
        NMIboot[b] = Iboot[b] / std::min(sA, sB);
    }
    auto stdv=[&](std::vector<double>& v){ double m=0; for(double x:v) m+=x; m/=v.size();
        double s=0; for(double x:v) s+=(x-m)*(x-m); s/=(v.size()-1); return std::sqrt(s); };

    CmNsbAllResult r;
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

struct CmAllPoints { double naive, mm, shuffle, nsb; };

CmAllPoints cmAllPointEstimates(const std::vector<int>& nA, const std::vector<int>& nB,
                                 long long alphabet, int nShuffles, TRandom3& rng) {
    CmHist h = cmBuildHist(nA, nB);
    double S_A = cmShannonEntropy(h.ca), S_B = cmShannonEntropy(h.cb), S_AB = cmShannonEntropy(h.cab);
    CmAllPoints r;
    r.naive = S_A + S_B - S_AB;
    r.mm = r.naive - (h.KAB - h.KA - h.KB + 1) / (2.0 * h.N);
    r.shuffle = cmShuffleI(nA, nB, nShuffles, rng);
    long long Kj = alphabet * alphabet;
    r.nsb = cmNsbEntropyMean(h.ca, alphabet) + cmNsbEntropyMean(h.cb, alphabet)
          - cmNsbEntropyMean(h.cab, Kj);
    return r;
}

// =============================================================================
// Mixed-partner construction: k-nearest-neighbor matching in pT2
// =============================================================================

// Returns idx such that mixed pairs are (nA[i], nB[idx[i]]). Also fills
// meanDpt with the achieved mean |pT2_i - pT2_idx[i]| (matching quality).
std::vector<int> cmBuildPartnerIdx(const std::vector<double>& pt2, int kNeighbors,
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
void cmFitLambdaVsPt(const std::vector<int>& n, const std::vector<double>& pt,
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
void cmGenerate(const std::vector<double>& pt1, const std::vector<double>& pt2,
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

void cmDiffJoint(const std::vector<int>& nA, const std::vector<int>& nB,
                 const std::vector<double>& pt2, long long alphabet,
                 int kNeighbors, int nShuffles, int nBoot, TRandom3& rng,
                 CmAllPoints& diff, CmAllPoints& err) {
    int N = nA.size();
    double dd;
    std::vector<int> idx = cmBuildPartnerIdx(pt2, kNeighbors, rng, dd);
    std::vector<int> nBm(N);
    for (int i = 0; i < N; ++i) nBm[i] = nB[idx[i]];
    CmAllPoints s1 = cmAllPointEstimates(nA, nB, alphabet, nShuffles, rng);
    CmAllPoints s2 = cmAllPointEstimates(nA, nBm, alphabet, nShuffles, rng);
    diff.naive = s1.naive - s2.naive; diff.mm = s1.mm - s2.mm;
    diff.shuffle = s1.shuffle - s2.shuffle; diff.nsb = s1.nsb - s2.nsb;

    std::vector<double> bN(nBoot), bM(nBoot), bS(nBoot), bB(nBoot);
    for (int b = 0; b < nBoot; ++b) {
        std::vector<int> ra(N), rb(N); std::vector<double> rp(N);
        for (int i = 0; i < N; ++i) {
            int q = rng.Integer(N); ra[i] = nA[q]; rb[i] = nB[q]; rp[i] = pt2[q];
        }
        std::vector<int> ridx = cmBuildPartnerIdx(rp, kNeighbors, rng, dd);
        std::vector<int> rbm(N);
        for (int i = 0; i < N; ++i) rbm[i] = rb[ridx[i]];
        CmAllPoints x1 = cmAllPointEstimates(ra, rb, alphabet, nShuffles, rng);
        CmAllPoints x2 = cmAllPointEstimates(ra, rbm, alphabet, nShuffles, rng);
        bN[b] = x1.naive - x2.naive; bM[b] = x1.mm - x2.mm;
        bS[b] = x1.shuffle - x2.shuffle; bB[b] = x1.nsb - x2.nsb;
    }
    auto stdv = [](std::vector<double>& v){ double m=0; for(double x:v) m+=x; m/=v.size();
        double sq=0; for(double x:v) sq+=(x-m)*(x-m); return std::sqrt(sq/(v.size()-1)); };
    err.naive = stdv(bN); err.mm = stdv(bM); err.shuffle = stdv(bS); err.nsb = stdv(bB);
}

// =============================================================================
// One scenario: truth + closure + report + plot
// =============================================================================

void cmRunScenario(double s, const std::vector<double>& pt1All,
                   const std::vector<double>& pt2All, double a1, double b1,
                   double a2, double b2, int NClosure, long long alphabet,
                   int kNeighbors, int nShuffles, int nBoot, int nRepeats,
                   int refN, TRandom3& rng, const TString& outPrefix) {
    TString tag = (s == 0) ? "NULL (s=0, true signal = 0)"
                            : TString::Format("INJECTED (s=%g)", s);
    std::cout << "\n" << std::string(72,'=') << "\nSCENARIO: " << tag
              << "\n" << std::string(72,'=') << "\n";

    // ---- pseudo-truth ----
    std::cout << "  Pseudo-truth: reference sample N=" << refN << "...\n";
    int Nall = pt1All.size();
    std::vector<double> p1r(refN), p2r(refN);
    for (int i = 0; i < refN; ++i) {
        int q = rng.Integer(Nall); p1r[i] = pt1All[q]; p2r[i] = pt2All[q];
    }
    std::vector<int> naR, nbR;
    cmGenerate(p1r, p2r, a1, b1, a2, b2, s, rng, naR, nbR);
    double ISameRef = cmNaiveI(naR, nbR);
    double dd;
    std::vector<int> idxR = cmBuildPartnerIdx(p2r, kNeighbors, rng, dd);
    std::vector<int> nbRm(refN);
    for (int i = 0; i < refN; ++i) nbRm[i] = nbR[idxR[i]];
    double IMixRef = cmNaiveI(naR, nbRm);
    double diffTrue = ISameRef - IMixRef;
    printf("  I_same(ref)=%.5f  I_mix(ref)=%.5f  ->  diff_true = %+.5f nats\n",
           ISameRef, IMixRef, diffTrue);

    // ---- closure loop ----
    printf("  Closure: %d repeats at N=%d (joint bootstrap x%d per repeat)...\n",
           nRepeats, NClosure, nBoot);
    std::vector<CmAllPoints> diffs(nRepeats), errs(nRepeats);
    for (int rep = 0; rep < nRepeats; ++rep) {
        std::vector<double> p1(NClosure), p2(NClosure);
        for (int i = 0; i < NClosure; ++i) {
            int q = rng.Integer(Nall); p1[i] = pt1All[q]; p2[i] = pt2All[q];
        }
        std::vector<int> na, nb;
        cmGenerate(p1, p2, a1, b1, a2, b2, s, rng, na, nb);
        cmDiffJoint(na, nb, p2, alphabet, kNeighbors, nShuffles, nBoot, rng,
                    diffs[rep], errs[rep]);
        if ((rep+1) % std::max(1, nRepeats/5) == 0)
            printf("    ... %d/%d repeats done\n", rep+1, nRepeats);
    }

    // ---- report ----
    const char* lab[4] = {"naive","Miller-Madow","shuffle","NSB"};
    printf("\n  diff_true = %+.5f nats\n", diffTrue);
    printf("  %14s | %11s | %9s | %9s | %9s | %8s\n", "method", "mean diff",
           "std/reps", "bias", "pull mean", "pull std");
    std::cout << "  " << std::string(74,'-') << "\n";
    std::vector<double> nsbPulls(nRepeats);
    for (int k = 0; k < 4; ++k) {
        std::vector<double> dv(nRepeats), pv(nRepeats);
        for (int r = 0; r < nRepeats; ++r) {
            double d = (k==0)?diffs[r].naive:(k==1)?diffs[r].mm:(k==2)?diffs[r].shuffle:diffs[r].nsb;
            double e = (k==0)?errs[r].naive:(k==1)?errs[r].mm:(k==2)?errs[r].shuffle:errs[r].nsb;
            dv[r] = d; pv[r] = (d - diffTrue) / e;
        }
        double dm=0, pm=0; for (int r=0;r<nRepeats;++r){dm+=dv[r];pm+=pv[r];}
        dm/=nRepeats; pm/=nRepeats;
        double ds=0, ps=0; for (int r=0;r<nRepeats;++r){ds+=(dv[r]-dm)*(dv[r]-dm);ps+=(pv[r]-pm)*(pv[r]-pm);}
        ds=std::sqrt(ds/(nRepeats-1)); ps=std::sqrt(ps/(nRepeats-1));
        printf("  %14s | %+11.5f | %9.5f | %+9.5f | %+9.3f | %8.3f\n",
               lab[k], dm, ds, dm-diffTrue, pm, ps);
        if (k == 3) nsbPulls = pv;
    }
    int in1=0, in2=0;
    for (double p : nsbPulls) { if (std::abs(p)<1) ++in1; if (std::abs(p)<2) ++in2; }
    printf("\n  NSB-diff calibration: %.1f%% within 1 sigma (expect ~68%%), "
           "%.1f%% within 2 sigma (expect ~95%%)\n",
           100.0*in1/nRepeats, 100.0*in2/nRepeats);

    // ---- plot: recovery bars + NSB pull histogram + per-repeat ----
    TString sfx = (s == 0) ? "null" : TString::Format("s%g", s).ReplaceAll(".","p");
    TCanvas* c = new TCanvas(Form("cCmClosure_%s", sfx.Data()),
                              "Mixed-subtraction closure", 1500, 480);
    c->Divide(3, 1);

    c->cd(1);
    TH1D* hRec = new TH1D(Form("hCmRec_%s", sfx.Data()),
        Form("Recovery, %s;;recovered I_{same}-I_{mix} (nats)", tag.Data()), 4, 0, 4);
    for (int k = 0; k < 4; ++k) {
        std::vector<double> dv(nRepeats);
        for (int r = 0; r < nRepeats; ++r)
            dv[r] = (k==0)?diffs[r].naive:(k==1)?diffs[r].mm:(k==2)?diffs[r].shuffle:diffs[r].nsb;
        double m=0; for (double x:dv) m+=x; m/=nRepeats;
        double sd=0; for (double x:dv) sd+=(x-m)*(x-m); sd=std::sqrt(sd/(nRepeats-1));
        hRec->SetBinContent(k+1, m); hRec->SetBinError(k+1, sd);
        hRec->GetXaxis()->SetBinLabel(k+1, lab[k]);
    }
    hRec->SetFillColorAlpha(kAzure+1, 0.6); hRec->SetStats(0);
    {
        double lo = std::min(0.0, diffTrue), hi = std::max(0.0, diffTrue);
        for (int k = 1; k <= 4; ++k) {
            lo = std::min(lo, hRec->GetBinContent(k)-hRec->GetBinError(k));
            hi = std::max(hi, hRec->GetBinContent(k)+hRec->GetBinError(k));
        }
        double r = hi - lo; hRec->SetMinimum(lo-0.15*r); hRec->SetMaximum(hi+0.25*r);
    }
    hRec->Draw("E1 HIST");
    TLine* lt = new TLine(gPad->GetUxmin(), diffTrue, gPad->GetUxmax(), diffTrue);
    lt->SetLineColor(kBlack); lt->SetLineStyle(2); lt->SetLineWidth(2); lt->Draw();
    TLine* l0 = new TLine(gPad->GetUxmin(), 0, gPad->GetUxmax(), 0);
    l0->SetLineColor(kGray+1); l0->Draw();

    c->cd(2);
    TH1D* hPull = new TH1D(Form("hCmPull_%s", sfx.Data()),
        "NSB difference: pull distribution;(#hat{D}-D_{true})/#sigma_{joint};repeats", 12, -4, 4);
    for (double p : nsbPulls) hPull->Fill(p);
    hPull->SetFillColorAlpha(kViolet+1, 0.6); hPull->SetStats(0);
    hPull->Draw("HIST");
    TF1* fG = new TF1(Form("fCmG_%s", sfx.Data()),
        Form("%f*exp(-0.5*x*x)/sqrt(2*pi)", nRepeats*8.0/12.0), -4, 4);
    fG->SetLineColor(kBlack); fG->SetLineStyle(2); fG->Draw("SAME");

    c->cd(3);
    std::vector<double> rx(nRepeats), ry(nRepeats), rye(nRepeats), rz(nRepeats, 0.0);
    for (int r = 0; r < nRepeats; ++r) {
        rx[r] = r; ry[r] = diffs[r].nsb; rye[r] = errs[r].nsb;
    }
    TGraphErrors* gRep = new TGraphErrors(nRepeats, rx.data(), ry.data(), rz.data(), rye.data());
    gRep->SetMarkerStyle(20); gRep->SetMarkerColor(kViolet+1); gRep->SetLineColor(kViolet+1);
    gRep->SetTitle("NSB difference per repeat vs. truth;repeat #;I_{same}-I_{mix} #pm joint-boot #sigma");
    gRep->Draw("AP");
    TLine* lt2 = new TLine(gPad->GetUxmin(), diffTrue, gPad->GetUxmax(), diffTrue);
    lt2->SetLineColor(kBlack); lt2->SetLineStyle(2); lt2->SetLineWidth(2); lt2->Draw();
    TLine* l02 = new TLine(gPad->GetUxmin(), 0, gPad->GetUxmax(), 0);
    l02->SetLineColor(kGray+1); l02->Draw();

    c->SaveAs(outPrefix + "_closure_mixed_" + sfx + ".png");
    std::cout << "  saved " << outPrefix << "_closure_mixed_" << sfx << ".png\n";
}

// =============================================================================
// Data loading + main
// =============================================================================

void cmLoadTree(const char* rootFile, const TString& level,
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

void closure_test_mixed(const char* rootFile, const char* levelIn = "truth",
                         long long alphabet = 80, int mixNeighbors = 10,
                         double signalStrength = 0.15, int nRepeats = 30,
                         int nBoot = 30, int nShuffles = 15, int refN = 1000000,
                         int closureN = -1,
                         const char* figsDir = "/Users/zhoudunmingtu/bnl_work/Work/MODELS/EBCs_toyMC/analysis/figs") {
    TString level(levelIn);
    gStyle->SetPadLeftMargin(0.15); gStyle->SetPadRightMargin(0.06);
    gStyle->SetPadBottomMargin(0.13); gStyle->SetPadTopMargin(0.09);
    gStyle->SetTitleOffset(1.5, "Y"); gStyle->SetTitleOffset(1.1, "X");
    gStyle->SetTitleSize(0.045, "XYZ"); gStyle->SetLabelSize(0.035, "XYZ");

    TString figsDirStr(figsDir);
    gSystem->mkdir(figsDirStr, kTRUE);
    TString outPrefix = figsDirStr + TString::Format("/closure_mixed_root_%s", levelIn);
    std::cout << "Figures will be saved under: " << figsDirStr << "\n";

    TRandom3 rng(20260723);
    std::vector<int> nA, nB; std::vector<double> pt1, pt2;
    cmLoadTree(rootFile, level, nA, nB, pt1, pt2);
    int NReal = nA.size();
    int NClosure = (closureN > 0) ? closureN : NReal;
    std::cout << "Loaded " << NReal << " events; closure repeats at N="
              << NClosure << ".\n";

    std::cout << "\n--- Calibrating lambda(pT) = a + b ln(pT) ---\n";
    double a1, b1, a2, b2;
    cmFitLambdaVsPt(nA, pt1, a1, b1);
    cmFitLambdaVsPt(nB, pt2, a2, b2);
    printf("  jet A: a=%.3f b=%.3f   jet B: a=%.3f b=%.3f\n", a1, b1, a2, b2);

    cmRunScenario(0.0, pt1, pt2, a1, b1, a2, b2, NClosure, alphabet,
                  mixNeighbors, nShuffles, nBoot, nRepeats, refN, rng, outPrefix);
    cmRunScenario(signalStrength, pt1, pt2, a1, b1, a2, b2, NClosure, alphabet,
                  mixNeighbors, nShuffles, nBoot, nRepeats, refN, rng, outPrefix);

    std::cout << "\nDone.\n";
}
