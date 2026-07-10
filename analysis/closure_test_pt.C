// closure_test_pt.C
//
// pT-DIFFERENTIAL Tier-2 closure test (ROOT version) -- see
// closure_test_pt.py for the full design rationale, and closure_test.C
// for the original whole-sample version this extends.
//
// closure_test.C answers "is NSB calibrated for a dataset shaped like my
// WHOLE sample". This answers the sharper question: "is NSB calibrated
// INSIDE EACH leading-jet-pT bin specifically" -- since both the
// underlying multiplicity/correlation structure AND the per-bin
// statistics change with pT (that's the whole point of the differential
// measurement).
//
// For each leading-jet-pT bin (quantile binning, same convention as
// analyze_dijet_entropy.C):
//   1. Select that bin's events only.
//   2. Fit a Gamma-Poisson model to THAT BIN's own marginals/correlation
//      (not the whole-sample fit) -- this is the key upgrade over running
//      closure_test.C once per bin with closureN set to that bin's N,
//      which would reuse the whole-sample model everywhere and miss any
//      real pT-dependence in the correlation structure itself.
//   3. Generate a reference sample from that bin-specific fitted model
//      for a bin-specific pseudo-truth (I_true, S(B|A)_true, NMI_true).
//   4. Run the closure test (NSB via the same bootstrap-of-composite-
//      quantities procedure as closure_test.C) at that bin's ACTUAL N,
//      and record the pull-distribution mean/std for I, S(B|A), NMI.
//
// Output: a console summary table and a 6-panel summary plot (pull mean
// and std vs. pT, for I, S(B|A), NMI) -- directly showing whether NSB's
// reported uncertainty can be trusted bin-by-bin, which is the number
// you actually need before quoting a per-bin significance in the real
// analysis.
//
// Purely diagnostic, same as closure_test.C: no correction is applied
// anywhere, only reported.
//
// Self-contained: duplicates (with renamed symbols, "...Pt" suffix) the
// estimator functions from closure_test.C/analyze_dijet_entropy.C rather
// than depending on cross-macro linkage (fragile in ROOT), so this file
// can be loaded alongside either without symbol collisions.
//
// Gamma-distributed latent variable sampling uses ROOT::Math::gamma_quantile
// (inverse-CDF method) -- confirmed to live in MathCore (no GSL/MathMore
// dependency) before use here.
//
// Usage (ACLiC-compiled -- strongly recommended, this does a LOT of work:
// n_bins times the cost of a single closure_test.C run):
//   root -l -q 'closure_test_pt.C+("pythia_dijet_pp200.root","truth",90,4,100,150,100,1000000,200)'
//
// Arguments: (rootFile, level, alphabet, ptBins, nRepeats, nBoot,
//             nShuffles, refN, minEvents)
//   ptBins    -- number of quantile pT bins (default 4)
//   minEvents -- skip bins with fewer events than this; a closure test
//                needs reasonable statistics just to FIT the per-bin
//                model in the first place (default 200)
//   refN default is smaller here (1,000,000) than closure_test.C's
//   (2,000,000) since this cost multiplies by ptBins.
//
// Same honest caveat as the other macros in this project: written against
// documented ROOT APIs and validated pure-math components, but not
// compiled/run in the environment this was written in (no ROOT there).
// Send compiler output if anything doesn't build.

#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TH1D.h>
#include <TGraph.h>
#include <TGraphErrors.h>
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

double ctPtDigamma(double x) {
    double result = 0.0;
    while (x < 6.0) { result -= 1.0/x; x += 1.0; }
    double f = 1.0/(x*x);
    result += std::log(x) - 0.5/x
        - f*(1.0/12.0 - f*(1.0/120.0 - f*(1.0/252.0 - f*(1.0/240.0 - f*(1.0/132.0)))));
    return result;
}

double ctPtTrigamma(double x) {
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

double ctPtShannonEntropy(const std::vector<int>& counts) {
    double N = 0; for (int c : counts) N += c;
    double S = 0.0;
    for (int c : counts) { if (c<=0) continue; double p=c/N; S -= p*std::log(p); }
    return S;
}

struct CtPtHist { std::vector<int> ca, cb, cab; int KA,KB,KAB; long long N; };

CtPtHist ctPtBuildHist(const std::vector<int>& nA, const std::vector<int>& nB) {
    std::map<int,int> mA, mB; std::map<std::pair<int,int>,int> mAB;
    for (size_t i=0;i<nA.size();++i){ mA[nA[i]]++; mB[nB[i]]++; mAB[{nA[i],nB[i]}]++; }
    CtPtHist h;
    for (auto& kv: mA) h.ca.push_back(kv.second);
    for (auto& kv: mB) h.cb.push_back(kv.second);
    for (auto& kv: mAB) h.cab.push_back(kv.second);
    h.KA=h.ca.size(); h.KB=h.cb.size(); h.KAB=h.cab.size(); h.N=nA.size();
    return h;
}

double naiveIPt(const std::vector<int>& nA, const std::vector<int>& nB) {
    CtPtHist h = ctPtBuildHist(nA, nB);
    return ctPtShannonEntropy(h.ca) + ctPtShannonEntropy(h.cb) - ctPtShannonEntropy(h.cab);
}

double millerMadowIPt(const std::vector<int>& nA, const std::vector<int>& nB) {
    CtPtHist h = ctPtBuildHist(nA, nB);
    double I = ctPtShannonEntropy(h.ca) + ctPtShannonEntropy(h.cb) - ctPtShannonEntropy(h.cab);
    double bias = (h.KAB - h.KA - h.KB + 1) / (2.0*h.N);
    return I - bias;
}

double shuffleIPt(const std::vector<int>& nA, const std::vector<int>& nB, int nShuffles, TRandom3& rng) {
    double I0 = naiveIPt(nA, nB);
    std::vector<int> nBs = nB;
    double sum = 0;
    for (int s=0; s<nShuffles; ++s) {
        for (int i=(int)nBs.size()-1;i>0;--i) { int j=rng.Integer(i+1); std::swap(nBs[i],nBs[j]); }
        sum += naiveIPt(nA, nBs);
    }
    return I0 - sum/nShuffles;
}

// NSB mean-only entropy (fast: skips the Wolpert-Wolf variance terms,
// since the closure test gets its uncertainty from bootstrapping the
// composite I/S(B|A) directly -- see nsbAllPt below).
double nsbEntropyMeanPt(const std::vector<int>& countsNonzero, long long K) {
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
        for (int c : countsNonzero) { double a=c+beta; sumTerm += a*ctPtDigamma(a+1.0); }
        sumTerm += Kzero*beta*ctPtDigamma(beta+1.0);
        means[ib] = ctPtDigamma(A+1.0) - sumTerm/A;

        double lp = TMath::LnGamma(K*beta) - TMath::LnGamma(N+K*beta) + TMath::LnGamma(N+1.0);
        for (int c: countsNonzero) lp += TMath::LnGamma(c+beta) - TMath::LnGamma(beta);
        for (int c: countsNonzero) lp -= TMath::LnGamma(c+1.0);
        logP[ib] = lp;
        rho[ib] = K*ctPtTrigamma(K*beta+1.0) - ctPtTrigamma(beta+1.0);
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

struct NsbAllResultPt { double SA,SB,SAB,I,I_err,SBA,SBA_err,NMI,NMI_err; };

NsbAllResultPt nsbAllPt(const std::vector<int>& nA, const std::vector<int>& nB,
                     long long alphaA, long long alphaB, int nBoot, TRandom3& rng) {
    CtPtHist h0 = ctPtBuildHist(nA, nB);
    long long Kj = alphaA*alphaB;
    double SA = nsbEntropyMeanPt(h0.ca, alphaA);
    double SB = nsbEntropyMeanPt(h0.cb, alphaB);
    double SAB = nsbEntropyMeanPt(h0.cab, Kj);

    int N = nA.size();
    std::vector<double> Iboot(nBoot), SBAboot(nBoot), NMIboot(nBoot);
    for (int b=0; b<nBoot; ++b) {
        std::vector<int> rA(N), rB(N);
        for (int i=0;i<N;++i){ int idx=rng.Integer(N); rA[i]=nA[idx]; rB[i]=nB[idx]; }
        CtPtHist hb = ctPtBuildHist(rA, rB);
        double sA = nsbEntropyMeanPt(hb.ca, alphaA);
        double sB = nsbEntropyMeanPt(hb.cb, alphaB);
        double sAB = nsbEntropyMeanPt(hb.cab, Kj);
        Iboot[b] = sA+sB-sAB;
        SBAboot[b] = sAB - sA;
        NMIboot[b] = Iboot[b] / std::min(sA, sB);
    }
    auto stdv=[&](std::vector<double>& v){ double m=0; for(double x:v) m+=x; m/=v.size();
        double s=0; for(double x:v) s+=(x-m)*(x-m); s/=(v.size()-1); return std::sqrt(s); };

    NsbAllResultPt r;
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
// Step 1: fit the Gamma-Poisson shared-latent model
// =============================================================================

void modelMomentsPt(double k, double theta, double c,
                   double& meanA, double& varA, double& meanB, double& varB, double& corrAB) {
    meanA = k*theta;
    varA = k*theta*(1+theta);
    meanB = c*k*theta;
    varB = c*k*theta*(1+c*theta);
    double covAB = c*k*theta*theta;
    corrAB = covAB/std::sqrt(varA*varB);
}

// TMinuit needs a global (C-style) FCN and globals for the fit targets.
double g_targets_pt[5];  // meanA, varA, meanB, varB, corrAB (empirical)

void fitFCNPt(Int_t& npar, Double_t* gin, Double_t& f, Double_t* par, Int_t iflag) {
    double k = std::exp(par[0]), theta = std::exp(par[1]), c = std::exp(par[2]);
    double meanA, varA, meanB, varB, corrAB;
    modelMomentsPt(k, theta, c, meanA, varA, meanB, varB, corrAB);
    double pred[5] = {meanA, varA, meanB, varB, corrAB};
    double loss = 0;
    for (int i=0;i<4;++i) { double rel=(pred[i]-g_targets_pt[i])/g_targets_pt[i]; loss += rel*rel; }
    double corrErr = pred[4]-g_targets_pt[4];
    loss += corrErr*corrErr;
    f = loss;
}

void fitModelPt(const std::vector<int>& nA, const std::vector<int>& nB,
              double& kFit, double& thetaFit, double& cFit) {
    int N = nA.size();
    double meanA=0, meanB=0;
    for (int i=0;i<N;++i){ meanA+=nA[i]; meanB+=nB[i]; }
    meanA/=N; meanB/=N;
    double varA=0, varB=0, cov=0;
    for (int i=0;i<N;++i){ varA+=(nA[i]-meanA)*(nA[i]-meanA); varB+=(nB[i]-meanB)*(nB[i]-meanB);
                           cov += (nA[i]-meanA)*(nB[i]-meanB); }
    varA/=(N-1); varB/=(N-1); cov/=(N-1);
    double corrAB = cov/std::sqrt(varA*varB);

    g_targets_pt[0]=meanA; g_targets_pt[1]=varA; g_targets_pt[2]=meanB; g_targets_pt[3]=varB; g_targets_pt[4]=corrAB;

    double k0 = (varA>meanA) ? meanA*meanA/(varA-meanA) : 3.0;
    double theta0 = meanA/k0;
    double c0 = meanB/meanA;

    TMinuit minuit(3);
    minuit.SetFCN(fitFCNPt);
    minuit.SetPrintLevel(-1);
    int ierflg = 0;
    minuit.mnparm(0, "logk",     std::log(k0),     0.1, 0, 0, ierflg);
    minuit.mnparm(1, "logtheta", std::log(theta0), 0.1, 0, 0, ierflg);
    minuit.mnparm(2, "logc",     std::log(c0),     0.1, 0, 0, ierflg);
    minuit.Migrad();

    double val, err;
    minuit.GetParameter(0, val, err); kFit = std::exp(val);
    minuit.GetParameter(1, val, err); thetaFit = std::exp(val);
    minuit.GetParameter(2, val, err); cFit = std::exp(val);

    double mA,vA,mB,vB,cAB;
    modelMomentsPt(kFit, thetaFit, cFit, mA, vA, mB, vB, cAB);
    std::cout << "  Model fit: k=" << kFit << "  theta=" << thetaFit << "  c=" << cFit << "\n";
    printf("  %12s | %12s | %12s | %10s\n", "quantity", "empirical", "model", "rel.diff");
    const char* names[5] = {"mean_A","var_A","mean_B","var_B","corr_AB"};
    double pred[5] = {mA,vA,mB,vB,cAB};
    for (int i=0;i<5;++i) {
        double rd = (g_targets_pt[i]!=0) ? (pred[i]-g_targets_pt[i])/g_targets_pt[i] : (pred[i]-g_targets_pt[i]);
        printf("  %12s | %12.4f | %12.4f | %+9.2f%%\n", names[i], g_targets_pt[i], pred[i], rd*100.0);
    }
}

// =============================================================================
// Sampling from the fitted model
// =============================================================================

void generateFromModelPt(int n, double k, double theta, double c, TRandom3& rng,
                        std::vector<int>& nA, std::vector<int>& nB) {
    nA.resize(n); nB.resize(n);
    for (int i=0;i<n;++i) {
        double u = rng.Uniform();
        double lambda = ROOT::Math::gamma_quantile(u, k, theta);
        nA[i] = rng.Poisson(lambda);
        nB[i] = rng.Poisson(c*lambda);
    }
}

// =============================================================================
// Data loading (mirrors analyze_dijet_entropy.C's loadDijetTreePt)
// =============================================================================

void loadDijetTreePt(const char* rootFile, const TString& level,
                    std::vector<int>& nA, std::vector<int>& nB,
                    std::vector<double>& jetPt) {
    TFile* f = TFile::Open(rootFile, "READ");
    if (!f || f->IsZombie()) { std::cerr << "ERROR: could not open " << rootFile << "\n"; return; }
    TTree* tree = (TTree*)f->Get("dijet");
    TString suffix = (level == "det") ? "_det" : "_truth";

    Int_t jet1_mult, jet2_mult;
    Double_t jet1_pt;
    Int_t hasDetDijet = 1;
    tree->SetBranchAddress(Form("jet1_mult%s", suffix.Data()), &jet1_mult);
    tree->SetBranchAddress(Form("jet2_mult%s", suffix.Data()), &jet2_mult);
    tree->SetBranchAddress(Form("jet1_pt%s", suffix.Data()), &jet1_pt);
    if (level == "det") tree->SetBranchAddress("hasDetDijet", &hasDetDijet);

    Long64_t nEntries = tree->GetEntries();
    for (Long64_t i=0;i<nEntries;++i) {
        tree->GetEntry(i);
        if (level=="det" && hasDetDijet!=1) continue;
        nA.push_back(jet1_mult); nB.push_back(jet2_mult);
        jetPt.push_back(jet1_pt);
    }
    f->Close();
}

// Quantile (equal-count) pT bin edges -- same convention as
// analyze_dijet_entropy.C's quantileBinEdges, since the jet pT spectrum
// falls steeply and fixed-width bins would leave some nearly empty.
std::vector<double> quantileBinEdgesPt(std::vector<double> pt, int nBins) {
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
// Per-bin closure test: fit a bin-specific model, get bin-specific
// pseudo-truth, run nRepeats closure test at that bin's actual N, and
// return NSB pull-distribution mean/std for I, S(B|A), NMI. Purely
// diagnostic -- no bias/coverage correction is applied anywhere.
// =============================================================================

struct BinResultPt {
    double ptLo, ptHi, ptMid;
    int N;
    double kFit, thetaFit, cFit;
    double I_true, SBA_true, NMI_true;
    double I_pullMean, I_pullStd;
    double SBA_pullMean, SBA_pullStd;
    double NMI_pullMean, NMI_pullStd;
};

BinResultPt analyzeOneBinPt(double lo, double hi, const std::vector<int>& naBin,
                             const std::vector<int>& nbBin, double ptMid,
                             long long alphabet, int nRepeats, int nBoot,
                             int nShuffles, int refN, TRandom3& rng) {
    int N = naBin.size();
    std::cout << "\n" << std::string(70,'=') << "\npT bin [" << lo << ", " << hi
              << ") GeV, N=" << N << "\n" << std::string(70,'=') << "\n";

    std::cout << "--- fitting bin-specific Gamma-Poisson model ---\n";
    double kFit, thetaFit, cFit;
    fitModelPt(naBin, nbBin, kFit, thetaFit, cFit);

    std::cout << "--- bin-specific pseudo-truth ---\n";
    std::cout << "  Generating reference sample (N=" << refN << ")...\n";
    std::vector<int> naRef, nbRef;
    generateFromModelPt(refN, kFit, thetaFit, cFit, rng, naRef, nbRef);
    double I_true = naiveIPt(naRef, nbRef);
    CtPtHist href = ctPtBuildHist(naRef, nbRef);
    double SA_true = ctPtShannonEntropy(href.ca);
    double SB_true = ctPtShannonEntropy(href.cb);
    double SAB_true = ctPtShannonEntropy(href.cab);
    double SBA_true = SAB_true - SA_true;
    double NMI_true = I_true / std::min(SA_true, SB_true);
    std::cout << "  I_true=" << I_true << "  S(B|A)_true=" << SBA_true
              << "  NMI_true=" << NMI_true << "\n";

    std::cout << "--- closure test: " << nRepeats << " repeats at N=" << N << " ---\n";
    std::vector<double> vI(nRepeats), vIerr(nRepeats);
    std::vector<double> vSBA(nRepeats), vSBAerr(nRepeats);
    std::vector<double> vNMI(nRepeats), vNMIerr(nRepeats);

    for (int rep=0; rep<nRepeats; ++rep) {
        std::vector<int> na, nb;
        generateFromModelPt(N, kFit, thetaFit, cFit, rng, na, nb);
        NsbAllResultPt nsbRes = nsbAllPt(na, nb, alphabet, alphabet, nBoot, rng);
        vI[rep] = nsbRes.I; vIerr[rep] = nsbRes.I_err;
        vSBA[rep] = nsbRes.SBA; vSBAerr[rep] = nsbRes.SBA_err;
        vNMI[rep] = nsbRes.NMI; vNMIerr[rep] = nsbRes.NMI_err;
        if ((rep+1) % std::max(1, nRepeats/5) == 0)
            std::cout << "    ... " << (rep+1) << "/" << nRepeats << " repeats done\n";
    }

    auto meanOf=[&](std::vector<double>& v){ double m=0; for(double x:v) m+=x; return m/v.size(); };
    auto stdOf=[&](std::vector<double>& v){ double m=meanOf(v); double s=0; for(double x:v) s+=(x-m)*(x-m); return std::sqrt(s/(v.size()-1)); };

    std::vector<double> iPulls(nRepeats), sbaPulls(nRepeats), nmiPulls(nRepeats);
    for (int i=0;i<nRepeats;++i) {
        iPulls[i] = (vI[i]-I_true)/vIerr[i];
        sbaPulls[i] = (vSBA[i]-SBA_true)/vSBAerr[i];
        nmiPulls[i] = (vNMI[i]-NMI_true)/vNMIerr[i];
    }

    BinResultPt r;
    r.ptLo=lo; r.ptHi=hi; r.ptMid=ptMid; r.N=N;
    r.kFit=kFit; r.thetaFit=thetaFit; r.cFit=cFit;
    r.I_true=I_true; r.SBA_true=SBA_true; r.NMI_true=NMI_true;
    r.I_pullMean=meanOf(iPulls); r.I_pullStd=stdOf(iPulls);
    r.SBA_pullMean=meanOf(sbaPulls); r.SBA_pullStd=stdOf(sbaPulls);
    r.NMI_pullMean=meanOf(nmiPulls); r.NMI_pullStd=stdOf(nmiPulls);

    printf("  I pull:      mean=%+.3f  std=%.3f\n", r.I_pullMean, r.I_pullStd);
    printf("  S(B|A) pull: mean=%+.3f  std=%.3f\n", r.SBA_pullMean, r.SBA_pullStd);
    printf("  NMI pull:    mean=%+.3f  std=%.3f\n", r.NMI_pullMean, r.NMI_pullStd);
    for (auto& pr : {std::make_pair("I", r.I_pullStd), std::make_pair("S(B|A)", r.SBA_pullStd),
                      std::make_pair("NMI", r.NMI_pullStd)}) {
        if (std::abs(pr.second - 1.0) > 0.3) {
            std::cout << "  NOTE: " << pr.first << " pull std=" << pr.second << " -> NSB "
                      << (pr.second > 1.3 ? "UNDER-covering" : "OVER-covering")
                      << " in this bin.\n";
        }
    }

    return r;
}

void printSummaryTablePt(const std::vector<BinResultPt>& results) {
    std::cout << "\n" << std::string(100,'=') << "\npT-DIFFERENTIAL CLOSURE TEST SUMMARY\n" << std::string(100,'=') << "\n";
    printf("%16s | %7s | %14s | %18s | %16s\n", "pT range", "N", "I pull (m/s)", "S(B|A) pull (m/s)", "NMI pull (m/s)");
    std::cout << std::string(100,'-') << "\n";
    for (auto& r : results) {
        printf("[%6.1f,%6.1f) | %7d | %+5.2f / %4.2f    | %+5.2f / %4.2f       | %+5.2f / %4.2f\n",
               r.ptLo, r.ptHi, r.N, r.I_pullMean, r.I_pullStd,
               r.SBA_pullMean, r.SBA_pullStd, r.NMI_pullMean, r.NMI_pullStd);
    }
}

void plotSummaryPt(const std::vector<BinResultPt>& results, const TString& outPrefix) {
    int n = results.size();
    std::vector<double> pt(n), zeros(n, 0.0);
    std::vector<double> iMean(n), iStd(n), sbaMean(n), sbaStd(n), nmiMean(n), nmiStd(n);
    for (int i = 0; i < n; ++i) {
        pt[i] = results[i].ptMid;
        iMean[i] = results[i].I_pullMean; iStd[i] = results[i].I_pullStd;
        sbaMean[i] = results[i].SBA_pullMean; sbaStd[i] = results[i].SBA_pullStd;
        nmiMean[i] = results[i].NMI_pullMean; nmiStd[i] = results[i].NMI_pullStd;
    }

    TCanvas* c = new TCanvas("cClosurePtSummary", "pT-differential closure test summary", 1600, 900);
    c->Divide(3, 2);

    struct Panel { std::vector<double>* vals; const char* label; int color; };
    Panel meanPanels[3] = {
        {&iMean, "I(A:B)", kViolet+1}, {&sbaMean, "S(B|A)", kMagenta+2}, {&nmiMean, "NMI", kGreen+2}
    };
    Panel stdPanels[3] = {
        {&iStd, "I(A:B)", kViolet+1}, {&sbaStd, "S(B|A)", kMagenta+2}, {&nmiStd, "NMI", kGreen+2}
    };

    for (int col = 0; col < 3; ++col) {
        c->cd(col+1);
        TGraph* g = new TGraph(n, pt.data(), meanPanels[col].vals->data());
        g->SetMarkerStyle(20); g->SetMarkerColor(meanPanels[col].color); g->SetLineColor(meanPanels[col].color);
        g->SetTitle(Form("%s: pull mean vs. p_{T} (should be ~0);leading jet p_{T} (GeV);NSB pull mean", meanPanels[col].label));
        g->Draw("APL");
        TLine* zeroLine = new TLine(pt.front(), 0, pt.back(), 0);
        zeroLine->SetLineColor(kGray+1);
        zeroLine->Draw();

        c->cd(col+4);
        TGraph* g2 = new TGraph(n, pt.data(), stdPanels[col].vals->data());
        g2->SetMarkerStyle(20); g2->SetMarkerColor(stdPanels[col].color); g2->SetLineColor(stdPanels[col].color);
        g2->SetTitle(Form("%s: pull std vs. p_{T} (>1: under-cov., <1: over-cov.);leading jet p_{T} (GeV);NSB pull std", stdPanels[col].label));
        g2->Draw("APL");
        TLine* oneLine = new TLine(pt.front(), 1, pt.back(), 1);
        oneLine->SetLineColor(kRed); oneLine->SetLineStyle(2);
        oneLine->Draw();
    }

    c->SaveAs(outPrefix + "_closure_test_pt_summary.png");
    std::cout << "\n  saved " << outPrefix << "_closure_test_pt_summary.png\n";
}

// =============================================================================
// Main entry point
// =============================================================================

void closure_test_pt(const char* rootFile, const char* levelIn = "truth",
                      long long alphabet = 80, int ptBins = 4, int nRepeats = 100,
                      int nBoot = 150, int nShuffles = 100, int refN = 1000000,
                      int minEvents = 200,
                      const char* figsDir = "/Users/zhoudunmingtu/bnl_work/Work/MODELS/EBCs_toyMC/analysis/figs") {
    TString level(levelIn);

    gStyle->SetPadLeftMargin(0.15);
    gStyle->SetPadRightMargin(0.06);
    gStyle->SetPadBottomMargin(0.13);
    gStyle->SetPadTopMargin(0.09);
    gStyle->SetTitleOffset(1.5, "Y");
    gStyle->SetTitleOffset(1.1, "X");
    gStyle->SetTitleSize(0.045, "XYZ");
    gStyle->SetLabelSize(0.035, "XYZ");
    gStyle->SetTitleSize(0.05, "T");

    TString figsDirStr(figsDir);
    gSystem->mkdir(figsDirStr, kTRUE);
    TString outPrefix = figsDirStr + TString::Format("/closure_test_pt_root_%s", levelIn);
    std::cout << "Figures will be saved under: " << figsDirStr << "\n";

    TRandom3 rng(20260710);

    std::cout << "Loading " << rootFile << " (level='" << level << "') ...\n";
    std::vector<int> nA, nB;
    std::vector<double> jetPt;
    loadDijetTreePt(rootFile, level, nA, nB, jetPt);
    std::cout << "Loaded " << nA.size() << " events.\n";

    std::vector<double> edges = quantileBinEdgesPt(jetPt, ptBins);
    std::cout << "pT bin edges (quantile): ";
    for (double e : edges) std::cout << e << " ";
    std::cout << "\n";

    std::vector<BinResultPt> binResults;
    for (int ib = 0; ib < ptBins; ++ib) {
        double lo = edges[ib], hi = edges[ib+1];
        std::vector<int> naBin, nbBin;
        double ptSum = 0;
        for (size_t i = 0; i < nA.size(); ++i) {
            bool inBin = (ib < ptBins-1) ? (jetPt[i] >= lo && jetPt[i] < hi)
                                          : (jetPt[i] >= lo && jetPt[i] <= hi);
            if (inBin) {
                naBin.push_back(nA[i]);
                nbBin.push_back(nB[i]);
                ptSum += jetPt[i];
            }
        }
        int n = naBin.size();
        if (n < minEvents) {
            std::cout << "\nSkipping bin [" << lo << "," << hi << "): only " << n
                      << " events (< minEvents=" << minEvents << ")\n";
            continue;
        }
        BinResultPt r = analyzeOneBinPt(lo, hi, naBin, nbBin, ptSum/n, alphabet,
                                         nRepeats, nBoot, nShuffles, refN, rng);
        binResults.push_back(r);
    }

    if (binResults.empty()) {
        std::cout << "\nNo bins had enough events -- nothing to summarize.\n";
        return;
    }

    printSummaryTablePt(binResults);
    plotSummaryPt(binResults, outPrefix);

    std::cout << "\nDone.\n";
}
