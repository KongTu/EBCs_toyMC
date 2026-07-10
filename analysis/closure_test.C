// closure_test.C
//
// Tier-2 closure test (ROOT version) -- see closure_test.py for the full
// design rationale. Summary: real PYTHIA8 output has no known "true" MI
// to validate against, so this fits the same Gamma-Poisson shared-latent
// model used throughout the toy study to the REAL data's marginals and
// correlation strength, generates a huge reference sample from THAT
// fitted model (giving a numerically-exact pseudo-truth calibrated to
// resemble your actual data), then repeatedly draws samples at your real
// N and checks whether naive/Miller-Madow/shuffle/NSB recover it -- both
// in bias (mean estimate vs. truth) and calibration (does the NSB pull
// distribution (estimate-truth)/sigma look standard-normal?).
//
// Self-contained: duplicates the estimator functions from
// analyze_dijet_entropy.C rather than depending on cross-macro linkage
// (which can be fragile in ROOT). The ctDigamma/ctTrigamma implementations
// are identical to (and were validated the same way as) those in
// analyze_dijet_entropy.C.
//
// Gamma-distributed latent variable sampling uses ROOT::Math::gamma_quantile
// (inverse-CDF method) -- confirmed to live in MathCore (no GSL/MathMore
// dependency) before use here.
//
// Usage (ACLiC-compiled -- strongly recommended, this does a lot of work):
//   root -l -q 'closure_test.C+("pythia_dijet_pp200.root","truth",90,30,50,100,2000000)'
//
// Arguments: (rootFile, level, alphabet, nRepeats, nBoot, nShuffles, refN)
//
// Same honest caveat as the other macros in this project: written against
// documented ROOT APIs and validated pure-math components, but not
// compiled/run in the environment this was written in (no ROOT there).
// Send compiler output if anything doesn't build.

#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TH1D.h>
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

double ctDigamma(double x) {
    double result = 0.0;
    while (x < 6.0) { result -= 1.0/x; x += 1.0; }
    double f = 1.0/(x*x);
    result += std::log(x) - 0.5/x
        - f*(1.0/12.0 - f*(1.0/120.0 - f*(1.0/252.0 - f*(1.0/240.0 - f*(1.0/132.0)))));
    return result;
}

double ctTrigamma(double x) {
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

double ctShannonEntropy(const std::vector<int>& counts) {
    double N = 0; for (int c : counts) N += c;
    double S = 0.0;
    for (int c : counts) { if (c<=0) continue; double p=c/N; S -= p*std::log(p); }
    return S;
}

struct CtHist { std::vector<int> ca, cb, cab; int KA,KB,KAB; long long N; };

CtHist ctBuildHist(const std::vector<int>& nA, const std::vector<int>& nB) {
    std::map<int,int> mA, mB; std::map<std::pair<int,int>,int> mAB;
    for (size_t i=0;i<nA.size();++i){ mA[nA[i]]++; mB[nB[i]]++; mAB[{nA[i],nB[i]}]++; }
    CtHist h;
    for (auto& kv: mA) h.ca.push_back(kv.second);
    for (auto& kv: mB) h.cb.push_back(kv.second);
    for (auto& kv: mAB) h.cab.push_back(kv.second);
    h.KA=h.ca.size(); h.KB=h.cb.size(); h.KAB=h.cab.size(); h.N=nA.size();
    return h;
}

double naiveI(const std::vector<int>& nA, const std::vector<int>& nB) {
    CtHist h = ctBuildHist(nA, nB);
    return ctShannonEntropy(h.ca) + ctShannonEntropy(h.cb) - ctShannonEntropy(h.cab);
}

double millerMadowI(const std::vector<int>& nA, const std::vector<int>& nB) {
    CtHist h = ctBuildHist(nA, nB);
    double I = ctShannonEntropy(h.ca) + ctShannonEntropy(h.cb) - ctShannonEntropy(h.cab);
    double bias = (h.KAB - h.KA - h.KB + 1) / (2.0*h.N);
    return I - bias;
}

double shuffleI(const std::vector<int>& nA, const std::vector<int>& nB, int nShuffles, TRandom3& rng) {
    double I0 = naiveI(nA, nB);
    std::vector<int> nBs = nB;
    double sum = 0;
    for (int s=0; s<nShuffles; ++s) {
        for (int i=(int)nBs.size()-1;i>0;--i) { int j=rng.Integer(i+1); std::swap(nBs[i],nBs[j]); }
        sum += naiveI(nA, nBs);
    }
    return I0 - sum/nShuffles;
}

// NSB mean-only entropy (fast: skips the Wolpert-Wolf variance terms,
// since the closure test gets its uncertainty from bootstrapping the
// composite I/S(B|A) directly -- see nsbAll below).
double nsbEntropyMean(const std::vector<int>& countsNonzero, long long K) {
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
        for (int c : countsNonzero) { double a=c+beta; sumTerm += a*ctDigamma(a+1.0); }
        sumTerm += Kzero*beta*ctDigamma(beta+1.0);
        means[ib] = ctDigamma(A+1.0) - sumTerm/A;

        double lp = TMath::LnGamma(K*beta) - TMath::LnGamma(N+K*beta) + TMath::LnGamma(N+1.0);
        for (int c: countsNonzero) lp += TMath::LnGamma(c+beta) - TMath::LnGamma(beta);
        for (int c: countsNonzero) lp -= TMath::LnGamma(c+1.0);
        logP[ib] = lp;
        rho[ib] = K*ctTrigamma(K*beta+1.0) - ctTrigamma(beta+1.0);
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

struct NsbAllResult { double SA,SB,SAB,I,I_err,SBA,SBA_err,NMI,NMI_err; };

NsbAllResult nsbAll(const std::vector<int>& nA, const std::vector<int>& nB,
                     long long alphaA, long long alphaB, int nBoot, TRandom3& rng) {
    CtHist h0 = ctBuildHist(nA, nB);
    long long Kj = alphaA*alphaB;
    double SA = nsbEntropyMean(h0.ca, alphaA);
    double SB = nsbEntropyMean(h0.cb, alphaB);
    double SAB = nsbEntropyMean(h0.cab, Kj);

    int N = nA.size();
    std::vector<double> Iboot(nBoot), SBAboot(nBoot), NMIboot(nBoot);
    for (int b=0; b<nBoot; ++b) {
        std::vector<int> rA(N), rB(N);
        for (int i=0;i<N;++i){ int idx=rng.Integer(N); rA[i]=nA[idx]; rB[i]=nB[idx]; }
        CtHist hb = ctBuildHist(rA, rB);
        double sA = nsbEntropyMean(hb.ca, alphaA);
        double sB = nsbEntropyMean(hb.cb, alphaB);
        double sAB = nsbEntropyMean(hb.cab, Kj);
        Iboot[b] = sA+sB-sAB;
        SBAboot[b] = sAB - sA;
        NMIboot[b] = Iboot[b] / std::min(sA, sB);
    }
    auto stdv=[&](std::vector<double>& v){ double m=0; for(double x:v) m+=x; m/=v.size();
        double s=0; for(double x:v) s+=(x-m)*(x-m); s/=(v.size()-1); return std::sqrt(s); };

    NsbAllResult r;
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

void modelMoments(double k, double theta, double c,
                   double& meanA, double& varA, double& meanB, double& varB, double& corrAB) {
    meanA = k*theta;
    varA = k*theta*(1+theta);
    meanB = c*k*theta;
    varB = c*k*theta*(1+c*theta);
    double covAB = c*k*theta*theta;
    corrAB = covAB/std::sqrt(varA*varB);
}

// TMinuit needs a global (C-style) FCN and globals for the fit targets.
double g_targets[5];  // meanA, varA, meanB, varB, corrAB (empirical)

void fitFCN(Int_t& npar, Double_t* gin, Double_t& f, Double_t* par, Int_t iflag) {
    double k = std::exp(par[0]), theta = std::exp(par[1]), c = std::exp(par[2]);
    double meanA, varA, meanB, varB, corrAB;
    modelMoments(k, theta, c, meanA, varA, meanB, varB, corrAB);
    double pred[5] = {meanA, varA, meanB, varB, corrAB};
    double loss = 0;
    for (int i=0;i<4;++i) { double rel=(pred[i]-g_targets[i])/g_targets[i]; loss += rel*rel; }
    double corrErr = pred[4]-g_targets[4];
    loss += corrErr*corrErr;
    f = loss;
}

void fitModel(const std::vector<int>& nA, const std::vector<int>& nB,
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

    g_targets[0]=meanA; g_targets[1]=varA; g_targets[2]=meanB; g_targets[3]=varB; g_targets[4]=corrAB;

    double k0 = (varA>meanA) ? meanA*meanA/(varA-meanA) : 3.0;
    double theta0 = meanA/k0;
    double c0 = meanB/meanA;

    TMinuit minuit(3);
    minuit.SetFCN(fitFCN);
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
    modelMoments(kFit, thetaFit, cFit, mA, vA, mB, vB, cAB);
    std::cout << "  Model fit: k=" << kFit << "  theta=" << thetaFit << "  c=" << cFit << "\n";
    printf("  %12s | %12s | %12s | %10s\n", "quantity", "empirical", "model", "rel.diff");
    const char* names[5] = {"mean_A","var_A","mean_B","var_B","corr_AB"};
    double pred[5] = {mA,vA,mB,vB,cAB};
    for (int i=0;i<5;++i) {
        double rd = (g_targets[i]!=0) ? (pred[i]-g_targets[i])/g_targets[i] : (pred[i]-g_targets[i]);
        printf("  %12s | %12.4f | %12.4f | %+9.2f%%\n", names[i], g_targets[i], pred[i], rd*100.0);
    }
}

// =============================================================================
// Sampling from the fitted model
// =============================================================================

void generateFromModel(int n, double k, double theta, double c, TRandom3& rng,
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
// Data loading (mirrors analyze_dijet_entropy.C's loadDijetTree)
// =============================================================================

void loadDijetTree(const char* rootFile, const TString& level,
                    std::vector<int>& nA, std::vector<int>& nB) {
    TFile* f = TFile::Open(rootFile, "READ");
    if (!f || f->IsZombie()) { std::cerr << "ERROR: could not open " << rootFile << "\n"; return; }
    TTree* tree = (TTree*)f->Get("dijet");
    TString suffix = (level == "det") ? "_det" : "_truth";

    Int_t jet1_mult, jet2_mult;
    Int_t hasDetDijet = 1;
    tree->SetBranchAddress(Form("jet1_mult%s", suffix.Data()), &jet1_mult);
    tree->SetBranchAddress(Form("jet2_mult%s", suffix.Data()), &jet2_mult);
    if (level == "det") tree->SetBranchAddress("hasDetDijet", &hasDetDijet);

    Long64_t nEntries = tree->GetEntries();
    for (Long64_t i=0;i<nEntries;++i) {
        tree->GetEntry(i);
        if (level=="det" && hasDetDijet!=1) continue;
        nA.push_back(jet1_mult); nB.push_back(jet2_mult);
    }
    f->Close();
}

// =============================================================================
// Main entry point
// =============================================================================

void closure_test(const char* rootFile, const char* levelIn = "truth",
                   long long alphabet = 80, int nRepeats = 30, int nBoot = 50,
                   int nShuffles = 100, int refN = 2000000, int closureN = -1,
                   const char* figsDir = "/Users/zhoudunmingtu/bnl_work/Work/MODELS/EBCs_toyMC/analysis/figs") {
    TString level(levelIn);

    // Global plotting style: larger margins + axis title offsets so
    // longer axis titles aren't clipped (same fix as analyze_dijet_entropy.C).
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
    TString outPrefix = figsDirStr + TString::Format("/closure_test_root_%s", levelIn);
    std::cout << "Figures will be saved under: " << figsDirStr << "\n";

    TRandom3 rng(20260709);

    std::cout << "Loading " << rootFile << " (level='" << level << "') ...\n";
    std::vector<int> nA, nB;
    loadDijetTree(rootFile, level, nA, nB);
    int N_real = nA.size();
    std::cout << "Loaded " << N_real << " events.\n";
    int N_closure = (closureN > 0) ? closureN : N_real;

    std::cout << "\n--- Step 1: fitting Gamma-Poisson model to real data ---\n";
    double kFit, thetaFit, cFit;
    fitModel(nA, nB, kFit, thetaFit, cFit);

    std::cout << "\n--- Step 2: pseudo-truth from fitted model ---\n";
    std::cout << "  Generating reference sample (N=" << refN << ") from the fitted model...\n";
    std::vector<int> naRef, nbRef;
    generateFromModel(refN, kFit, thetaFit, cFit, rng, naRef, nbRef);
    double I_true = naiveI(naRef, nbRef);
    CtHist href = ctBuildHist(naRef, nbRef);
    double SA_true = ctShannonEntropy(href.ca);
    double SB_true = ctShannonEntropy(href.cb);
    double SAB_true = ctShannonEntropy(href.cab);
    double SBA_true = SAB_true - SA_true;
    double NMI_true = I_true / std::min(SA_true, SB_true);
    std::cout << "  I_true (pseudo-truth)      = " << I_true << " nats\n";
    std::cout << "  S(B|A)_true (pseudo-truth) = " << SBA_true << " nats\n";
    std::cout << "  NMI_true (pseudo-truth)    = " << NMI_true
              << "  (SA_true=" << SA_true << ", SB_true=" << SB_true << ")\n";

    std::cout << "\n--- Step 3: closure test (recovery at your actual N) ---\n";
    std::cout << "  Running " << nRepeats << " independent draws at N=" << N_closure << "...\n";

    std::vector<double> vNaive(nRepeats), vMM(nRepeats), vShuffle(nRepeats);
    std::vector<double> vNSB(nRepeats), vNSBerr(nRepeats);
    std::vector<double> vNMI(nRepeats), vNMIerr(nRepeats);

    for (int rep=0; rep<nRepeats; ++rep) {
        std::vector<int> na, nb;
        generateFromModel(N_closure, kFit, thetaFit, cFit, rng, na, nb);

        vNaive[rep] = naiveI(na, nb);
        vMM[rep] = millerMadowI(na, nb);
        vShuffle[rep] = shuffleI(na, nb, nShuffles, rng);
        NsbAllResult nsbRes = nsbAll(na, nb, alphabet, alphabet, nBoot, rng);
        vNSB[rep] = nsbRes.I;
        vNSBerr[rep] = nsbRes.I_err;
        vNMI[rep] = nsbRes.NMI;
        vNMIerr[rep] = nsbRes.NMI_err;

        if ((rep+1) % std::max(1, nRepeats/5) == 0)
            std::cout << "    ... " << (rep+1) << "/" << nRepeats << " repeats done\n";
    }

    auto meanOf=[&](std::vector<double>& v){ double m=0; for(double x:v) m+=x; return m/v.size(); };
    auto stdOf=[&](std::vector<double>& v){ double m=meanOf(v); double s=0; for(double x:v) s+=(x-m)*(x-m); return std::sqrt(s/(v.size()-1)); };

    std::cout << "\n" << std::string(70,'=') << "\nCLOSURE TEST RESULTS (N=" << N_closure << " per repeat)\n" << std::string(70,'=') << "\n";
    std::cout << "I_true (pseudo-truth) = " << I_true << " nats\n\n";
    printf("%14s | %14s | %16s | %10s | %9s\n", "method","mean estimate","std across reps","bias","bias/std");
    std::cout << std::string(78,'-') << "\n";

    struct MethodStat { const char* label; std::vector<double>* vals; };
    MethodStat methods[4] = {
        {"naive", &vNaive}, {"Miller-Madow", &vMM}, {"shuffle", &vShuffle}, {"NSB", &vNSB}
    };
    double means[4], stds[4];
    for (int m=0; m<4; ++m) {
        means[m] = meanOf(*methods[m].vals);
        stds[m] = stdOf(*methods[m].vals);
        double bias = means[m] - I_true;
        printf("%14s | %14.5f | %16.5f | %+10.5f | %+9.2f\n",
               methods[m].label, means[m], stds[m], bias, bias/stds[m]);
    }

    // ---- NSB calibration: pull distribution ----
    std::vector<double> pulls(nRepeats);
    for (int i=0;i<nRepeats;++i) pulls[i] = (vNSB[i]-I_true)/vNSBerr[i];
    double pullMean = meanOf(pulls), pullStd = stdOf(pulls);
    int within1=0, within2=0;
    for (double p : pulls) { if (std::abs(p)<1.0) within1++; if (std::abs(p)<2.0) within2++; }
    std::cout << "\nNSB pull distribution (estimate - truth) / reported_sigma:\n";
    std::cout << "  mean pull = " << pullMean << "  (should be ~0)\n";
    std::cout << "  std  pull = " << pullStd  << "  (should be ~1 if calibrated)\n";
    printf("  fraction within +/-1 sigma: %.1f%%  (expect ~68%%)\n", 100.0*within1/nRepeats);
    printf("  fraction within +/-2 sigma: %.1f%%  (expect ~95%%)\n", 100.0*within2/nRepeats);
    if (std::abs(pullStd - 1.0) > 0.3) {
        std::cout << "  NOTE: pull std deviates from 1 by >0.3 -- NSB uncertainty appears "
                  << (pullStd > 1.3 ? "UNDER-covering (error bars too small)" : "OVER-covering (too conservative)")
                  << " for data shaped like this, at N=" << N_closure << ".\n";
    }

    // ---- plots ----
    TCanvas* c = new TCanvas("cClosure", "Closure test", 1600, 480);
    c->Divide(3,1);

    c->cd(1);
    TH1D* hBar = new TH1D("hBar", "Closure test: recovery;;recovered I(A:B) (nats)", 4, 0, 4);
    for (int m=0;m<4;++m) { hBar->SetBinContent(m+1, means[m]); hBar->SetBinError(m+1, stds[m]);
                             hBar->GetXaxis()->SetBinLabel(m+1, methods[m].label); }
    hBar->SetFillColorAlpha(kAzure+1, 0.6);
    hBar->SetStats(0);
    hBar->Draw("E1 HIST");
    TLine* trueLine = new TLine(0, I_true, 4, I_true);
    trueLine->SetLineColor(kBlack); trueLine->SetLineStyle(2); trueLine->SetLineWidth(2);
    trueLine->Draw();

    c->cd(2);
    double pmin = *std::min_element(pulls.begin(),pulls.end())-0.5;
    double pmax = *std::max_element(pulls.begin(),pulls.end())+0.5;
    TH1D* hPull = new TH1D("hPull", "NSB calibration: pull distribution;(#hat{I}_{NSB}-I_{true})/#sigma_{NSB};density",
                            std::max(8, nRepeats/4), std::min(-4.0,pmin), std::max(4.0,pmax));
    for (double p : pulls) hPull->Fill(p);
    if (hPull->Integral() > 0) hPull->Scale(1.0/hPull->Integral("width"));
    hPull->SetFillColorAlpha(kViolet+1, 0.5);
    hPull->SetStats(0);
    hPull->Draw("HIST");
    TF1* fGaus = new TF1("fGaus", "TMath::Gaus(x,0,1,1)", -4, 4);
    fGaus->SetLineColor(kBlack); fGaus->SetLineStyle(2); fGaus->SetLineWidth(2);
    fGaus->Draw("SAME");
    TLegend* legP = new TLegend(0.6,0.7,0.88,0.88);
    legP->AddEntry(hPull, "NSB pulls", "f");
    legP->AddEntry(fGaus, "standard normal", "l");
    legP->Draw();

    c->cd(3);
    std::vector<double> repIdx(nRepeats), zeros(nRepeats,0.0);
    for (int i=0;i<nRepeats;++i) repIdx[i]=i;
    TGraphErrors* gRep = new TGraphErrors(nRepeats, repIdx.data(), vNSB.data(), zeros.data(), vNSBerr.data());
    gRep->SetMarkerStyle(20); gRep->SetMarkerColor(kViolet+1); gRep->SetLineColor(kViolet+1);
    gRep->SetTitle("NSB per-repeat estimates vs. truth;repeat #;NSB estimate #pm bootstrap #sigma");
    gRep->Draw("AP");
    TLine* trueLine2 = new TLine(0, I_true, nRepeats-1, I_true);
    trueLine2->SetLineColor(kBlack); trueLine2->SetLineStyle(2); trueLine2->SetLineWidth(2);
    trueLine2->Draw();

    c->SaveAs(outPrefix + "_closure_test.png");
    std::cout << "\n  saved " << outPrefix << "_closure_test.png\n";

    // ---- NEW: NMI closure test (separate report + separate plot file;
    // the figure above is untouched) ----
    std::cout << "\n" << std::string(70,'=') << "\nNMI CLOSURE TEST (N=" << N_closure << " per repeat)\n" << std::string(70,'=') << "\n";
    std::cout << "NMI_true (pseudo-truth) = " << NMI_true
              << "  (classically bounded in [0,1]; NMI>1 would be a classical-bound violation)\n\n";

    double nmiMean = meanOf(vNMI), nmiStd = stdOf(vNMI);
    double nmiBias = nmiMean - NMI_true;
    printf("%14s | %14s | %16s | %10s | %9s\n", "method","mean estimate","std across reps","bias","bias/std");
    std::cout << std::string(78,'-') << "\n";
    printf("%14s | %14.5f | %16.5f | %+10.5f | %+9.2f\n", "NSB", nmiMean, nmiStd, nmiBias, nmiBias/nmiStd);

    std::vector<double> nmiPulls(nRepeats);
    for (int i=0;i<nRepeats;++i) nmiPulls[i] = (vNMI[i]-NMI_true)/vNMIerr[i];
    double nmiPullMean = meanOf(nmiPulls), nmiPullStd = stdOf(nmiPulls);
    int nmiWithin1=0, nmiWithin2=0;
    for (double p : nmiPulls) { if (std::abs(p)<1.0) nmiWithin1++; if (std::abs(p)<2.0) nmiWithin2++; }
    std::cout << "\nNSB NMI pull distribution (estimate - truth) / reported_sigma:\n";
    std::cout << "  mean pull = " << nmiPullMean << "  (should be ~0)\n";
    std::cout << "  std  pull = " << nmiPullStd  << "  (should be ~1 if calibrated)\n";
    printf("  fraction within +/-1 sigma: %.1f%%  (expect ~68%%)\n", 100.0*nmiWithin1/nRepeats);
    printf("  fraction within +/-2 sigma: %.1f%%  (expect ~95%%)\n", 100.0*nmiWithin2/nRepeats);

    TCanvas* c2 = new TCanvas("cClosureNMI", "NMI closure test", 1600, 480);
    c2->Divide(3,1);

    c2->cd(1);
    TH1D* hBarNMI = new TH1D("hBarNMI", "NMI closure test: recovery;;recovered NMI", 1, 0, 1);
    hBarNMI->SetBinContent(1, nmiMean); hBarNMI->SetBinError(1, nmiStd);
    hBarNMI->GetXaxis()->SetBinLabel(1, "NSB");
    hBarNMI->SetFillColorAlpha(kGreen+2, 0.6);
    hBarNMI->SetStats(0);
    hBarNMI->Draw("E1 HIST");
    TLine* trueLineNMI = new TLine(0, NMI_true, 1, NMI_true);
    trueLineNMI->SetLineColor(kBlack); trueLineNMI->SetLineStyle(2); trueLineNMI->SetLineWidth(2);
    trueLineNMI->Draw();

    c2->cd(2);
    double pminN = *std::min_element(nmiPulls.begin(),nmiPulls.end())-0.5;
    double pmaxN = *std::max_element(nmiPulls.begin(),nmiPulls.end())+0.5;
    TH1D* hPullNMI = new TH1D("hPullNMI", "NSB NMI calibration: pull distribution;(#hat{NMI}_{NSB}-NMI_{true})/#sigma_{NMI};density",
                               std::max(8, nRepeats/4), std::min(-4.0,pminN), std::max(4.0,pmaxN));
    for (double p : nmiPulls) hPullNMI->Fill(p);
    if (hPullNMI->Integral() > 0) hPullNMI->Scale(1.0/hPullNMI->Integral("width"));
    hPullNMI->SetFillColorAlpha(kGreen+2, 0.5);
    hPullNMI->SetStats(0);
    hPullNMI->Draw("HIST");
    TF1* fGausNMI = new TF1("fGausNMI", "TMath::Gaus(x,0,1,1)", -4, 4);
    fGausNMI->SetLineColor(kBlack); fGausNMI->SetLineStyle(2); fGausNMI->SetLineWidth(2);
    fGausNMI->Draw("SAME");
    TLegend* legPNMI = new TLegend(0.6,0.7,0.88,0.88);
    legPNMI->AddEntry(hPullNMI, "NSB NMI pulls", "f");
    legPNMI->AddEntry(fGausNMI, "standard normal", "l");
    legPNMI->Draw();

    c2->cd(3);
    TGraphErrors* gRepNMI = new TGraphErrors(nRepeats, repIdx.data(), vNMI.data(), zeros.data(), vNMIerr.data());
    gRepNMI->SetMarkerStyle(20); gRepNMI->SetMarkerColor(kGreen+2); gRepNMI->SetLineColor(kGreen+2);
    gRepNMI->SetTitle("NSB NMI per-repeat estimates vs. truth;repeat #;NSB NMI estimate #pm bootstrap #sigma");
    gRepNMI->Draw("AP");
    TLine* trueLine2NMI = new TLine(0, NMI_true, nRepeats-1, NMI_true);
    trueLine2NMI->SetLineColor(kBlack); trueLine2NMI->SetLineStyle(2); trueLine2NMI->SetLineWidth(2);
    trueLine2NMI->Draw();

    c2->SaveAs(outPrefix + "_closure_test_NMI.png");
    std::cout << "\n  saved " << outPrefix << "_closure_test_NMI.png\n";

    std::cout << "\nDone.\n";
}
