// analyze_dijet_mixed.C
//
// Mixed-event subtraction of the TRIVIAL KINEMATIC contribution to the
// dijet multiplicity MI (ROOT version) -- see analyze_dijet_mixed.py
// for the full design rationale. Summary:
//
//   I_signal = I_corrected(same-event pairs) - I_corrected(mixed pairs)
//
// where the mixed sample keeps event i's jet A intact and takes jet B
// from another event j with pT2_j ~= pT2_i (k-nearest-neighbor matching
// in pT2). The mixed sample's MI IS the trivial kinematic (common-cause)
// part induced by integrating over a finite pT window. Bias correction
// is applied PER TERM (never to the already-subtracted difference --
// the leading naive bias largely cancels between the terms already,
// same marginals + same N by construction). Errors: JOINT bootstrap of
// the difference, re-matching within each resample.
//
// Separate file from analyze_dijet_entropy.C on purpose, so the
// raw-MI analysis remains available unchanged. Self-contained, all
// symbols "mx"-prefixed (no collisions if loaded alongside the other
// macros).
//
// Usage (ACLiC):
//   root -l -q 'analyze_dijet_mixed.C+("file.root","truth",80,10,50,30,4)'
// Arguments: (rootFile, level, alphabet, mixNeighbors, nBoot,
//             nShuffles, ptBins, minBinEvents, figsDir)
//
// Same honest caveat as the other macros: written against documented
// ROOT APIs but not compiled here (no ROOT in this environment) -- send
// compiler output if anything doesn't build.

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

double mxDigamma(double x) {
    double result = 0.0;
    while (x < 6.0) { result -= 1.0/x; x += 1.0; }
    double f = 1.0/(x*x);
    result += std::log(x) - 0.5/x
        - f*(1.0/12.0 - f*(1.0/120.0 - f*(1.0/252.0 - f*(1.0/240.0 - f*(1.0/132.0)))));
    return result;
}

double mxTrigamma(double x) {
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

double mxShannonEntropy(const std::vector<int>& counts) {
    double N = 0; for (int c : counts) N += c;
    double S = 0.0;
    for (int c : counts) { if (c<=0) continue; double p=c/N; S -= p*std::log(p); }
    return S;
}

struct MxHist { std::vector<int> ca, cb, cab; int KA,KB,KAB; long long N; };

MxHist mxBuildHist(const std::vector<int>& nA, const std::vector<int>& nB) {
    std::map<int,int> mA, mB; std::map<std::pair<int,int>,int> mAB;
    for (size_t i=0;i<nA.size();++i){ mA[nA[i]]++; mB[nB[i]]++; mAB[{nA[i],nB[i]}]++; }
    MxHist h;
    for (auto& kv: mA) h.ca.push_back(kv.second);
    for (auto& kv: mB) h.cb.push_back(kv.second);
    for (auto& kv: mAB) h.cab.push_back(kv.second);
    h.KA=h.ca.size(); h.KB=h.cb.size(); h.KAB=h.cab.size(); h.N=nA.size();
    return h;
}

double mxNaiveI(const std::vector<int>& nA, const std::vector<int>& nB) {
    MxHist h = mxBuildHist(nA, nB);
    return mxShannonEntropy(h.ca) + mxShannonEntropy(h.cb) - mxShannonEntropy(h.cab);
}

double mxMillerMadowI(const std::vector<int>& nA, const std::vector<int>& nB) {
    MxHist h = mxBuildHist(nA, nB);
    double I = mxShannonEntropy(h.ca) + mxShannonEntropy(h.cb) - mxShannonEntropy(h.cab);
    double bias = (h.KAB - h.KA - h.KB + 1) / (2.0*h.N);
    return I - bias;
}

double mxShuffleI(const std::vector<int>& nA, const std::vector<int>& nB, int nShuffles, TRandom3& rng) {
    double I0 = mxNaiveI(nA, nB);
    std::vector<int> nBs = nB;
    double sum = 0;
    for (int s=0; s<nShuffles; ++s) {
        for (int i=(int)nBs.size()-1;i>0;--i) { int j=rng.Integer(i+1); std::swap(nBs[i],nBs[j]); }
        sum += mxNaiveI(nA, nBs);
    }
    return I0 - sum/nShuffles;
}

// NSB mean-only entropy (fast: skips the Wolpert-Wolf variance terms,
// since the closure test gets its uncertainty from bootstrapping the
// composite I/S(B|A) directly -- see mxNsbAll below).
double mxNsbEntropyMean(const std::vector<int>& countsNonzero, long long K) {
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
        for (int c : countsNonzero) { double a=c+beta; sumTerm += a*mxDigamma(a+1.0); }
        sumTerm += Kzero*beta*mxDigamma(beta+1.0);
        means[ib] = mxDigamma(A+1.0) - sumTerm/A;

        double lp = TMath::LnGamma(K*beta) - TMath::LnGamma(N+K*beta) + TMath::LnGamma(N+1.0);
        for (int c: countsNonzero) lp += TMath::LnGamma(c+beta) - TMath::LnGamma(beta);
        for (int c: countsNonzero) lp -= TMath::LnGamma(c+1.0);
        logP[ib] = lp;
        rho[ib] = K*mxTrigamma(K*beta+1.0) - mxTrigamma(beta+1.0);
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

struct MxNsbAllResult { double SA,SB,SAB,I,I_err,SBA,SBA_err,NMI,NMI_err; };

MxNsbAllResult mxNsbAll(const std::vector<int>& nA, const std::vector<int>& nB,
                     long long alphaA, long long alphaB, int nBoot, TRandom3& rng) {
    MxHist h0 = mxBuildHist(nA, nB);
    long long Kj = alphaA*alphaB;
    double SA = mxNsbEntropyMean(h0.ca, alphaA);
    double SB = mxNsbEntropyMean(h0.cb, alphaB);
    double SAB = mxNsbEntropyMean(h0.cab, Kj);

    int N = nA.size();
    std::vector<double> Iboot(nBoot), SBAboot(nBoot), NMIboot(nBoot);
    for (int b=0; b<nBoot; ++b) {
        std::vector<int> rA(N), rB(N);
        for (int i=0;i<N;++i){ int idx=rng.Integer(N); rA[i]=nA[idx]; rB[i]=nB[idx]; }
        MxHist hb = mxBuildHist(rA, rB);
        double sA = mxNsbEntropyMean(hb.ca, alphaA);
        double sB = mxNsbEntropyMean(hb.cb, alphaB);
        double sAB = mxNsbEntropyMean(hb.cab, Kj);
        Iboot[b] = sA+sB-sAB;
        SBAboot[b] = sAB - sA;
        NMIboot[b] = Iboot[b] / std::min(sA, sB);
    }
    auto stdv=[&](std::vector<double>& v){ double m=0; for(double x:v) m+=x; m/=v.size();
        double s=0; for(double x:v) s+=(x-m)*(x-m); s/=(v.size()-1); return std::sqrt(s); };

    MxNsbAllResult r;
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

struct MxAllPoints { double naive, mm, shuffle, nsb; };

MxAllPoints mxAllPointEstimates(const std::vector<int>& nA, const std::vector<int>& nB,
                                 long long alphabet, int nShuffles, TRandom3& rng) {
    MxHist h = mxBuildHist(nA, nB);
    double S_A = mxShannonEntropy(h.ca), S_B = mxShannonEntropy(h.cb), S_AB = mxShannonEntropy(h.cab);
    MxAllPoints r;
    r.naive = S_A + S_B - S_AB;
    r.mm = r.naive - (h.KAB - h.KA - h.KB + 1) / (2.0 * h.N);
    r.shuffle = mxShuffleI(nA, nB, nShuffles, rng);
    long long Kj = alphabet * alphabet;
    r.nsb = mxNsbEntropyMean(h.ca, alphabet) + mxNsbEntropyMean(h.cb, alphabet)
          - mxNsbEntropyMean(h.cab, Kj);
    return r;
}

// =============================================================================
// Mixed-partner construction: k-nearest-neighbor matching in pT2
// =============================================================================

// Returns idx such that mixed pairs are (nA[i], nB[idx[i]]). Also fills
// meanDpt with the achieved mean |pT2_i - pT2_idx[i]| (matching quality).
std::vector<int> mxBuildPartnerIdx(const std::vector<double>& pt2, int kNeighbors,
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
// Same-vs-mixed on one event set, with joint bootstrap of the difference
// =============================================================================

struct MxResult {
    MxAllPoints same, mix, diff, diffErr;
    double meanDpt; int N;
};

MxResult mxSameVsMixed(const std::vector<int>& nA, const std::vector<int>& nB,
                        const std::vector<double>& pt2, long long alphabet,
                        int kNeighbors, int nShuffles, int nBoot, TRandom3& rng,
                        const char* tag) {
    MxResult R; R.N = nA.size();
    std::vector<int> idx = mxBuildPartnerIdx(pt2, kNeighbors, rng, R.meanDpt);
    std::vector<int> nBmix(R.N);
    for (int i = 0; i < R.N; ++i) nBmix[i] = nB[idx[i]];
    R.same = mxAllPointEstimates(nA, nB, alphabet, nShuffles, rng);
    R.mix  = mxAllPointEstimates(nA, nBmix, alphabet, nShuffles, rng);
    R.diff.naive = R.same.naive - R.mix.naive;
    R.diff.mm = R.same.mm - R.mix.mm;
    R.diff.shuffle = R.same.shuffle - R.mix.shuffle;
    R.diff.nsb = R.same.nsb - R.mix.nsb;

    std::vector<double> bN(nBoot), bM(nBoot), bS(nBoot), bB(nBoot);
    for (int b = 0; b < nBoot; ++b) {
        std::vector<int> ra(R.N), rb(R.N); std::vector<double> rpt(R.N);
        for (int i = 0; i < R.N; ++i) {
            int s = rng.Integer(R.N);
            ra[i] = nA[s]; rb[i] = nB[s]; rpt[i] = pt2[s];
        }
        double dd; std::vector<int> ridx = mxBuildPartnerIdx(rpt, kNeighbors, rng, dd);
        std::vector<int> rbm(R.N);
        for (int i = 0; i < R.N; ++i) rbm[i] = rb[ridx[i]];
        MxAllPoints s1 = mxAllPointEstimates(ra, rb, alphabet, nShuffles, rng);
        MxAllPoints s2 = mxAllPointEstimates(ra, rbm, alphabet, nShuffles, rng);
        bN[b] = s1.naive - s2.naive; bM[b] = s1.mm - s2.mm;
        bS[b] = s1.shuffle - s2.shuffle; bB[b] = s1.nsb - s2.nsb;
    }
    auto stdv = [](std::vector<double>& v){ double m=0; for(double x:v) m+=x; m/=v.size();
        double s=0; for(double x:v) s+=(x-m)*(x-m); return std::sqrt(s/(v.size()-1)); };
    R.diffErr.naive = stdv(bN); R.diffErr.mm = stdv(bM);
    R.diffErr.shuffle = stdv(bS); R.diffErr.nsb = stdv(bB);

    printf("\n--- same vs. mixed %s (N=%d, k=%d, mean|dpT2|=%.2f GeV) ---\n",
           tag, R.N, kNeighbors, R.meanDpt);
    printf("%14s | %9s | %9s | %s\n", "method", "I_same", "I_mix", "I_same - I_mix");
    const char* lab[4] = {"naive","Miller-Madow","shuffle","NSB"};
    double s_[4]={R.same.naive,R.same.mm,R.same.shuffle,R.same.nsb};
    double m_[4]={R.mix.naive,R.mix.mm,R.mix.shuffle,R.mix.nsb};
    double d_[4]={R.diff.naive,R.diff.mm,R.diff.shuffle,R.diff.nsb};
    double e_[4]={R.diffErr.naive,R.diffErr.mm,R.diffErr.shuffle,R.diffErr.nsb};
    for (int k = 0; k < 4; ++k)
        printf("%14s | %9.5f | %9.5f | %+9.5f +/- %.5f\n", lab[k], s_[k], m_[k], d_[k], e_[k]);
    return R;
}

// =============================================================================
// Data loading (same tree layout as the other macros, plus jet2_pt)
// =============================================================================

void mxLoadTree(const char* rootFile, const TString& level,
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

// =============================================================================
// Main
// =============================================================================

void analyze_dijet_mixed(const char* rootFile, const char* levelIn = "truth",
                          long long alphabet = 80, int mixNeighbors = 10,
                          int nBoot = 50, int nShuffles = 30, int ptBins = 4,
                          int minBinEvents = 500,
                          const char* figsDir = "/Users/zhoudunmingtu/bnl_work/Work/MODELS/EBCs_toyMC/analysis/figs") {
    TString level(levelIn);
    gStyle->SetPadLeftMargin(0.15); gStyle->SetPadRightMargin(0.06);
    gStyle->SetPadBottomMargin(0.13); gStyle->SetPadTopMargin(0.09);
    gStyle->SetTitleOffset(1.5, "Y"); gStyle->SetTitleOffset(1.1, "X");
    gStyle->SetTitleSize(0.045, "XYZ"); gStyle->SetLabelSize(0.035, "XYZ");

    TString figsDirStr(figsDir);
    gSystem->mkdir(figsDirStr, kTRUE);
    TString outPrefix = figsDirStr + TString::Format("/dijet_mixed_root_%s", levelIn);
    std::cout << "Figures will be saved under: " << figsDirStr << "\n";

    TRandom3 rng(20260722);
    std::vector<int> nA, nB; std::vector<double> pt1, pt2;
    mxLoadTree(rootFile, level, nA, nB, pt1, pt2);
    std::cout << "Loaded " << nA.size() << " events.\n";

    // ---- whole sample ----
    MxResult W = mxSameVsMixed(nA, nB, pt2, alphabet, mixNeighbors, nShuffles,
                                nBoot, rng, "(whole sample)");

    // QA canvas: same vs mixed (left), differences (right)
    TCanvas* c1 = new TCanvas("cMxQA", "Mixed-event QA", 1300, 520);
    c1->Divide(2, 1);
    const char* lab[4] = {"naive","Miller-Madow","shuffle","NSB"};

    c1->cd(1);
    TH1D* hSame = new TH1D("hMxSame", "Same-event vs. mixed-event MI;;I(A:B) (nats)", 4, 0, 4);
    TH1D* hMix  = new TH1D("hMxMix", "", 4, 0, 4);
    double s_[4]={W.same.naive,W.same.mm,W.same.shuffle,W.same.nsb};
    double m_[4]={W.mix.naive,W.mix.mm,W.mix.shuffle,W.mix.nsb};
    for (int k = 0; k < 4; ++k) {
        hSame->SetBinContent(k+1, s_[k]); hSame->GetXaxis()->SetBinLabel(k+1, lab[k]);
        hMix->SetBinContent(k+1, m_[k]);
    }
    hSame->SetFillColorAlpha(kAzure+1, 0.7); hSame->SetStats(0);
    hMix->SetFillColorAlpha(kOrange+1, 0.55); hMix->SetStats(0);
    hSame->SetBarWidth(0.4); hSame->SetBarOffset(0.1);
    hMix->SetBarWidth(0.4);  hMix->SetBarOffset(0.5);
    double yMaxL = std::max(*std::max_element(s_, s_+4), *std::max_element(m_, m_+4));
    hSame->SetMaximum(yMaxL * 1.35); hSame->SetMinimum(0);
    hSame->Draw("BAR"); hMix->Draw("BAR SAME");
    TLegend* leg1 = new TLegend(0.55, 0.74, 0.90, 0.88);
    leg1->AddEntry(hSame, "same-event", "f");
    leg1->AddEntry(hMix, "mixed (pT2-matched)", "f");
    leg1->Draw();

    c1->cd(2);
    TH1D* hDiff = new TH1D("hMxDiff", "Beyond-kinematics MI (joint-bootstrap errors);;I_{same} - I_{mix} (nats)", 4, 0, 4);
    double d_[4]={W.diff.naive,W.diff.mm,W.diff.shuffle,W.diff.nsb};
    double e_[4]={W.diffErr.naive,W.diffErr.mm,W.diffErr.shuffle,W.diffErr.nsb};
    for (int k = 0; k < 4; ++k) {
        hDiff->SetBinContent(k+1, d_[k]); hDiff->SetBinError(k+1, e_[k]);
        hDiff->GetXaxis()->SetBinLabel(k+1, lab[k]);
    }
    hDiff->SetFillColorAlpha(kViolet+1, 0.6); hDiff->SetStats(0);
    {
        double lo = 1e300, hi = -1e300;
        for (int k = 0; k < 4; ++k) { lo = std::min(lo, d_[k]-e_[k]); hi = std::max(hi, d_[k]+e_[k]); }
        lo = std::min(lo, 0.0); hi = std::max(hi, 0.0);
        double r = hi - lo;
        hDiff->SetMinimum(lo - 0.15*r); hDiff->SetMaximum(hi + 0.25*r);
    }
    hDiff->Draw("E1 HIST");
    TLine* z1 = new TLine(gPad->GetUxmin(), 0, gPad->GetUxmax(), 0);
    z1->SetLineColor(kGray+1); z1->Draw();
    c1->SaveAs(outPrefix + "_mixed_qa.png");
    std::cout << "  saved " << outPrefix << "_mixed_qa.png\n";

    // ---- pT-differential ----
    if (ptBins > 0) {
        std::vector<double> sortedPt = pt1;
        std::sort(sortedPt.begin(), sortedPt.end());
        int Nall = sortedPt.size();
        std::vector<double> edges(ptBins + 1);
        for (int i = 0; i <= ptBins; ++i)
            edges[i] = sortedPt[std::min(Nall-1, (int)((double)i/ptBins*Nall))];
        edges[0] = sortedPt.front(); edges[ptBins] = sortedPt.back();

        std::vector<double> ptMid; std::vector<MxResult> binRes;
        for (int ib = 0; ib < ptBins; ++ib) {
            double lo = edges[ib], hi = edges[ib+1];
            std::vector<int> ba, bb; std::vector<double> bp2; double ptSum = 0;
            for (size_t i = 0; i < nA.size(); ++i) {
                bool in = (ib < ptBins-1) ? (pt1[i] >= lo && pt1[i] < hi)
                                          : (pt1[i] >= lo && pt1[i] <= hi);
                if (in) { ba.push_back(nA[i]); bb.push_back(nB[i]); bp2.push_back(pt2[i]); ptSum += pt1[i]; }
            }
            if ((int)ba.size() < minBinEvents) {
                printf("  skipping bin [%.1f,%.1f): only %d events\n", lo, hi, (int)ba.size());
                continue;
            }
            MxResult R = mxSameVsMixed(ba, bb, bp2, alphabet, mixNeighbors, nShuffles,
                                        nBoot, rng, Form("[%.1f,%.1f) GeV", lo, hi));
            ptMid.push_back(ptSum / ba.size());
            binRes.push_back(R);
        }

        if (!binRes.empty()) {
            int nb = binRes.size();
            TCanvas* c2 = new TCanvas("cMxPt", "Mixed-subtracted MI vs pT", 850, 560);
            TMultiGraph* mg = new TMultiGraph();
            int cols[4] = {kRed+1, kOrange+1, kBlue+1, kViolet+1};
            int mks[4] = {20, 21, 22, 23};
            double span = (nb > 1) ? (ptMid.back()-ptMid.front())/nb : 1.0;
            TGraphErrors* graphs[4];
            for (int k = 0; k < 4; ++k) {
                std::vector<double> x(nb), y(nb), ye(nb), zx(nb, 0.0);
                for (int i = 0; i < nb; ++i) {
                    x[i] = ptMid[i] + (k-1.5)*0.08*span;
                    double dv[4]={binRes[i].diff.naive,binRes[i].diff.mm,binRes[i].diff.shuffle,binRes[i].diff.nsb};
                    double de[4]={binRes[i].diffErr.naive,binRes[i].diffErr.mm,binRes[i].diffErr.shuffle,binRes[i].diffErr.nsb};
                    y[i] = dv[k]; ye[i] = de[k];
                }
                graphs[k] = new TGraphErrors(nb, x.data(), y.data(), zx.data(), ye.data());
                graphs[k]->SetMarkerStyle(mks[k]); graphs[k]->SetMarkerColor(cols[k]); graphs[k]->SetLineColor(cols[k]);
                mg->Add(graphs[k], "P");
            }
            mg->SetTitle("Beyond-kinematics MI vs. p_{T} (mixed-event subtracted);leading jet p_{T} (GeV);I_{same} - I_{mix} (nats)");
            mg->Draw("A");
            TLegend* leg = new TLegend(0.65, 0.68, 0.90, 0.88);
            for (int k = 0; k < 4; ++k) leg->AddEntry(graphs[k], lab[k], "lep");
            leg->Draw();
            TLine* z2 = new TLine(gPad->GetUxmin(), 0, gPad->GetUxmax(), 0);
            z2->SetLineColor(kGray+1); z2->Draw();
            c2->SaveAs(outPrefix + "_mixed_diff_vs_pt.png");
            std::cout << "  saved " << outPrefix << "_mixed_diff_vs_pt.png\n";
        }
    }
    std::cout << "\nDone.\n";
}
