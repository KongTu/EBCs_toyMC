// main_dijet_pp200.cc
//
// Generate pp collisions at sqrt(s) = 200 GeV (RHIC/STAR energy) and, for
// each generator-level back-to-back dijet event:
//
//   (a) save the full list of charged, final-state (stable) hadrons
//       ("tracks"), and reconstruct jets from them at TRUTH level (wide,
//       idealized acceptance, no tracking inefficiency) via PYTHIA 8's
//       built-in SlowJet anti-kT finder;
//
//   (b) on top of (a), ALSO reconstruct jets from the same event using a
//       second SlowJet instance restricted to a STAR-TPC-like detector
//       acceptance (pT > trkPtMinDet, |eta| < trkEtaMaxDet) and a
//       pT-dependent tracking efficiency, via a custom SlowJetHook.
//
// Both truth-level and detector-level jet kinematics + charged-hadron
// multiplicity per jet are written to the same ROOT TTree, so the
// downstream Python entropy/MI analysis can be run unchanged against
// either "_truth" or "_det" branches to see exactly how acceptance +
// efficiency alter the measurement.
//
// Physics choices (all overridable via command-line flags, see parseArgs
// below):
//   - sqrt(s) = 200 GeV, HardQCD:all = on, pTHat cut to focus on genuine
//     dijet events
//   - anti-kT jets (SlowJet power = -1), R = 0.4, CHARGED final-state
//     particles only (SlowJet `select = 3`)
//   - TRUTH level: wide constituent |eta| < etaMaxConstTruth (default 4.0,
//     effectively "no realistic detector limit" for RHIC kinematics), no
//     pT threshold, no inefficiency. Jet-axis |y| < yMaxJetTruth (3.0).
//   - DETECTOR level: constituent pT > trkPtMinDet (default 0.15 GeV),
//     |eta| < trkEtaMaxDet (default 1.0), plus a pT-dependent tracking
//     efficiency turn-on (see trackEfficiency() below) -- applied via
//     SlowJetHook, so the *same* jet algorithm/radius is used, only the
//     input particle list differs. Jet-axis |y| < yMaxJetDet (0.6, keeps
//     the R=0.4 cone inside |eta|<1.0).
//   - back-to-back selection (both levels): |Delta_phi - pi| < dPhiCut
//
// Event selection: the EVENT-LEVEL sample kept (whether an entry is
// written at all) is defined by the TRUTH-level dijet selection -- this
// is the standard "generate & select at truth level, then evaluate
// detector response on the same events" pattern. If the detector-level
// reconstruction *also* finds a valid back-to-back dijet in that same
// event, its kinematics are stored and `hasDetDijet=1`; if efficiency/
// acceptance losses mean no detector-level dijet survives, `hasDetDijet=0`
// and the det-level branches are filled with sentinel values (-999) --
// filter on `hasDetDijet==1` in the analysis for a clean detector-level
// sample.
//
// Build: see accompanying Makefile.

#include "Pythia8/Pythia.h"
#include <TFile.h>
#include <TTree.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>

using namespace Pythia8;

// =====================================================================
// Detector-level particle selection hook: applies a kinematic acceptance
// window (pT, |eta|) and a pT-dependent tracking efficiency to decide
// which particles SlowJet sees when reconstructing "detector-level" jets.
//
// Efficiency model: a simple rising turn-on saturating at a plateau,
//     eff(pT) = effPlateau * (1 - exp(-(pT - pTMin) / pTScale))   for pT > pTMin
// This is a schematic stand-in for a real STAR TPC efficiency curve (flat
// efficiency all the way down to a hard pT threshold is not physically
// realistic -- real tracking efficiency turns on gradually just above
// threshold due to track curvature/finite lever arm). Replace with a
// real embedding-derived efficiency table if you have one; the hook
// interface here is exactly where that substitution would go.
// =====================================================================
double trackEfficiency(double pT, double pTMin, double effPlateau, double pTScale) {
    if (pT <= pTMin) return 0.0;
    return effPlateau * (1.0 - std::exp(-(pT - pTMin) / pTScale));
}

class DetectorTrackHook : public SlowJetHook {
public:
    DetectorTrackHook(double pTMinIn, double etaMaxIn, double effPlateauIn,
                       double pTScaleIn, Rndm* rndmPtrIn)
        : pTMin(pTMinIn), etaMax(etaMaxIn), effPlateau(effPlateauIn),
          pTScale(pTScaleIn), rndmPtr(rndmPtrIn) {}

    // Called by SlowJet once per candidate particle index iSel. Return
    // false to exclude it from clustering; return true (optionally
    // modifying pSel/mSel) to include it as-is.
    bool include(int iSel, const Event& event, Vec4& pSel, double& mSel) override {
        const Particle& p = event[iSel];
        // Redundant with select=3 at construction, but explicit/safe in
        // case a hook fully overrides the built-in selection machinery.
        if (!p.isFinal() || !p.isCharged()) return false;
        if (p.pT() < pTMin) return false;
        if (std::abs(p.eta()) > etaMax) return false;

        double eff = trackEfficiency(p.pT(), pTMin, effPlateau, pTScale);
        if (rndmPtr->flat() > eff) return false;   // tracking-efficiency loss

        pSel = p.p();
        mSel = p.m();
        return true;
    }

private:
    double pTMin, etaMax, effPlateau, pTScale;
    Rndm* rndmPtr;
};

// ---------------------------------------------------------------------
// Minimal command-line argument parsing: --flag value
// ---------------------------------------------------------------------
struct Config {
    long long nEvents      = 10000;  // number of GENERATED events to try
    double pTHatMin        = 40;      // GeV, parton-level hard-process cut
    double jetR            = 0.4;      // anti-kT radius (shared, both levels)
    double jetPtMin        = 20.0;      // GeV, minimum jet pT (shared, both levels)
    double dPhiCut         = 0.4;      // back-to-back cut, both levels

    // truth level (wide, idealized)
    double etaMaxConstTruth = 4.0;
    double yMaxJetTruth      = 3.0;

    // detector level (STAR-TPC-like)
    double trkPtMinDet     = 0.15;     // GeV
    double trkEtaMaxDet    = 1.0;
    double yMaxJetDet       = 0.6;     // keeps R=0.4 cone inside |eta|<1.0
    double trkEffPlateau   = 0.80;     // plateau tracking efficiency
    double trkEffPtScale   = 0.15;     // GeV, efficiency turn-on scale

    int    seed             = 42;
    std::string outFile     = "pythia_dijet_pp200_full.root";
};

Config parseArgs(int argc, char* argv[]) {
    Config c;
    for (int i = 1; i < argc - 1; ++i) {
        std::string flag = argv[i];
        std::string val  = argv[i+1];
        if      (flag == "--nEvents")        c.nEvents          = std::atoll(val.c_str());
        else if (flag == "--pTHatMin")       c.pTHatMin         = std::atof(val.c_str());
        else if (flag == "--jetR")           c.jetR             = std::atof(val.c_str());
        else if (flag == "--jetPtMin")       c.jetPtMin         = std::atof(val.c_str());
        else if (flag == "--dPhiCut")        c.dPhiCut          = std::atof(val.c_str());
        else if (flag == "--etaMaxConstTruth") c.etaMaxConstTruth = std::atof(val.c_str());
        else if (flag == "--yMaxJetTruth")   c.yMaxJetTruth     = std::atof(val.c_str());
        else if (flag == "--trkPtMinDet")    c.trkPtMinDet      = std::atof(val.c_str());
        else if (flag == "--trkEtaMaxDet")   c.trkEtaMaxDet     = std::atof(val.c_str());
        else if (flag == "--yMaxJetDet")     c.yMaxJetDet       = std::atof(val.c_str());
        else if (flag == "--trkEffPlateau")  c.trkEffPlateau    = std::atof(val.c_str());
        else if (flag == "--trkEffPtScale")  c.trkEffPtScale    = std::atof(val.c_str());
        else if (flag == "--seed")           c.seed             = std::atoi(val.c_str());
        else if (flag == "--out")            c.outFile          = val;
    }
    return c;
}

int main(int argc, char* argv[]) {

    Config cfg = parseArgs(argc, argv);

    std::cout << "==================================================\n"
               << " PYTHIA 8 pp dijet generator, sqrt(s) = 200 GeV\n"
               << " (truth-level + detector-level jet reconstruction)\n"
               << "==================================================\n"
               << "  nEvents (attempted)     : " << cfg.nEvents         << "\n"
               << "  pTHatMin                : " << cfg.pTHatMin        << " GeV\n"
               << "  jet R (anti-kT, shared) : " << cfg.jetR            << "\n"
               << "  jet pT min (shared)     : " << cfg.jetPtMin        << " GeV\n"
               << "  back-to-back cut        : |dphi - pi| < " << cfg.dPhiCut << "\n"
               << "  --- truth level ---\n"
               << "  constituent |eta| max   : " << cfg.etaMaxConstTruth << "\n"
               << "  jet |y| max             : " << cfg.yMaxJetTruth     << "\n"
               << "  --- detector level ---\n"
               << "  track pT min            : " << cfg.trkPtMinDet     << " GeV\n"
               << "  track |eta| max         : " << cfg.trkEtaMaxDet    << "\n"
               << "  jet |y| max             : " << cfg.yMaxJetDet      << "\n"
               << "  tracking eff. plateau   : " << cfg.trkEffPlateau   << "\n"
               << "  tracking eff. pT scale  : " << cfg.trkEffPtScale   << " GeV\n"
               << "  random seed             : " << cfg.seed            << "\n"
               << "  output file             : " << cfg.outFile         << "\n"
               << "==================================================\n";

    // -------------------------------------------------------------
    // PYTHIA 8 setup
    // -------------------------------------------------------------
    Pythia pythia;

    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 200.");

    pythia.readString("HardQCD:all = on");
    pythia.readString("PhaseSpace:pTHatMin = " + std::to_string(cfg.pTHatMin));

    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:seed = " + std::to_string(cfg.seed));
    pythia.readString("PartonLevel:MPI = off");

    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");
    pythia.readString("Print:quiet = on");

    if (!pythia.init()) {
        std::cerr << "ERROR: Pythia failed to initialize.\n";
        return 1;
    }

    // -------------------------------------------------------------
    // Jet finders
    // -------------------------------------------------------------
    // Truth level: wide/idealized acceptance, built-in select=3 (charged
    // final-state only) + etaMax is sufficient, no hook needed.
    SlowJet slowJetTruth(-1, cfg.jetR, cfg.jetPtMin, cfg.etaMaxConstTruth,
                          /*select=*/3, /*massSet=*/1);

    // Detector level: constructor etaMax left generous (25 = effectively
    // unlimited); the REAL pT/eta/efficiency cuts are enforced entirely
    // inside DetectorTrackHook, so there is no ambiguity about which
    // layer is doing the cutting.
    DetectorTrackHook detHook(cfg.trkPtMinDet, cfg.trkEtaMaxDet,
                               cfg.trkEffPlateau, cfg.trkEffPtScale,
                               &pythia.rndm);
    SlowJet slowJetDet(-1, cfg.jetR, cfg.jetPtMin, 25.0,
                        /*select=*/3, /*massSet=*/1, &detHook);

    // -------------------------------------------------------------
    // ROOT output
    // -------------------------------------------------------------
    TFile* fout = new TFile(cfg.outFile.c_str(), "RECREATE");
    TTree* tree = new TTree("dijet", "PYTHIA8 pp 200 GeV back-to-back charged dijets: truth + detector level");

    Double_t pTHat, weight;

    // (a) full charged stable-hadron list for this event (truth kinematics;
    // detector acceptance/efficiency can be re-applied downstream from
    // these if desired, independent of the C++-side detector jets below).
    std::vector<Float_t> trk_pt, trk_eta, trk_phi;
    std::vector<Int_t>   trk_pid;

    // truth-level dijet (defines event selection)
    Int_t    nJetsTruth;
    Double_t jet1_pt_truth, jet1_y_truth, jet1_phi_truth; Int_t jet1_mult_truth;
    Double_t jet2_pt_truth, jet2_y_truth, jet2_phi_truth; Int_t jet2_mult_truth;
    Double_t dPhi_truth;

    // detector-level dijet (may or may not be found in a given event)
    Int_t    hasDetDijet, nJetsDet;
    Double_t jet1_pt_det, jet1_y_det, jet1_phi_det; Int_t jet1_mult_det;
    Double_t jet2_pt_det, jet2_y_det, jet2_phi_det; Int_t jet2_mult_det;
    Double_t dPhi_det;

    tree->Branch("pTHat",  &pTHat,  "pTHat/D");
    tree->Branch("weight", &weight, "weight/D");

    tree->Branch("trk_pt",  &trk_pt);
    tree->Branch("trk_eta", &trk_eta);
    tree->Branch("trk_phi", &trk_phi);
    tree->Branch("trk_pid", &trk_pid);

    tree->Branch("nJetsTruth",     &nJetsTruth,     "nJetsTruth/I");
    tree->Branch("jet1_pt_truth",  &jet1_pt_truth,  "jet1_pt_truth/D");
    tree->Branch("jet1_y_truth",   &jet1_y_truth,   "jet1_y_truth/D");
    tree->Branch("jet1_phi_truth", &jet1_phi_truth, "jet1_phi_truth/D");
    tree->Branch("jet1_mult_truth",&jet1_mult_truth,"jet1_mult_truth/I");  // N_A, truth
    tree->Branch("jet2_pt_truth",  &jet2_pt_truth,  "jet2_pt_truth/D");
    tree->Branch("jet2_y_truth",   &jet2_y_truth,   "jet2_y_truth/D");
    tree->Branch("jet2_phi_truth", &jet2_phi_truth, "jet2_phi_truth/D");
    tree->Branch("jet2_mult_truth",&jet2_mult_truth,"jet2_mult_truth/I");  // N_B, truth
    tree->Branch("dPhi_truth",     &dPhi_truth,     "dPhi_truth/D");

    tree->Branch("hasDetDijet",  &hasDetDijet,  "hasDetDijet/I");
    tree->Branch("nJetsDet",     &nJetsDet,     "nJetsDet/I");
    tree->Branch("jet1_pt_det",  &jet1_pt_det,  "jet1_pt_det/D");
    tree->Branch("jet1_y_det",   &jet1_y_det,   "jet1_y_det/D");
    tree->Branch("jet1_phi_det", &jet1_phi_det, "jet1_phi_det/D");
    tree->Branch("jet1_mult_det",&jet1_mult_det,"jet1_mult_det/I");  // N_A, detector
    tree->Branch("jet2_pt_det",  &jet2_pt_det,  "jet2_pt_det/D");
    tree->Branch("jet2_y_det",   &jet2_y_det,   "jet2_y_det/D");
    tree->Branch("jet2_phi_det", &jet2_phi_det, "jet2_phi_det/D");
    tree->Branch("jet2_mult_det",&jet2_mult_det,"jet2_mult_det/I");  // N_B, detector
    tree->Branch("dPhi_det",     &dPhi_det,     "dPhi_det/D");

    // -------------------------------------------------------------
    // Event loop
    // -------------------------------------------------------------
    long long nAccepted  = 0;
    long long nGenerated = 0;
    long long nWithDetDijet = 0;

    for (long long iEvent = 0; iEvent < cfg.nEvents; ++iEvent) {
        if (!pythia.next()) continue;
        ++nGenerated;

        // ---- truth-level jet finding: defines whether we keep this event ----
        slowJetTruth.analyze(pythia.event);
        int nJetsFoundTruth = slowJetTruth.sizeJet();
        if (nJetsFoundTruth < 2) continue;

        double pt1t = slowJetTruth.pT(0), y1t = slowJetTruth.y(0), phi1t = slowJetTruth.phi(0);
        double pt2t = slowJetTruth.pT(1), y2t = slowJetTruth.y(1), phi2t = slowJetTruth.phi(1);

        if (std::abs(y1t) > cfg.yMaxJetTruth || std::abs(y2t) > cfg.yMaxJetTruth) continue;

        double deltaPhiT = std::abs(phi1t - phi2t);
        if (deltaPhiT > M_PI) deltaPhiT = 2.0*M_PI - deltaPhiT;
        if (std::abs(deltaPhiT - M_PI) > cfg.dPhiCut) continue;

        // --- event kept: fill truth-level branches ---
        pTHat  = pythia.info.pTHat();
        weight = pythia.info.weight();

        nJetsTruth = nJetsFoundTruth;
        jet1_pt_truth = pt1t; jet1_y_truth = y1t; jet1_phi_truth = phi1t;
        jet1_mult_truth = slowJetTruth.multiplicity(0);
        jet2_pt_truth = pt2t; jet2_y_truth = y2t; jet2_phi_truth = phi2t;
        jet2_mult_truth = slowJetTruth.multiplicity(1);
        dPhi_truth = deltaPhiT;

        // --- (a) save full charged stable-hadron list for this event ---
        trk_pt.clear(); trk_eta.clear(); trk_phi.clear(); trk_pid.clear();
        for (int i = 0; i < pythia.event.size(); ++i) {
            const Particle& p = pythia.event[i];
            if (!p.isFinal() || !p.isCharged()) continue;
            trk_pt.push_back(static_cast<Float_t>(p.pT()));
            trk_eta.push_back(static_cast<Float_t>(p.eta()));
            trk_phi.push_back(static_cast<Float_t>(p.phi()));
            trk_pid.push_back(static_cast<Int_t>(p.id()));
        }

        // --- (b) detector-level jet finding, same event ---
        slowJetDet.analyze(pythia.event);
        int nJetsFoundDet = slowJetDet.sizeJet();
        hasDetDijet = 0;
        nJetsDet = nJetsFoundDet;
        jet1_pt_det = -999; jet1_y_det = -999; jet1_phi_det = -999; jet1_mult_det = -1;
        jet2_pt_det = -999; jet2_y_det = -999; jet2_phi_det = -999; jet2_mult_det = -1;
        dPhi_det = -1;

        if (nJetsFoundDet >= 2) {
            double pt1d = slowJetDet.pT(0), y1d = slowJetDet.y(0), phi1d = slowJetDet.phi(0);
            double pt2d = slowJetDet.pT(1), y2d = slowJetDet.y(1), phi2d = slowJetDet.phi(1);

            bool passEta = (std::abs(y1d) <= cfg.yMaxJetDet && std::abs(y2d) <= cfg.yMaxJetDet);
            double deltaPhiD = std::abs(phi1d - phi2d);
            if (deltaPhiD > M_PI) deltaPhiD = 2.0*M_PI - deltaPhiD;
            bool passB2B = std::abs(deltaPhiD - M_PI) <= cfg.dPhiCut;

            if (passEta && passB2B) {
                hasDetDijet = 1;
                jet1_pt_det = pt1d; jet1_y_det = y1d; jet1_phi_det = phi1d;
                jet1_mult_det = slowJetDet.multiplicity(0);
                jet2_pt_det = pt2d; jet2_y_det = y2d; jet2_phi_det = phi2d;
                jet2_mult_det = slowJetDet.multiplicity(1);
                dPhi_det = deltaPhiD;
                ++nWithDetDijet;
            }
        }

        tree->Fill();
        ++nAccepted;

        if (nAccepted % 50000 == 0) {
            std::cout << "  ... " << nAccepted << " truth dijet events accepted "
                      << "(from " << nGenerated << " generated; "
                      << nWithDetDijet << " also have a detector-level dijet)\n";
        }
    }

    pythia.stat();

    tree->Write();
    fout->Close();

    std::cout << "==================================================\n"
               << "Done. Generated: " << nGenerated
               << "   Accepted (truth) dijets: " << nAccepted
               << "   (efficiency " << 100.0*nAccepted/std::max(1LL,nGenerated) << "%)\n"
               << "   Of those, also passing detector-level dijet selection: "
               << nWithDetDijet << " (" << 100.0*nWithDetDijet/std::max(1LL,nAccepted) << "%)\n"
               << "Wrote: " << cfg.outFile << "\n"
               << "==================================================\n";

    return 0;
}
