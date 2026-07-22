# EBCs_toyMC

Entanglement/correlation studies of back-to-back dijet hadron multiplicities:
measuring mutual information (MI), conditional entropy S(B|A), and normalized
mutual information NMI = I(A:B)/min(S_A, S_B) between the two jets, with
careful treatment of finite-sample estimator bias (naive / Miller-Madow /
shuffle-subtraction / NSB estimators, validated by closure tests).

Physics idea (Kharzeev-Levin): a jet's entanglement entropy maps onto the
Shannon entropy of its hadron multiplicity distribution, so MI and
conditional entropy between two jets become measurable from multiplicity
counting. Classically S(B|A) >= 0 and NMI <= 1 always; a significant
violation would signal that the dijet cannot be described by a classical
joint distribution. Most of the practical work is statistical: the naive
plug-in estimator is biased upward at finite N, so bias correction and
uncertainty calibration (closure tests) are essential before any claim.

## Repository layout

```
.
├── src/               PYTHIA 8 event generator (main_dijet_pp_full.cc):
│                      pp dijets, anti-kT R=0.4, charged particles,
│                      truth-level + detector-level (STAR-TPC-like) output
│                      to a ROOT TTree
├── analysis/          Analysis pipeline, each in Python AND ROOT/C++:
│                      - analyze_dijet_entropy.{py,C}: main analysis
│                        (entropies, MI, S(B|A), NMI vs. leading-jet pT,
│                        multi-method comparison)
│                      - closure_test.{py,C}: whole-sample closure test
│                        (calibrated pseudo-truth, NSB pull distributions)
│                      - closure_test_pt.{py,C}: pT-differential closure
│                        test (bin-specific model fits)
│                      - run_analysis.sh: interactive menu / quick-mode
│                        wrapper so you don't have to remember arguments
├── python_notebook/   Toy Monte Carlo study (known ground truth):
│                      estimator bias demonstration and validation of all
│                      four estimators, detection-sensitivity curves
├── documents/         LaTeX note (methods write-up: toy study, PYTHIA8
│                      validation, closure tests) + figures
└── README.md
```

## Quick start

```bash
cd analysis
./run_analysis.sh                          # interactive menu (6 options)
./run_analysis.sh 2 myfile.root            # quick mode: option 2, all defaults
./run_analysis.sh 2 myfile.root -y         # same, no confirmation prompt
```

Python scripts need: `pip install uproot numpy scipy matplotlib`

The ROOT (.C) macros are run from `$HOME` by the wrapper (Google-Drive path
workaround) and are ACLiC-compiled (`.C+`). Figures go to the `figs/`
directory configured in each script.

## run pythia (personal reminder)

compile:

```
g++ -std=c++17 -O2 main_dijet_pp_full.cc -o run_pythia   -I/Users/zhoudunmingtu/bnl_work/Work/EIC/Pythia/Pythia8/pythia8235/include $(root-config --cflags)   -L/Users/zhoudunmingtu/bnl_work/Work/EIC/Pythia/Pythia8/pythia8235/lib -lpythia8   $(root-config --glibs)
```

Run:

```
export PYTHIA8DATA=/Users/zhoudunmingtu/bnl_work/Work/EIC/Pythia/Pythia8/pythia8235/share/Pythia8/xmldoc
```

```
./run_pythia
```
