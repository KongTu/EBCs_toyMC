#!/usr/bin/env bash
#
# run_analysis.sh
#
# Interactive wrapper for the six analysis scripts in this directory
# (analyze_dijet_entropy / closure_test / closure_test_pt, each in
# Python and ROOT). Prompts for each argument with its default shown --
# just hit Enter to accept the default, or type a value to override it.
# Prints the exact command before running it, so you can also just copy
# that line directly next time instead of going through the menu.
#
# Usage:
#   ./run_analysis.sh                     interactive menu
#   ./run_analysis.sh 1                   jump straight to menu item 1, still prompts for each argument
#   ./run_analysis.sh 1 myfile.root       QUICK MODE: run item 1 on myfile.root with every other
#                                         argument at its default -- no prompts at all
#   ./run_analysis.sh 1 myfile.root -y    same, and also skip the final "run this now?" confirmation
#
# The ROOT (.C) scripts are run from $HOME (see ROOT_SCRIPT_DIR below --
# update this one line if you move the folder) and control returns to
# your original directory afterward either way. This sidesteps a
# Google-Drive-sync path issue in the user's setup. The Python scripts
# are unaffected and still run from wherever you currently are.
#
# Written for bash 3.2 (macOS default) compatibility -- no associative
# arrays, no ${var,,} case conversion, etc.

set -u

# Folder containing the .C macros -- referenced by full path since we cd
# to $HOME before running ROOT (see run_root_macro below).
ROOT_SCRIPT_DIR="/Users/zhoudunmingtu/bnl_work/Work/MODELS/EBCs_toyMC/analysis"

# ---- small helpers ------------------------------------------------------

# ask VAR "prompt text" "default"
# In QUICK mode (triggered by the "<choice> <rootfile>" command-line
# form), every ask/ask_yn silently takes its default -- no prompt at all.
ask() {
    local __var="$1" __prompt="$2" __default="$3" __input
    if [ "${QUICK:-false}" = "true" ]; then
        eval "${__var}=\"\${__default}\""
        return
    fi
    read -r -p "  ${__prompt} [${__default}]: " __input
    if [ -z "${__input}" ]; then
        eval "${__var}=\"\${__default}\""
    else
        eval "${__var}=\"\${__input}\""
    fi
}

# ask_yn VAR "prompt text" "default(y/n)"
ask_yn() {
    local __var="$1" __prompt="$2" __default="$3" __input
    if [ "${QUICK:-false}" = "true" ]; then
        __input="${__default}"
    else
        read -r -p "  ${__prompt} [y/n, default ${__default}]: " __input
        __input="${__input:-${__default}}"
    fi
    case "${__input}" in
        y|Y|yes|YES) eval "${__var}=true" ;;
        *)           eval "${__var}=false" ;;
    esac
}

confirm_and_run() {
    echo ""
    echo "  About to run:"
    echo "  ------------------------------------------------------------"
    echo "  $1"
    echo "  ------------------------------------------------------------"
    if [ "${AUTO_YES:-false}" = "true" ]; then
        eval "$1"
        return
    fi
    read -r -p "  Run this now? [Y/n]: " go
    go="${go:-Y}"
    case "${go}" in
        y|Y|yes|YES) eval "$1" ;;
        *) echo "  Skipped. (You can copy the line above and run it yourself later.)" ;;
    esac
}

# Resolves a possibly-relative path to an absolute one, against the
# CURRENT directory (i.e. before any pushd). Needed because the ROOT
# macros run from $HOME (run_root_macro below) -- a relative path typed
# at the prompt would otherwise get resolved against $HOME instead of
# wherever you actually ran this script from, and silently fail.
to_abs_path() {
    local __path="$1"
    case "${__path}" in
        /*) printf '%s' "${__path}" ;;
        *)  printf '%s' "$(pwd)/${__path}" ;;
    esac
}

# Runs a ROOT macro invocation from $HOME (sidesteps a Google-Drive-sync
# path issue), then always returns to the original directory afterward,
# whether or not root succeeded.
#   $1 = macro invocation, e.g. 'analyze_dijet_entropy.C+("f.root",...)'
run_root_macro() {
    local __full_call="${ROOT_SCRIPT_DIR}/$1"
    local __cmd="pushd \"\$HOME\" > /dev/null && root -l -q '${__full_call}' ; popd > /dev/null"
    confirm_and_run "${__cmd}"
}

require_root_file() {
    if [ -z "${ROOT_FILE:-}" ]; then
        ask ROOT_FILE "Path to your PYTHIA8 dijet ROOT file" ""
        while [ -z "${ROOT_FILE}" ]; do
            echo "  (required -- this one has no sensible default)"
            ask ROOT_FILE "Path to your PYTHIA8 dijet ROOT file" ""
        done
    fi
    # Always resolve to an absolute path -- covers both the interactive
    # prompt above and the quick-mode "<choice> <rootfile>" form, and is
    # a no-op if the path was already absolute. Harmless for the Python
    # scripts (which never change directory), essential for the ROOT
    # ones (which do).
    ROOT_FILE="$(to_abs_path "${ROOT_FILE}")"
}

pause() { read -r -p "Press Enter to return to the menu..." _; }

# ---- 1: analyze_dijet_entropy.py --------------------------------------

run_analyze_py() {
    echo ""
    echo "== analyze_dijet_entropy.py =="
    require_root_file
    ask LEVEL "Level (truth = wide/idealized, det = detector acceptance+efficiency)" "truth"
    ask ALPHABET "Alphabet (fixed upper bound on plausible multiplicity per jet)" "80"
    ask PT_BINS "Number of leading-jet-pT bins" "4"
    ask PT_BIN_MODE "pT bin mode (quantile or fixed)" "quantile"
    ask MIN_BIN_EVENTS "Warn if a pT bin has fewer events than this" "30"
    ask N_BOOT_COMPARE "Bootstrap resamples per pT bin per method (method comparison + summary)" "50"
    ask N_SHUFFLES_INNER "Shuffles per bootstrap replicate (shuffle-correction estimator)" "30"
    ask_yn SKIP_METHOD_COMPARISON "Skip the multi-method comparison plots (faster)?" "n"
    ask_yn SKIP_PT_DIFFERENTIAL "Skip the pT-differential analysis entirely (faster)?" "n"
    ask FIGS_DIR "Directory to save figures in" "figs"

    CMD="python3 analyze_dijet_entropy.py \"${ROOT_FILE}\" --level ${LEVEL} --alphabet ${ALPHABET} --pt-bins ${PT_BINS} --pt-bin-mode ${PT_BIN_MODE} --min-bin-events ${MIN_BIN_EVENTS} --n-boot-compare ${N_BOOT_COMPARE} --n-shuffles-inner ${N_SHUFFLES_INNER} --figs-dir \"${FIGS_DIR}\""
    if [ "${SKIP_METHOD_COMPARISON}" = "true" ]; then CMD="${CMD} --skip-method-comparison"; fi
    if [ "${SKIP_PT_DIFFERENTIAL}" = "true" ]; then CMD="${CMD} --skip-pt-differential"; fi

    confirm_and_run "${CMD}"
}

# ---- 2: analyze_dijet_entropy.C ---------------------------------------

run_analyze_root() {
    echo ""
    echo "== analyze_dijet_entropy.C (ROOT, ACLiC-compiled, runs from \$HOME) =="
    require_root_file
    ask LEVEL "Level (truth = wide/idealized, det = detector acceptance+efficiency)" "truth"
    ask ALPHABET "Alphabet (fixed upper bound on plausible multiplicity per jet)" "80"
    ask N_BOOT "Bootstrap resamples for the whole-sample S(B|A) noise floor" "200"
    ask PT_BINS "Number of leading-jet-pT bins" "4"
    ask_yn COMPARE_METHODS "Run the multi-method comparison plots (slower)?" "y"
    ask N_BOOT_COMPARE "Bootstrap resamples per pT bin per method (if comparison enabled)" "50"
    ask N_SHUFFLES_INNER "Shuffles per bootstrap replicate (shuffle-correction estimator)" "30"
    ask FIGS_DIR "Directory to save figures in (absolute path recommended)" \
        "/Users/zhoudunmingtu/bnl_work/Work/MODELS/EBCs_toyMC/analysis/figs"
    FIGS_DIR="$(to_abs_path "${FIGS_DIR}")"

    MACRO="analyze_dijet_entropy.C+(\"${ROOT_FILE}\",\"${LEVEL}\",${ALPHABET},${N_BOOT},${PT_BINS},${COMPARE_METHODS},${N_BOOT_COMPARE},${N_SHUFFLES_INNER},\"${FIGS_DIR}\")"
    run_root_macro "${MACRO}"
}

# ---- 3: closure_test.py -----------------------------------------------

run_closure_py() {
    echo ""
    echo "== closure_test.py =="
    require_root_file
    ask LEVEL "Level (truth or det)" "truth"
    ask ALPHABET "Alphabet (same value you used in the real analysis)" "80"
    ask N_REPEATS "Independent draws for the closure test (higher = more trustworthy pull stats)" "30"
    ask N_BOOT "Bootstrap resamples per repeat for NSB uncertainty" "50"
    ask N_SHUFFLES "Shuffles for the shuffle-correction estimator" "100"
    ask REF_N "Reference sample size for pseudo-truth" "5000000"
    ask CLOSURE_N "Sample size per repeat (blank = same as your real dataset's N)" ""
    ask FIGS_DIR "Directory to save figures in" "figs"

    CMD="python3 closure_test.py \"${ROOT_FILE}\" --level ${LEVEL} --alphabet ${ALPHABET} --n-repeats ${N_REPEATS} --n-boot ${N_BOOT} --n-shuffles ${N_SHUFFLES} --ref-n ${REF_N} --figs-dir \"${FIGS_DIR}\""
    if [ -n "${CLOSURE_N}" ]; then CMD="${CMD} --closure-n ${CLOSURE_N}"; fi

    confirm_and_run "${CMD}"
}

# ---- 4: closure_test.C -------------------------------------------------

run_closure_root() {
    echo ""
    echo "== closure_test.C (ROOT, ACLiC-compiled, runs from \$HOME) =="
    require_root_file
    ask LEVEL "Level (truth or det)" "truth"
    ask ALPHABET "Alphabet (same value you used in the real analysis)" "80"
    ask N_REPEATS "Independent draws for the closure test" "30"
    ask N_BOOT "Bootstrap resamples per repeat for NSB uncertainty" "50"
    ask N_SHUFFLES "Shuffles for the shuffle-correction estimator" "100"
    ask REF_N "Reference sample size for pseudo-truth" "2000000"
    ask CLOSURE_N "Sample size per repeat (-1 = same as your real dataset's N)" "-1"
    ask FIGS_DIR "Directory to save figures in (absolute path recommended)" \
        "/Users/zhoudunmingtu/bnl_work/Work/MODELS/EBCs_toyMC/analysis/figs"
    FIGS_DIR="$(to_abs_path "${FIGS_DIR}")"

    MACRO="closure_test.C+(\"${ROOT_FILE}\",\"${LEVEL}\",${ALPHABET},${N_REPEATS},${N_BOOT},${N_SHUFFLES},${REF_N},${CLOSURE_N},\"${FIGS_DIR}\")"
    run_root_macro "${MACRO}"
}

# ---- 5: closure_test_pt.py ---------------------------------------------

run_closure_pt_py() {
    echo ""
    echo "== closure_test_pt.py (this is the expensive one: bins x repeats x bootstrap) =="
    require_root_file
    ask LEVEL "Level (truth or det)" "truth"
    ask ALPHABET "Alphabet (same value you used in the real analysis)" "80"
    ask PT_BINS "Number of leading-jet-pT bins" "4"
    ask PT_BIN_MODE "pT bin mode (quantile or fixed)" "quantile"
    ask MIN_EVENTS "Skip bins with fewer events than this" "200"
    ask N_REPEATS "Independent draws PER BIN" "100"
    ask N_BOOT "Bootstrap resamples per repeat for NSB uncertainty" "150"
    ask N_SHUFFLES "Shuffles for the shuffle-correction estimator" "100"
    ask REF_N "Reference sample size for pseudo-truth, PER BIN" "1000000"
    ask FIGS_DIR "Directory to save figures in" "figs"

    CMD="python3 closure_test_pt.py \"${ROOT_FILE}\" --level ${LEVEL} --alphabet ${ALPHABET} --pt-bins ${PT_BINS} --pt-bin-mode ${PT_BIN_MODE} --min-events ${MIN_EVENTS} --n-repeats ${N_REPEATS} --n-boot ${N_BOOT} --n-shuffles ${N_SHUFFLES} --ref-n ${REF_N} --figs-dir \"${FIGS_DIR}\""

    confirm_and_run "${CMD}"
}

# ---- 6: closure_test_pt.C ----------------------------------------------

run_closure_pt_root() {
    echo ""
    echo "== closure_test_pt.C (ROOT, ACLiC-compiled, runs from \$HOME -- this is the expensive one) =="
    require_root_file
    ask LEVEL "Level (truth or det)" "truth"
    ask ALPHABET "Alphabet (same value you used in the real analysis)" "80"
    ask PT_BINS "Number of leading-jet-pT bins" "4"
    ask N_REPEATS "Independent draws PER BIN" "100"
    ask N_BOOT "Bootstrap resamples per repeat for NSB uncertainty" "150"
    ask N_SHUFFLES "Shuffles for the shuffle-correction estimator" "100"
    ask REF_N "Reference sample size for pseudo-truth, PER BIN" "1000000"
    ask MIN_EVENTS "Skip bins with fewer events than this" "200"
    ask FIGS_DIR "Directory to save figures in (absolute path recommended)" \
        "/Users/zhoudunmingtu/bnl_work/Work/MODELS/EBCs_toyMC/analysis/figs"
    FIGS_DIR="$(to_abs_path "${FIGS_DIR}")"

    MACRO="closure_test_pt.C+(\"${ROOT_FILE}\",\"${LEVEL}\",${ALPHABET},${PT_BINS},${N_REPEATS},${N_BOOT},${N_SHUFFLES},${REF_N},${MIN_EVENTS},\"${FIGS_DIR}\")"
    run_root_macro "${MACRO}"
}

# ---- main menu ----------------------------------------------------------

show_menu() {
    echo ""
    echo "================================================================"
    echo "  Dijet entropy/MI analysis -- pick a script to run"
    echo "================================================================"
    echo "   1) analyze_dijet_entropy.py     (main analysis, Python)"
    echo "   2) analyze_dijet_entropy.C      (main analysis, ROOT)"
    echo "   3) closure_test.py              (whole-sample closure test, Python)"
    echo "   4) closure_test.C               (whole-sample closure test, ROOT)"
    echo "   5) closure_test_pt.py           (pT-differential closure test, Python)"
    echo "   6) closure_test_pt.C            (pT-differential closure test, ROOT)"
    echo "   0) Exit"
    echo "================================================================"
    echo "  Tip: './run_analysis.sh <N> yourfile.root' skips all prompts and"
    echo "  runs with every default -- add a trailing -y to also skip the"
    echo "  final confirmation."
}

run_choice() {
    case "$1" in
        1) run_analyze_py ;;
        2) run_analyze_root ;;
        3) run_closure_py ;;
        4) run_closure_root ;;
        5) run_closure_pt_py ;;
        6) run_closure_pt_root ;;
        0) exit 0 ;;
        *) echo "  Not a valid option." ;;
    esac
}

# ---- entry point ---------------------------------------------------------
#
#   ./run_analysis.sh                      interactive menu
#   ./run_analysis.sh N                    interactive prompts for choice N
#   ./run_analysis.sh N rootfile.root      QUICK MODE: choice N, that file, every
#                                          other argument at its default, no prompts
#   ./run_analysis.sh N rootfile.root -y   same, and also auto-confirm (no final y/n)

if [ "$#" -ge 2 ]; then
    QUICK=true
    AUTO_YES=false
    ROOT_FILE="$2"
    if [ "$#" -ge 3 ] && [ "$3" = "-y" ]; then
        AUTO_YES=true
    fi
    run_choice "$1"
    exit 0
elif [ "$#" -eq 1 ]; then
    QUICK=false
    run_choice "$1"
    exit 0
fi

QUICK=false
while true; do
    show_menu
    read -r -p "Choice: " choice
    run_choice "${choice}"
    [ "${choice}" != "0" ] && pause
done
