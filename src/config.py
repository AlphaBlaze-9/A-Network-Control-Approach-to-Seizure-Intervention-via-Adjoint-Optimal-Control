"""
config.py
=========
Central configuration for the seizure-control project. Every script imports
from here so that paths, model constants, and the (honest) scale strategy live
in exactly one place.

Why a single config file:
    The Week-1 deliverable demands that every figure be labelled with the scale
    it was actually run at and that a reduced validation scale never be passed
    off as a 360-node result. Centralising ``ACTIVE_SCALE`` here makes that rule
    mechanical: each figure title is stamped with ``scale_label()`` automatically.
"""

from __future__ import annotations
import os

# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
IMAGE_DIR   = os.path.join(PROJECT_DIR, "images")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

MAT_PATH    = os.path.join(DATA_DIR, "brain_connectivity.mat")   # raw HCP-MMP connectome
A_MATRIX_NPY = os.path.join(DATA_DIR, "brain_A_matrix.npy")      # cleaned, normalised A

for _d in (DATA_DIR, IMAGE_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Atlas / network
# --------------------------------------------------------------------------- #
N_NODES_FULL = 360            # Glasser HCP-MMP1 cortical parcellation (design target)
ATLAS_NAME   = "HCP-MMP (Glasser 360)"

# Seizure-onset focus: L_H = left hippocampus (index 119), the canonical
# mesial-temporal-lobe-epilepsy onset zone. Its mesial-temporal neighbourhood
# (entorhinal cortex, presubiculum) is listed for interpretation.
FOCUS_NODE        = 119                       # L_H
FOCUS_NAME        = "L_H (left hippocampus)"
MESIAL_TEMPORAL   = {117: "L_EC", 118: "L_PreS", 119: "L_H", 121: "L_PeEc"}

# --------------------------------------------------------------------------- #
#  Scale strategy  (Week-1 [FIX])
# --------------------------------------------------------------------------- #
# "full"    -> all 360 Glasser nodes (design target; used for cheap weeks and
#              the documented scale-up checkpoint).
# "reduced" -> a dense, seizure-relevant sub-network selected around the focus,
#              used for fast development of the heavy optimal-control weeks.
ACTIVE_SCALE   = os.environ.get("SCALE", "full")     # override with  SCALE=reduced
REDUCED_SIZE   = 60          # nodes in the reduced validation sub-network


def scale_label() -> str:
    """Human-readable scale tag stamped onto every figure title."""
    if ACTIVE_SCALE == "full":
        return f"scale=FULL {N_NODES_FULL}-node Glasser"
    return f"scale=REDUCED {REDUCED_SIZE}-node sub-network"


# --------------------------------------------------------------------------- #
#  Coupled-Hopf model constants  (Deco et al., 2017 whole-brain Hopf model)
# --------------------------------------------------------------------------- #
G_COUPLING   = 2.5           # global coupling factor G (tuned: visible connectome-mediated
                             #   spread to limbic neighbours, while all-healthy net stays stable)
A_REST       = -0.02         # healthy bifurcation parameter (slightly < 0: stable focus,
                             #   near criticality as in resting whole-brain Hopf models)
A_SEIZURE    =  0.6          # seizure-prone bifurcation parameter (> 0: limit cycle)
OMEGA_HZ     = 0.10          # intrinsic frequency f (Hz); omega = 2*pi*f
NOISE_BETA   = 0.02          # additive noise amplitude (set 0.0 for deterministic runs)

# Integration
DT           = 0.05          # RK4 step (s)
SEED         = 7             # master RNG seed for reproducibility

# Seizure-state definition (Week 3): a node is "ictal" when its oscillation
# amplitude envelope exceeds this radius for a sustained window.
SEIZURE_AMP_THRESHOLD = 0.30

# --------------------------------------------------------------------------- #
#  Reviewer-response additions  (statistical rigor + actuator saturation)
# --------------------------------------------------------------------------- #
# [Reviewer #3 -- unconstrained control] Actuator saturation bound applied to the
# Level-2 (DPO) control schedule. ``None`` == unconstrained (the original
# behaviour, so nothing in your existing pipeline changes until you set this).
#
# HOW TO CHOOSE IT: run  scripts/analysis_umax_sweep.py  -- it sweeps a range of
# bounds and reports, for each, whether suppression still holds and at what
# energy. Pick the SMALLEST bound that still suppresses with a little margin,
# set it here, then regenerate the DPO + closed-loop figures so the reported
# result is the saturated (clinically meaningful) one.
U_MAX = 0.5                 # e.g. 0.5 once justified by the sweep

# [Reviewer #1 -- N=1 seed] Monte-Carlo statistical validation settings.
# Replaces single-seed point values with a success proportion + 95% CI and
# energy mean +/- 95% CI (scripts/monte_carlo_validation.py).
N_SEEDS  = 100               # number of independent trials (randomised init + noise)
MC_SEED0 = 1000              # trial i uses seed = MC_SEED0 + i

# --------------------------------------------------------------------------- #
#  Week-10 literature baseline reference (chosen now, per Week-1 [FIX])
# --------------------------------------------------------------------------- #
LITERATURE_BASELINE = (
    "Wang, Niebur, Hu & Li (2016), 'Suppressing epileptic activity in a neural "
    "mass model using a closed-loop proportional-integral controller', "
    "Scientific Reports 6:27344. Implemented in src/baselines.py as a PI "
    "controller acting on the measured ictal amplitude."
)

# =========================================================================== #
#  REVIEWER-RESPONSE ANALYSIS SETTINGS  (second pass: issues #5,#6,#7,#20-24)
#  Used only by the new scripts/analysis_*.py; nothing above changes.
# =========================================================================== #

# ---- #5  Step-size / strong-convergence study (analysis_step_size.py) ------ #
# All coarse steps must be integer multiples of the finest reference step so
# that coarse Brownian increments are exact sums of fine ones (shared path).
DT_CONV_FINEST = 0.00625                 # reference (finest) step, s
DT_CONV_LEVELS = [0.0125, 0.025, 0.05, 0.10, 0.20]   # tested steps (incl. production DT)
DT_CONV_T      = 6.0                     # integration horizon, s
DT_CONV_PATHS  = 64                      # Brownian paths averaged for the strong error

# ---- #6/#24  Detector validation set + threshold leakage ------------------- #
# Validation fold is used ONLY to pick the operating threshold; the test fold
# (independent seeds) is used ONLY to report TPR/FPR/precision/recall + AUC CI.
DET_VAL_PER_CLASS  = 50                  # was 12: validation-fold trajectories per class
DET_TEST_PER_CLASS = 50                  # independent test-fold trajectories per class
DET_BOOTSTRAP      = 2000                # bootstrap resamples for the AUC 95% CI
DET_PERSIST        = 3                   # consecutive points required to declare an alarm
# Distinct seed banks so the three sets never overlap:
DET_CAL_SEED0  = 9000                    # healthy-baseline calibration signals
DET_VAL_SEED0  = 1000                    # validation positives/negatives (+0.../+5000)
DET_TEST_SEED0 = 30000                   # test positives/negatives     (+0.../+5000)

# ---- #7  Out-of-distribution grid (analysis_ood_test.py) ------------------- #
# One-axis-at-a-time sweeps around the in-distribution defaults (G=2.5,
# omega=0.10 Hz, a_seizure=0.6). Success = onset zone held below threshold.
OOD_N_SEEDS    = 20
OOD_G_GRID     = [1.5, 2.0, 2.5, 3.0, 3.5]
OOD_OMEGA_GRID = [0.06, 0.08, 0.10, 0.12, 0.14]
OOD_ASEIZ_GRID = [0.30, 0.45, 0.60, 0.75, 0.90]

# ---- #21  Open-loop DBS amplitude sweep (analysis_dbs_amplitude_sweep.py) -- #
DBS_AMP_GRID   = [0.03, 0.06, 0.09, 0.12, 0.15]

# ---- #20  Refractory-period probe (analysis_refractory.py) ----------------- #
DPO_HORIZON    = 200                     # Level-2 open-loop horizon (steps)
REFRACTORY     = 20                      # refractory after a schedule ends (steps)

# ---- #22  FPR-vs-energy daily accounting (analysis_fpr_energy.py) ---------- #
# STATE THESE ASSUMPTIONS EXPLICITLY IN THE PAPER -- they are not measured here.
DETECTOR_FPR        = None               # None -> read the TEST-fold FPR from results/week05_detector_testfold.csv
ASSUMED_SEIZURES_DAY = 8                 # assumed genuine clinical events per day (illustrative)
DETECTOR_RECORD_S    = 120.0             # length of one detector evaluation record (2400 steps x 0.05 s)
DETECTOR_EVALS_DAY   = int(24 * 3600 / DETECTOR_RECORD_S)   # 720 independent 120-s records per day
