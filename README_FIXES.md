# Reviewer-Response Fixes — What's In This Folder

This package addresses the five reviewer points on your seizure-control paper.
I could **not run any of it here** — your `hopf_model.py`, `lqr_control.py`,
`metrics.py`, `viz.py`, `_common.py`, `baselines.py` and the
`data/brain_connectivity.mat` connectome weren't uploaded — so everything is
written against the exact interfaces your uploaded files already use, and **you
run it in your repo.**

## Where the files go

Drop these into your existing project, preserving the layout:

```
your_project/
├─ src/
│   ├─ config.py          ← REPLACE  (adds U_MAX + Monte-Carlo settings)
│   ├─ dpo_control.py     ← REPLACE  (adds u_max actuator saturation)
│   └─ closed_loop.py     ← REPLACE  (adds plant_noise + dpo_u_max passthrough)
└─ scripts/
    ├─ monte_carlo_validation.py     ← NEW
    ├─ analysis_umax_sweep.py        ← NEW
    ├─ analysis_duty_cycle_energy.py ← NEW
    └─ analysis_scale_timing.py      ← NEW
```

The three `src/` files are **drop-in replacements**: with the new options left
at their defaults they behave **identically** to your originals, so nothing in
your existing pipeline changes until you opt in.

## Two interface assumptions I relied on (please sanity-check)

1. `net.rk4_step(s, dt, u=u, noise=<float>, rng=<Generator>)` accepts `noise`
   and `rng` keywords. Your `week05_detector_validation.py` already calls
   `rk4_step(..., noise=C.NOISE_BETA, rng=rng)`, so this should match. The
   closed loop only uses that form when `plant_noise > 0`; otherwise it calls
   `net.rk4_step(s, dt, u=u)` exactly as before.
2. `_common` (`cm`) exposes `load_network()`, `onset_zone()`, `seizing_network()`,
   `savetxt_table()`, and `src.metrics.amplitude()` exists — all used exactly as
   in your `week08_closed_loop.py`.

If either differs, it's a one-line tweak — tell me and I'll adjust.

---

## What each reviewer point maps to

| Reviewer point | File(s) | Output | Replaces / new |
|---|---|---|---|
| **#1 — N=1 seed, no CIs** | `scripts/monte_carlo_validation.py` | `results/monte_carlo_summary.csv`, `results/monte_carlo_per_trial.csv`, `images/monte_carlo_success_energy.png` | **Replaces the headline numbers**: the "90% (9/10)" success and single-value energy `0.876` in §4.6 / Table 3, and the "1 seed" statements in §3.10 / §5.4. Now: success % with 95% CI + energy mean±CI. The PNG is a **new** figure. |
| **#3 — unconstrained control** | `src/dpo_control.py`, `src/closed_loop.py`, `src/config.py`, `scripts/analysis_umax_sweep.py` | `results/umax_sweep.csv`, `images/umax_sweep.png` | The sweep PNG is **new**. Once you set `config.U_MAX`, re-running week07/week08 makes **Fig 9 (DPO suppression)** and **Fig 10 (closed-loop)** the *saturated* versions — i.e. it updates those existing figures. |
| **#4 — FPR vs energy** | `scripts/analysis_duty_cycle_energy.py` | `images/duty_cycle_energy.png` (+ printed crossover) | **New addition.** Supports a new caveat sentence in §4.7 / §5.3 (your energy claim is per-intervention, not per-day; here's the daily crossover). |
| **#5 — 60 vs 360 scaling** | `scripts/analysis_scale_timing.py` | `results/scale_timing.csv` (+ printed wall-clock) | **New addition** (no figure). Gives concrete timings to quote for the "is 360-node DPO feasible?" objection. |
| **#2 — CSD / Wilkat contradiction** | *(not code — paper text)* | — | See note at the bottom; it's a framing sentence, not a code change. |

---

## Run order (recommended)

Start small to estimate wall-clock, then scale up. `SCALE=reduced` = your
60-node sub-network; omit it (or `SCALE=full`) for the 360-node design target.

1. **Choose the actuator bound (#3):**
   ```
   SCALE=reduced python scripts/analysis_umax_sweep.py
   ```
   Read `results/umax_sweep.csv` / `images/umax_sweep.png`, pick the smallest
   `u_max` that still suppresses with margin, and set `U_MAX = <that value>` in
   `src/config.py`. (Leaving it `None` = unconstrained, i.e. the unfixed state.)

2. **Re-generate the saturated DPO + closed-loop figures (#3):**
   re-run your `scripts/week07_*.py` and `scripts/week08_closed_loop.py` now that
   `U_MAX` is set — Fig 9 and Fig 10 become the clinically-bounded versions.

3. **Monte-Carlo validation (#1) — the important one:**
   ```
   SCALE=reduced N_SEEDS=20 python scripts/monte_carlo_validation.py   # quick test
   SCALE=reduced            python scripts/monte_carlo_validation.py   # full 100
   ```
   Put the new success%+CI and energy mean±CI into Table 3 / §4.6, and update
   the §3.10 and §5.4 "single seed" sentences to say multi-seed Monte Carlo.

4. **Duty-cycle energy (#4):**
   ```
   python scripts/analysis_duty_cycle_energy.py
   ```
   (Defaults read your `week08` summary; override numbers with `--e-int`,
   `--n-seizures`, `--n-control`, `--openloop-amp` and justify them in the text.)

5. **Scaling timings (#5):**
   ```
   SCALE=reduced python scripts/analysis_scale_timing.py
   SCALE=full    python scripts/analysis_scale_timing.py
   ```

## After you have the numbers

Send me the new values (success%+CI, energy mean±CI, chosen `U_MAX`, crossover,
timings) and I'll drop them into the `.tex` precisely — without changing any of
your wording — and update Table 3, §3.10, §4.6, and §5.4 to match.

## #2 — the one that isn't code

The CSD/Wilkat point is a framing fix, not a simulation. Your strongest, honest
position is that in your model the variance / lag-1-autocorrelation rise is a
*mathematical consequence* of approaching the Hopf bifurcation (your
Fokker-Planck analysis), not an empirical human claim — and the Wilkat citation
is there precisely as an acknowledged limitation. One sentence in §4.3 makes
that explicit; I gave you the exact wording last turn and can paste it on your
go-ahead.
