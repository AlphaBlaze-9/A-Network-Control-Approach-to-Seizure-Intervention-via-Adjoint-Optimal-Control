"""
closed_loop.py  --  Phase 3 / Week 8: Closed-Loop System Integration
====================================================================
Ties the Week-4/5 detector and the two-level corrector (Week-6 LQR + Week-7
DPO) into one framework and runs it on the seizing connectome:

  * Level-1 (LQR) runs continuously at a gentle gain -- local anti-drift.
  * The detector watches a monitored scalar (network mean amplitude) over a
    rolling window and emits a continuous risk score.
  * When risk crosses a threshold, Level-2 (DPO) is triggered: a short-horizon
    optimal control schedule is solved from the current state and replayed
    (receding-horizon / MPC style) while Level-1 keeps running.
  * Robustness: detector noise/bias can be injected to test that the controller
    still drives the system off the unsafe set even when the trigger is wrong
    (directly probing the "biased evaluation pushes you into the unsafe state"
    warning).

The plant itself is always integrated with the high-accuracy RK4 step; only the
DPO *design model* uses Euler internally (see dpo_control.py).

[Reviewer #1 -- statistics] ``plant_noise`` injects a stochastic Wiener
increment into the plant integration so that repeated runs with different seeds
explore genuinely different noise realisations (this is what makes the
Monte-Carlo validation meaningful). Default 0.0 reproduces the original
deterministic plant exactly.

[Reviewer #3 -- actuator saturation] ``dpo_u_max`` is forwarded to the DPO
optimiser to bound the Level-2 control schedule. Default None == unconstrained.
"""

from __future__ import annotations
import numpy as np

from . import config
from .hopf_model import HopfNetwork
from .lqr_control import LocalLQR
from .dpo_control import DPOController
from .detector import CSDDetector
from .metrics import amplitude


def run_closed_loop(net: HopfNetwork, state0, n_steps, dt=config.DT,
                    monitor_nodes=None, detector=None,
                    risk_threshold=0.55, dpo_horizon=200,
                    dpo_iters=120, dpo_rho=0.02, dpo_q=1.0,
                    l1_gain=0.15, control_nodes=None,
                    detector_noise=0.0, detector_bias=0.0,
                    refractory=150, decision_stride=20,
                    plant_noise=0.0, dpo_u_max=None,
                    seed=config.SEED, verbose=False, real_only=False):
    """Integrate the full closed-loop system and log everything.

    Returns a dict with the trajectory, applied control, per-step risk, and the
    list of intervention onset steps -- everything the Week-8 figures need.

    ``decision_stride`` is how often (in steps) the CSD detector is actually
    polled; between polls the last risk value is held. A real early-warning
    detector runs on a fixed monitoring cadence rather than re-deriving its
    windowed statistics every integration step, and polling periodically keeps
    the (relatively expensive) rolling-feature computation tractable.

    ``plant_noise`` (float): amplitude of the additive Wiener increment applied
    to the plant during integration. 0.0 (default) == deterministic plant, i.e.
    identical to the original behaviour. Set to config.NOISE_BETA for the
    Monte-Carlo trials so each seed is a distinct noise realisation.

    ``dpo_u_max`` (float or None): actuator-saturation bound forwarded to the
    Level-2 DPO optimiser. None (default) == unconstrained control schedule.

    ``real_only`` (bool): restrict BOTH control levels to the real component
    of z (single-electrode actuation, B = [1, 0]^T per node). Default False ==
    original two-component actuation.
    """
    rng = np.random.default_rng(seed)
    N = net.N
    monitor_nodes = (np.arange(N) if monitor_nodes is None
                     else np.asarray(monitor_nodes))
    detector = detector or CSDDetector(window=150)

    # Level-1 controller (always on) built on the *resting* linearisation.
    lqr = LocalLQR(a=np.full(N, config.A_REST), omega=net.omega,
                   real_only=real_only)
    # Level-2 controller (triggered).
    dpo = DPOController(net, q=dpo_q, rho=dpo_rho, control_nodes=control_nodes,
                        real_only=real_only)

    traj = np.empty((n_steps + 1, 2 * N))
    traj[0] = state0
    ctrl = np.zeros((n_steps, 2 * N))
    risk_log = np.full(n_steps, np.nan)
    onsets = []

    s = state0.copy()
    monitor_buf = []                      # rolling buffer of monitored signal
    active_schedule = None                # current DPO schedule (or None)
    sched_start = 0
    cooldown = 0
    last_risk = 0.0                       # held between detector polls

    for t in range(n_steps):
        # --- monitored signal: mean oscillation amplitude over monitor set ---
        amp_now = np.sqrt(s[monitor_nodes] ** 2 + s[N + monitor_nodes] ** 2).mean()
        monitor_buf.append(amp_now)

        # --- detector risk (with optional injected error) ---
        # The CSD rolling features need MORE than ``window`` samples to produce
        # a usable series to normalise over (a single windowed value min-maxes
        # to 0). We evaluate on a trailing buffer of a few windows, and only on
        # the polling cadence ``decision_stride`` (holding the last value in
        # between) so the rolling-feature cost stays tractable.
        eval_len = 3 * detector.window
        if t % decision_stride == 0 and len(monitor_buf) >= eval_len:
            r, _ = detector.risk(np.asarray(monitor_buf[-eval_len:]))
            last_risk = r[-1] if np.isfinite(r[-1]) else last_risk
        risk = last_risk
        risk_obs = risk + detector_bias + detector_noise * rng.standard_normal()
        risk_log[t] = risk_obs

        # --- trigger Level-2 on threshold crossing (respecting refractory) ---
        if (active_schedule is None and cooldown <= 0
                and risk_obs >= risk_threshold):
            U, _ = dpo.optimize(s, horizon=dpo_horizon, dt=dt,
                                n_iters=dpo_iters, lr=0.05,
                                u_max=dpo_u_max)
            active_schedule = U
            sched_start = t
            onsets.append(t)
            if verbose:
                print(f"   [t={t}] risk={risk_obs:.2f} -> Level-2 intervention")

        # --- assemble control: Level-1 always, Level-2 if active ---
        u = l1_gain * lqr.control(s)
        if active_schedule is not None:
            k = t - sched_start
            if k < active_schedule.shape[0]:
                u = u + active_schedule[k]
            else:
                active_schedule = None
                cooldown = refractory
        cooldown -= 1
        ctrl[t] = u

        # --- advance the plant with RK4 (optionally with process noise) ---
        if plant_noise and plant_noise > 0.0:
            s = net.rk4_step(s, dt, u=u, noise=plant_noise, rng=rng)
        else:
            s = net.rk4_step(s, dt, u=u)
        traj[t + 1] = s

    return {
        "traj": traj,
        "ctrl": ctrl,
        "risk": risk_log,
        "onsets": onsets,
        "energy": float(dt * np.sum(ctrl ** 2)),
    }
