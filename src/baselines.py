"""
baselines.py  --  Phase 4 / Weeks 9-10: Baselines & Benchmarks
==============================================================
Three honest baselines the Hamiltonian-backed DPO controller is measured
against:

  Week 9
    * Ablation baseline   -- run control on a single ISOLATED oscillator vs the
                             full connectome mapping, isolating whether the
                             connectome adds value.
    * Clinical baseline   -- an always-on open-loop stimulation protocol
                             (mimicking conventional, non-adaptive DBS); compare
                             total control energy ||u||^2 against targeted DPO.
  Week 10
    * Literature baseline -- a closed-loop proportional-integral (PI) controller
                             on the measured ictal amplitude, following
                             Wang et al. (2016), Sci. Rep. 6:27344. Compare
                             success rate, false-activation rate, and off-target
                             stimulation against DPO.
"""

from __future__ import annotations
import numpy as np

from . import config
from .hopf_model import HopfNetwork
from .metrics import amplitude


# --------------------------------------------------------------------------- #
#  Clinical baseline: always-on open-loop stimulation
# --------------------------------------------------------------------------- #
def always_on_open_loop(net, state0, n_steps, stim_amp, stim_nodes,
                        dt=config.DT, seed=config.SEED):
    """Constant, non-adaptive stimulation applied to a fixed set of nodes.

    Models conventional open-loop DBS: a fixed-amplitude drive is delivered
    continuously regardless of state. Returns (trajectory, total_energy).
    The control opposes the local oscillation (-stim_amp * unit state) to give
    the open-loop protocol its best honest chance at suppression.
    """
    N = net.N
    mask = np.zeros(N)
    mask[np.asarray(stim_nodes)] = 1.0

    def u_func(t, s):
        x, y = s[:N], s[N:]
        r = np.sqrt(x * x + y * y) + 1e-9
        # push each stimulated node radially inward at constant magnitude
        return -stim_amp * np.concatenate([mask * x / r, mask * y / r])

    traj, ctrl = net.simulate(state0, n_steps, dt=dt, u_func=u_func,
                              record_control=True, seed=seed)
    return traj, float(dt * np.sum(ctrl ** 2))


# --------------------------------------------------------------------------- #
#  Literature baseline: closed-loop PI controller (Wang et al., 2016)
# --------------------------------------------------------------------------- #
class PIController:
    """Closed-loop proportional-integral controller on ictal amplitude.

    Follows the scheme of Wang et al. (2016): a biomarker (here the mean
    oscillation amplitude over the controlled nodes, relative to a healthy set
    point) drives a PI law whose output stimulates the network. PI is the most
    widely used closed-loop control scheme, which makes it a fair, recognisable
    literature standard.
    """

    def __init__(self, kp=2.0, ki=0.5, setpoint=0.05, stim_nodes=None, N=None):
        self.kp, self.ki, self.setpoint = kp, ki, setpoint
        self.integral = 0.0
        self.N = N
        self.mask = np.ones(N) if stim_nodes is None else np.zeros(N)
        if stim_nodes is not None:
            self.mask[np.asarray(stim_nodes)] = 1.0

    def u_func(self, dt):
        """Return a u(t, state) callable implementing the PI feedback law."""
        def _u(t, s):
            N = self.N
            x, y = s[:N], s[N:]
            r = np.sqrt(x * x + y * y)
            err = max(0.0, r[self.mask > 0].mean() - self.setpoint)  # excess amplitude
            self.integral += err * dt
            gain = self.kp * err + self.ki * self.integral
            rr = r + 1e-9
            return -gain * np.concatenate([self.mask * x / rr, self.mask * y / rr])
        return _u


def run_pi_baseline(net, state0, n_steps, stim_nodes, dt=config.DT,
                    kp=2.0, ki=0.5, seed=config.SEED):
    """Integrate the PI literature baseline; return (trajectory, energy)."""
    pi = PIController(kp=kp, ki=ki, stim_nodes=stim_nodes, N=net.N)
    traj, ctrl = net.simulate(state0, n_steps, dt=dt, u_func=pi.u_func(dt),
                              record_control=True, seed=seed)
    return traj, float(dt * np.sum(ctrl ** 2))


# --------------------------------------------------------------------------- #
#  Benchmark metrics shared across baselines
# --------------------------------------------------------------------------- #
def off_target_fraction(ctrl, N, focus_nodes):
    """Fraction of total control energy delivered OUTSIDE the focus set.

    A precision metric: a targeted controller should spend its energy on the
    seizure focus, not the whole brain.
    """
    e = ctrl[:, :N] ** 2 + ctrl[:, N:] ** 2          # per-node energy over time
    total = e.sum()
    if total <= 0:
        return 0.0
    off = np.delete(e, np.asarray(focus_nodes), axis=1).sum()
    return float(off / total)


def seizure_aborted(traj, N, threshold=config.SEIZURE_AMP_THRESHOLD, tail=200):
    """True if the focus/network amplitude is driven below threshold by the end."""
    amp = amplitude(traj, N)[-tail:].mean(axis=0)
    return bool(amp.max() < threshold)
