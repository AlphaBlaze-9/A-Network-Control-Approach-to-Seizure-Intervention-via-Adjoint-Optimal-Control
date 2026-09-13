"""
detector.py  --  Phase 2 / Weeks 4-5: Critical-Slowing-Down Detector
====================================================================
An *independent* early-warning detector that ingests a scalar (or low-D)
time-series and outputs a continuous bifurcation-risk score, using the three
classic critical-slowing-down (CSD) signatures that grow as a dynamical system
approaches a bifurcation:

  1. Rolling variance        -- fluctuations grow as the restoring force weakens.
  2. Lag-1 autocorrelation   -- the system "remembers" longer (AC1 -> 1).
  3. Short-horizon divergence (a rolling largest-Lyapunov-exponent estimate via
     Rosenstein's nearest-neighbour method) -- trajectories separate faster.

The detector is deliberately model-free (it never sees a_i or the equations),
so that in Week 5 it can be validated *on its own* before it is ever allowed to
drive the controller -- the safeguard against a biased evaluator steering the
system into the unsafe state.
"""

from __future__ import annotations
import numpy as np


def rolling_variance(signal, window):
    """Causal rolling variance over ``window`` samples (NaN until filled)."""
    s = np.asarray(signal, dtype=float)
    out = np.full(s.shape, np.nan)
    for t in range(window, len(s) + 1):
        out[t - 1] = np.var(s[t - window:t])
    return out


def rolling_autocorr1(signal, window):
    """Causal rolling lag-1 autocorrelation (AC1) over ``window`` samples.

    AC1 rises toward 1 as critical slowing down sets in: the dominant
    eigenvalue of the local linearisation approaches the imaginary axis, so the
    decay of perturbations slows and successive samples become more correlated.
    """
    s = np.asarray(signal, dtype=float)
    out = np.full(s.shape, np.nan)
    for t in range(window, len(s) + 1):
        w = s[t - window:t]
        w = w - w.mean()
        denom = np.sum(w * w)
        out[t - 1] = (np.sum(w[1:] * w[:-1]) / denom) if denom > 0 else 0.0
    return out


def rolling_lyapunov(signal, window, emb_dim=3, tau=1, horizon=6):
    """Rolling largest-Lyapunov-exponent estimate (Rosenstein, 1993).

    Within each window the signal is delay-embedded; for each embedded point we
    find its nearest neighbour and measure how fast the pair diverges over a
    short horizon. The mean log-divergence rate approximates the largest
    Lyapunov exponent, which climbs as the trajectory becomes less contractive
    near a bifurcation. Returns NaN where a window is too short to embed.
    """
    s = np.asarray(signal, dtype=float)
    out = np.full(s.shape, np.nan)
    span = (emb_dim - 1) * tau
    for t in range(window, len(s) + 1):
        w = s[t - window:t]
        M = len(w) - span
        if M <= horizon + 2:
            continue
        emb = np.stack([w[i: i + M] for i in range(0, span + 1, tau)], axis=1)
        rates = []
        for i in range(M - horizon):
            # nearest neighbour excluding a small temporal window (Theiler win)
            d = np.linalg.norm(emb - emb[i], axis=1)
            d[max(0, i - 1): i + 2] = np.inf
            j = int(np.argmin(d))
            if j + horizon >= M or d[j] == 0 or not np.isfinite(d[j]):
                continue
            d0 = d[j]
            dh = np.linalg.norm(emb[i + horizon] - emb[j + horizon])
            if dh > 0:
                rates.append(np.log(dh / d0) / horizon)
        if rates:
            out[t - 1] = np.mean(rates)
    return out


def _minmax(a):
    """Normalise a vector to [0, 1] ignoring NaNs (flat -> zeros)."""
    a = np.asarray(a, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if hi - lo < 1e-12:
        return np.nan_to_num((a - lo))
    return np.clip((a - lo) / (hi - lo), 0, 1)


class CSDDetector:
    """Critical-slowing-down detector producing a continuous risk score.

    Parameters
    ----------
    window : int       -- samples per rolling statistic
    weights : tuple    -- (w_var, w_ac1, w_lyap) fusion weights
    """

    def __init__(self, window=200, weights=(0.4, 0.4, 0.2)):
        self.window = int(window)
        self.weights = np.asarray(weights, dtype=float)
        self.weights /= self.weights.sum()

    def features(self, signal):
        """Return the three raw CSD feature time-series for a signal."""
        return {
            "variance": rolling_variance(signal, self.window),
            "autocorr1": rolling_autocorr1(signal, self.window),
            "lyapunov": rolling_lyapunov(signal, self.window),
        }

    def risk(self, signal):
        """Fuse the three CSD features into one continuous risk score in [0, 1].

        Each feature is min-max normalised over the record and combined with the
        configured weights. The score is the quantity Week-5 thresholds, and the
        signal Week-8's closed loop watches to trigger the optimal controller.
        """
        f = self.features(signal)
        r = (self.weights[0] * _minmax(f["variance"])
             + self.weights[1] * _minmax(f["autocorr1"])
             + self.weights[2] * _minmax(f["lyapunov"]))
        return r, f
