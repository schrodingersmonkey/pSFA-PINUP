"""
Feature representations fed to SFA.

Ports the seven builders from `correlation_testing.ipynb` (PINUP repo) so the
VAR simulations can be swept in exactly the feature spaces that were compared on
the Neuropixels data. Definitions are kept verbatim where possible; the only
substantive adaptation is `node-delta`, which needs a band definition that makes
sense for a simulation with no physical sampling rate (see `sw_band_power`).

Every builder returns an array of the same length, T - W + 1, where W is the
(odd-forced) edge window, so representations are directly comparable and share a
common aligned driver.

One identity worth knowing before reading any result:

  * `edges-corr-glob` is `edges-cov-win` times a per-edge constant. SFA whitens
    its input, which absorbs any diagonal rescaling, so the two SFA curves are
    identical to numerical precision. This is a *check*, not a redundancy: it
    isolates the claim that only the LOCAL, time-varying normalisation in
    `edges-corr` actually removes shared gain.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from src.util import moving_average_centered


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ma(Y: np.ndarray, W: int) -> np.ndarray:
    """Centred moving average accepting 1-D or 2-D input; forces odd W."""
    W = max(1, int(W))
    if W % 2 == 0:
        W += 1
    Y = np.asarray(Y, dtype=float)
    if W == 1:
        return Y
    if Y.ndim == 1:
        return moving_average_centered(Y[:, None], W)[:, 0]
    return moving_average_centered(Y, W)


def _pairs(N: int):
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


# ---------------------------------------------------------------------------
# edge features
# ---------------------------------------------------------------------------

def edges_cov_global(X: np.ndarray, win: int) -> np.ndarray:
    """Covariance edges, GLOBAL mean centred — the manuscript's p-SFA recipe.

    smooth( x_i x_j ) - global_mean( x_i x_j ).
    Differs from a true windowed covariance by the time-varying leak term
    local_mean(x_i)(t) * local_mean(x_j)(t), which this construction leaves in.
    """
    T, N = X.shape
    prs = _pairs(N)
    E = np.empty((T, len(prs)))
    for k, (i, j) in enumerate(prs):
        E[:, k] = X[:, i] * X[:, j]
    return _ma(E - E.mean(axis=0, keepdims=True), win)


def edges_cov_windowed(X: np.ndarray, win: int) -> np.ndarray:
    """TRUE windowed covariance: <x_i x_j>_W - <x_i>_W <x_j>_W."""
    T, N = X.shape
    prs = _pairs(N)
    m1 = _ma(X, win)
    Cov = np.empty((m1.shape[0], len(prs)))
    for k, (i, j) in enumerate(prs):
        Cov[:, k] = _ma(X[:, i] * X[:, j], win) - m1[:, i] * m1[:, j]
    return Cov


def edges_corr_global(X: np.ndarray, win: int) -> np.ndarray:
    """Windowed covariance / GLOBAL per-node std — a per-edge CONSTANT rescale.

    Included deliberately: SFA should match `edges_cov_windowed` exactly, which
    demonstrates that a time-invariant normalisation removes nothing.
    """
    T, N = X.shape
    prs = _pairs(N)
    gstd = X.std(axis=0)
    m1 = _ma(X, win)
    R = np.empty((m1.shape[0], len(prs)))
    for k, (i, j) in enumerate(prs):
        cov = _ma(X[:, i] * X[:, j], win) - m1[:, i] * m1[:, j]
        R[:, k] = cov / (gstd[i] * gstd[j])
    return R


def edges_corr_local(X: np.ndarray, win: int) -> np.ndarray:
    """Windowed Pearson correlation: Cov_ij(t) / sqrt(Var_i(t) Var_j(t)).

    Under a shared multiplicative gain x_i(t) = g(t) s_i(t), the g(t)^2 cancels
    exactly, so any surviving time variation is gain-independent by construction.

    No epsilon floor: `Var` is only clipped at zero to prevent an illegal sqrt of
    a tiny negative rounding artefact. A genuinely zero-variance node surfaces as
    a visible NaN rather than a silently plausible number.
    """
    T, N = X.shape
    prs = _pairs(N)
    m1 = _ma(X, win)
    Var = np.maximum(_ma(X ** 2, win) - m1 ** 2, 0.0)
    R = np.empty((m1.shape[0], len(prs)))
    for k, (i, j) in enumerate(prs):
        cov = _ma(X[:, i] * X[:, j], win) - m1[:, i] * m1[:, j]
        R[:, k] = cov / np.sqrt(Var[:, i] * Var[:, j])
    return R


# ---------------------------------------------------------------------------
# node features
# ---------------------------------------------------------------------------

def nodes_raw(X: np.ndarray, win: int) -> np.ndarray:
    """Windowed raw node amplitudes — the node-SFA baseline."""
    return _ma(X, win)


def sw_variance(X: np.ndarray, win: int) -> np.ndarray:
    """Per-node sliding-window variance (the amplitude/power envelope)."""
    m1 = _ma(X, win)
    m2 = _ma(X ** 2, win)
    return m2 - m1 ** 2


# 1-4 Hz at the Neuropixels analysis rate of 125 Hz, expressed as a fraction of
# Nyquist. Reused here so the simulated "delta" band occupies the same relative
# position in the spectrum as it did in the LFP work.
DELTA_BAND_FRAC = (1.0 / 62.5, 4.0 / 62.5)


def sw_band_power(X: np.ndarray, win: int,
                  band_frac: tuple[float, float] = DELTA_BAND_FRAC,
                  order: int = 4) -> np.ndarray:
    """Per-node sliding-window band power, band given as a fraction of Nyquist.

    A VAR simulation has no physical sampling rate, so the LFP recipe's literal
    1-4 Hz is meaningless here. `band_frac` preserves the *relative* spectral
    position instead: the default is 1-4 Hz at 125 Hz as a fraction of Nyquist,
    which places the band well above the driver frequency (1/P per sample),
    exactly as delta sits well above the pupil fluctuation rate in the real data.
    """
    lo, hi = band_frac
    if not (0 < lo < hi < 1):
        raise ValueError(f"band_frac must satisfy 0 < lo < hi < 1, got {band_frac}")
    b, a = butter(order, [lo, hi], btype="band")
    P = np.empty_like(X, dtype=float)
    for i in range(X.shape[1]):
        P[:, i] = np.abs(hilbert(filtfilt(b, a, X[:, i]))) ** 2
    return _ma(P, win)



FEATURES = {
    "edges-cov":       edges_cov_global,
    "edges-cov-win":   edges_cov_windowed,
    "edges-corr-glob": edges_corr_global,
    "edges-corr":      edges_corr_local,
    "nodes":           nodes_raw,
    "node-var":        sw_variance,
    "node-delta":      sw_band_power,
}

ALL_FEATURES = [
    "edges-cov", "edges-cov-win", "edges-corr-glob", "edges-corr",
    "nodes", "node-var", "node-delta",
]


# ---------------------------------------------------------------------------
# composite (concatenated) features
# ---------------------------------------------------------------------------
# Correlation combines coupling and amplitude information by *dividing* one by
# the other -- a fixed, nonlinear, structural operation. The alternative is to
# hand SFA both blocks side by side and let it find whatever linear combination
# is slowest. Whitening equalises the per-feature scale, so the two blocks do
# not need manual balancing, though the edge block contributes N(N-1)/2 columns
# against the node block's N and therefore dominates the count as N grows.
#
# The two approaches differ in an important way: division removes shared gain
# *exactly*, at the cost of also removing genuine driver-related variance
# structure. Concatenation removes nothing and discards nothing -- it can only
# cancel gain to the extent that a linear projection can.

def _concat(*builders):
    def build(X, win):
        blocks = [b(X, win) for b in builders]
        n = min(b.shape[0] for b in blocks)
        return np.hstack([b[:n] for b in blocks])
    return build


FEATURES["cov+var"]  = _concat(edges_cov_global, sw_variance)
FEATURES["corr+var"] = _concat(edges_corr_local, sw_variance)
FEATURES["cov+corr"] = _concat(edges_cov_global, edges_corr_local)

COMPOSITE_FEATURES = ["cov+var", "corr+var", "cov+corr"]

# Display names and a consistent colour per feature, used across every figure.
FEATURE_LABELS = {
    "edges-cov":       "covariance, global-mean centred",
    "edges-cov-win":   "covariance, local-mean centred",
    "edges-corr-glob": "correlation, global std",
    "edges-corr":      "correlation, local std",
    "nodes":           "node amplitude (raw)",
    "node-var":        "node variance (window)",
    "node-delta":      "node band power",
    "cov+var":         "covariance $\\oplus$ node variance",
    "corr+var":        "correlation $\\oplus$ node variance",
    "cov+corr":        "covariance $\\oplus$ correlation",
}

FEATURE_COLORS = {
    "edges-cov":       "#2a6f72",
    "edges-cov-win":   "#7bb6b8",
    "edges-corr-glob": "#a8d0d1",
    "edges-corr":      "#c58a3d",
    "nodes":           "#b8bcc4",
    "node-var":        "#8a8f99",
    "node-delta":      "#5a5f67",
    "cov+var":         "#7d3c98",
    "corr+var":        "#b07cc6",
    "cov+corr":        "#4a235a",
}


def build_feature(name: str, X: np.ndarray, win: int) -> np.ndarray:
    if name not in FEATURES:
        raise KeyError(f"Unknown feature '{name}'. Options: {ALL_FEATURES}")
    return FEATURES[name](X, win)
