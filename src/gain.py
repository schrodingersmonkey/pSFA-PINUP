"""
Shared multiplicative gain channel for the TVP-VAR.

The VAR in the manuscript has no gain: the driver enters only through the
coupling matrix, so the diagonal of the covariance barely moves and correlation
edges reduce to covariance edges times a per-edge constant (which SFA's
whitening absorbs exactly). That is why the two features tie in simulation while
correlation clearly wins on real data.

This module adds the missing mechanism:

    xtilde_t = (A0 + d_t C) xtilde_{t-1} + eps_t        (coupling channel)
    x_t      = (1 + gamma * d_t) * xtilde_t             (gain channel)

Both channels are driven by the same latent process, so `gamma` interpolates
between a purely coupling-expressed non-stationarity and a purely amplitude-
expressed one.

Why this separates the two features. Under a shared gain,

    Var_i(t)   = g(t)^2 var(s_i)
    Cov_ij(t)  = g(t)^2 cov(s_i, s_j)

so a covariance edge is degree-2 in amplitude and cannot tell a gain change from
a coupling change, whereas the gain cancels exactly in the correlation ratio.
Concretely the covariance edge becomes

    Cov_ij(t) = (1 + gamma d_t)^2 (c0 + c1 d_t)

which carries terms in d, d^2 and d^3. SFA extracts a *linear* projection of the
features, so it can only track a nonlinear function of the driver and R^2 falls.
This is the same linear-identifiability principle as the A0 = 0 pathology.
"""

from __future__ import annotations

import os
from copy import deepcopy

import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt

from src.VAR import simulate_var_based_on_C
from src.sweeps import CANON, RESULTS_DIR, build_C, cfg_with, evaluate_features


# ---------------------------------------------------------------------------
# Default regime
# ---------------------------------------------------------------------------
# CANON currently runs edge_win / tvp_period = 1.5, i.e. the window spans more
# than a full driver cycle and averages the driver away (recovery drops to
# ~0.49). The gain experiment needs headroom above and below to be readable, so
# the default here sets the window to one fifth of the driver period.

GAIN_CFG = cfg_with(edge_win=401)          # W/P = 0.2


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def simulate_gain(cfg: dict, seed: int, gamma: float = 0.0,
                  coupling: bool = True):
    """Simulate the two-channel system.

    Parameters
    ----------
    gamma     : strength of the shared multiplicative gain. The realised gain is
                1 + gamma*d_t, so gamma * tvp_amp must stay below 1 for the gain
                to remain positive.
    coupling  : if False the modulation pattern is zeroed, giving a stationary
                VAR observed through a time-varying gain (the gain-only control).

    Returns (X, d, info).
    """
    N = cfg["N"]
    amp = cfg["tvp_amp"]
    if gamma * amp >= 1.0:
        raise ValueError(
            f"gamma*tvp_amp = {gamma*amp:.2f} >= 1 would drive the gain "
            "non-positive; keep it below 1."
        )

    C_base = build_C(cfg, seed) if coupling else np.zeros((N, N))

    X0, d, A0, alpha_eff, C, rho_trace = simulate_var_based_on_C(
        N=N, T=cfg["T"], Sigma=cfg["Sigma"],
        tvp_amp=amp, tvp_period=cfg["tvp_period"], seed=seed,
        target_rho0=cfg["target_rho0"], alpha=cfg["alpha"],
        rho_max=cfg["rho_max"], tvp_phase_rad=cfg["tvp_phase_rad"],
        A_0_is_zero=cfg["A_0_is_zero"], C_base=C_base,
    )

    g = 1.0 + gamma * d
    X = X0 * g[:, None]

    info = dict(gamma=gamma, coupling=coupling, alpha_eff=alpha_eff,
                g_min=float(g.min()), g_max=float(g.max()),
                rho_max_realised=float(rho_trace.max()))
    return X, d, info


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

FEATURES_3 = ["edges-cov", "edges-corr", "node-var"]


def run_gain_sweep(gammas, seeds=range(8), cfg: dict | None = None,
                   names=None, cache: str | None = None,
                   overwrite: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Recovery of each feature as the gain channel is strengthened."""
    cfg = GAIN_CFG if cfg is None else cfg
    names = FEATURES_3 if names is None else names

    path = os.path.join(RESULTS_DIR, cache) if cache else None
    if path and os.path.exists(path) and not overwrite:
        if verbose:
            print(f"[cache] loading {path}")
        return pd.read_csv(path)

    rows = []
    for g in gammas:
        for s in seeds:
            X, d, info = simulate_gain(cfg, seed=s, gamma=g, coupling=True)
            r2 = evaluate_features(X, d, cfg, names=names)
            rows.append(dict(gamma=g, seed=s,
                             **{f"R2_{k}": v for k, v in r2.items()}))
        if verbose:
            blk = rows[-len(list(seeds)):]
            msg = "  ".join(f"{k.split('-')[-1]}={np.nanmean([r['R2_'+k] for r in blk]):.3f}"
                            for k in names)
            print(f"[gain] gamma = {g:.2f}   {msg}")

    df = pd.DataFrame(rows)
    if path:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df.to_csv(path, index=False)
        if verbose:
            print(f"[cache] wrote {path}")
    return df


def null_by_feature(cfg: dict, names=None, n_rep: int = 12, seed: int = 0,
                    gamma: float = 0.0) -> pd.DataFrame:
    """Driver-free null floor, computed separately for each feature.

    Simulates with zero driver amplitude, so the system is genuinely stationary,
    then scores each feature's slowest component against the sinusoid that would
    have been the driver. Anything above zero is the method manufacturing a
    driver-shaped trajectory out of intrinsic slowness.

    This matters for composite features specifically: concatenating blocks
    increases the feature count, which gives SFA more freedom to find a slow
    direction by chance. Comparing composites to single blocks on raw recovery
    alone would not control for that.
    """
    from src.features import ALL_FEATURES
    from src.util import make_driver

    names = FEATURES_3 if names is None else names
    flat = deepcopy(cfg)
    flat["tvp_amp"] = 0.0

    rows = []
    for k in range(n_rep):
        X, _, _ = simulate_gain(flat, seed=seed * 1000 + k, gamma=0.0,
                                coupling=True)
        d_ref = make_driver(cfg["T"], cfg["tvp_amp"], cfg["tvp_period"],
                            phase=cfg["tvp_phase_rad"])
        rows.append(evaluate_features(X, d_ref, cfg, names=names))
    return pd.DataFrame(rows)


CONDITIONS = {
    "neither":       dict(coupling=False, gamma=0.0),
    "coupling only": dict(coupling=True,  gamma=0.0),
    "gain only":     dict(coupling=False, gamma=0.6),
    "both":          dict(coupling=True,  gamma=0.6),
}


def run_conditions(seeds=range(8), cfg: dict | None = None, names=None,
                   conditions: dict | None = None,
                   cache: str | None = None, overwrite: bool = False,
                   verbose: bool = True) -> pd.DataFrame:
    """The gain x coupling dissociation.

    The decisive cell is `gain only`: with no coupling modulation present,
    covariance edges and node variance should both track the driver while
    correlation edges should read near zero. Any appreciable correlation there
    is an artefact of the normalisation rather than a coupling signal.
    """
    cfg = GAIN_CFG if cfg is None else cfg
    names = FEATURES_3 if names is None else names
    conditions = CONDITIONS if conditions is None else conditions

    path = os.path.join(RESULTS_DIR, cache) if cache else None
    if path and os.path.exists(path) and not overwrite:
        if verbose:
            print(f"[cache] loading {path}")
        return pd.read_csv(path)

    rows = []
    for label, spec in conditions.items():
        for s in seeds:
            X, d, info = simulate_gain(cfg, seed=s, **spec)
            r2 = evaluate_features(X, d, cfg, names=names)
            rows.append(dict(condition=label, seed=s, **spec,
                             **{f"R2_{k}": v for k, v in r2.items()}))
        if verbose:
            print(f"[cond] {label}")

    df = pd.DataFrame(rows)
    if path:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def run_phase_diagram(alpha_vals, gamma_vals, seeds=range(6), cfg: dict | None = None,
                      names=None, cache: str | None = None, overwrite: bool = False,
                      verbose: bool = True) -> pd.DataFrame:
    """2-D sweep: coupling strength (alpha) x gain strength (gamma), coupling=True throughout.

    Coupling strength is varied through `alpha` (pre-scaling: C = alpha * C_base)
    rather than `rho_max`, since rho_max caps A0 and C jointly and would not
    isolate the coupling axis. The global stability rescaling in
    `simulate_var_based_on_C` still shrinks A0 and C together to keep
    max_t rho(A_t) <= rho_max, so the requested `alpha` and the realised
    coupling strength (`alpha_eff`) can diverge near that cap -- `alpha_eff` is
    recorded per row so this compression is visible rather than silently biasing
    the diagram. Grid points with gamma * tvp_amp >= 1 are skipped (the gain
    channel would otherwise go non-positive).
    """
    cfg = GAIN_CFG if cfg is None else cfg
    names = ["edges-corr", "node-var"] if names is None else names
    tvp_amp = cfg["tvp_amp"]

    path = os.path.join(RESULTS_DIR, cache) if cache else None
    if path and os.path.exists(path) and not overwrite:
        if verbose:
            print(f"[cache] loading {path}")
        return pd.read_csv(path)

    rows = []
    for a in alpha_vals:
        cfg_a = deepcopy(cfg)
        cfg_a["alpha"] = a
        for g in gamma_vals:
            if g * tvp_amp >= 1.0:
                continue
            for s in seeds:
                X, d, info = simulate_gain(cfg_a, seed=s, gamma=g, coupling=True)
                r2 = evaluate_features(X, d, cfg_a, names=names)
                info = {k: v for k, v in info.items() if k not in ("gamma", "coupling")}
                rows.append(dict(alpha=a, gamma=g, coupling=True, seed=s, **info,
                                 **{f"R2_{k}": v for k, v in r2.items()}))
        if verbose:
            print(f"[phase] alpha = {a:.2f} done")

    df = pd.DataFrame(rows)
    if path:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df.to_csv(path, index=False)
        if verbose:
            print(f"[cache] wrote {path}")
    return df


# ---------------------------------------------------------------------------
# Continuous coupling <-> gain mixture (VAR_new_mechanism.ipynb)
# ---------------------------------------------------------------------------
# `run_conditions` and `run_phase_diagram` above compare discrete/gridded
# points. This section adds a single continuous path between the two pure
# mechanisms, parameterised by a mixture fraction `xi` in [0, 1]:
#
#     xi = 1  ->  coupling only  (alpha = alpha_max, gamma = 0)
#     xi = 0  ->  gain only      (alpha = 0,          gamma = gamma_max)
#
# via the linear path alpha(xi) = alpha_max*xi, gamma(xi) = gamma_max*(1-xi).
#
# `alpha` and `gamma` act through unrelated mechanisms (coupling vs a shared
# multiplicative envelope) and are not on a common physical scale, so `xi` is
# a label for *which point on this one chosen path* is being simulated, not an
# equal-effort or equal-energy interpolation. `calibrate_mixture_endpoints`
# below picks `gamma_max` so the two pure endpoints land at comparable R^2,
# which makes the path more interpretable, but does not make it a matched
# physical continuum -- that would require knowing the true relative
# magnitudes of coupling and gain non-stationarity in the real data, which is
# exactly the open question this simulation is informing.

def calibrate_mixture_endpoints(alpha_max: float, gamma_candidates,
                                seeds=range(6), cfg: dict | None = None,
                                target_range: tuple[float, float] = (0.75, 0.9),
                                verbose: bool = True) -> dict:
    """Pick `gamma_max` so the gain-only and coupling-only endpoints are comparable.

    Runs the coupling-only endpoint once (`alpha=alpha_max, gamma=0`, scored on
    `edges-corr`, since that is the feature specific to coupling) to get a
    target R^2, then the gain-only endpoint (`alpha=0`, scored on `node-var`,
    the feature specific to gain) at each candidate gamma. Reports the full
    table and recommends whichever candidate's node-variance R^2 is closest to
    the coupling-only correlation-edge R^2, without asserting the two effect
    sizes are actually matched in any physical sense -- see the module note
    above.
    """
    cfg = GAIN_CFG if cfg is None else cfg
    seeds = list(seeds)

    cfg_coupling = deepcopy(cfg)
    cfg_coupling["alpha"] = alpha_max
    coupling_r2 = np.empty(len(seeds))
    for i, s in enumerate(seeds):
        X, d, _ = simulate_gain(cfg_coupling, seed=s, gamma=0.0, coupling=True)
        coupling_r2[i] = evaluate_features(X, d, cfg_coupling, names=["edges-corr"])["edges-corr"]
    target = float(coupling_r2.mean())

    cfg_gain = deepcopy(cfg)
    cfg_gain["alpha"] = 0.0
    rows = []
    for g in gamma_candidates:
        r2s = np.empty(len(seeds))
        g_min = np.inf
        rho_max_realised = 0.0
        for i, s in enumerate(seeds):
            X, d, info = simulate_gain(cfg_gain, seed=s, gamma=g, coupling=True)
            r2s[i] = evaluate_features(X, d, cfg_gain, names=["node-var"])["node-var"]
            g_min = min(g_min, info["g_min"])
            rho_max_realised = max(rho_max_realised, info["rho_max_realised"])
        rows.append(dict(gamma=g, node_var_R2_mean=float(r2s.mean()),
                         node_var_R2_sd=float(r2s.std()),
                         gap_to_target=float(r2s.mean() - target),
                         g_min=float(g_min), rho_max_realised=float(rho_max_realised)))

    table = pd.DataFrame(rows)
    best = table.iloc[int(table["gap_to_target"].abs().values.argmin())]
    in_range = target_range[0] <= best["node_var_R2_mean"] <= target_range[1]

    if verbose:
        print(f"[calibrate] coupling-only (alpha_max={alpha_max}): "
              f"R2_edges-corr = {target:.3f} +/- {coupling_r2.std():.3f}  (n={len(seeds)} seeds)")
        print(table.round(3).to_string(index=False))
        print(f"[calibrate] recommended gamma_max = {best['gamma']:.3f}  "
              f"-> node-var R2 = {best['node_var_R2_mean']:.3f}  "
              f"(target {target:.3f}, gap {best['gap_to_target']:+.3f})")
        print(f"[calibrate] within target range {target_range}: {in_range}  "
              f"(exact matching is not required, see docstring)")

    return dict(alpha_max=alpha_max, coupling_R2=target, table=table,
               recommended_gamma_max=float(best["gamma"]),
               recommended_gamma_max_R2=float(best["node_var_R2_mean"]),
               within_target_range=bool(in_range))


def run_mechanism_mixture_sweep(xis, alpha_max: float, gamma_max: float,
                                seeds=range(8), cfg: dict | None = None,
                                names=None, cache: str | None = None,
                                overwrite: bool = False,
                                verbose: bool = True) -> pd.DataFrame:
    """Continuous coupling<->gain mixture along the linear path in `xi`.

        alpha(xi) = alpha_max * xi,   gamma(xi) = gamma_max * (1 - xi)

    `xi=0` is gain-only, `xi=1` is coupling-only. `coupling=True` is passed
    to `simulate_gain` at every `xi`, including 0: at `xi=0`, `alpha=0.0`
    makes `C = alpha*C_base = 0` exactly (float multiplication by 0.0 is exact,
    and `C_base`'s entries are all finite), and `simulate_var_based_on_C` calls
    `set_seed(seed)` -- resetting the global RNG -- only *after* `C_base` has
    already been computed, so nothing about how `C_base` was built can leak
    into the A0 draw or the driver. This is verified to be numerically exact
    (not just statistically similar) against `coupling=False` in the smoke
    test in `VAR_new_mechanism.ipynb`.

    Stability (`max_t rho(A_t) < 1`) and gain positivity (`1+gamma*d_t > 0` for
    all t) both reuse the existing safeguards: the former is the global
    rescaling already inside `simulate_var_based_on_C` (triggered via `alpha`,
    exactly as every other sweep in this codebase varies coupling strength),
    the latter is the `gamma*tvp_amp < 1` check already inside `simulate_gain`
    -- raised here eagerly, per `xi`, with a clearer message.

    See the module note above `calibrate_mixture_endpoints` for why `xi` is a
    path label, not a matched-effect-size interpolation.
    """
    cfg = GAIN_CFG if cfg is None else cfg
    names = FEATURES_3 if names is None else names
    tvp_amp = cfg["tvp_amp"]
    xis = list(xis)

    path = os.path.join(RESULTS_DIR, cache) if cache else None
    if path and os.path.exists(path) and not overwrite:
        if verbose:
            print(f"[cache] loading {path}")
        return pd.read_csv(path)

    rows = []
    for xi in xis:
        a = alpha_max * xi
        g = gamma_max * (1.0 - xi)
        if g * tvp_amp >= 1.0:
            raise ValueError(
                f"gamma({xi})*tvp_amp = {g*tvp_amp:.3f} >= 1 at xi={xi}; "
                "lower gamma_max."
            )
        cfg_xi = deepcopy(cfg)
        cfg_xi["alpha"] = a
        for s in seeds:
            X, d, info = simulate_gain(cfg_xi, seed=s, gamma=g, coupling=True)
            r2 = evaluate_features(X, d, cfg_xi, names=names)
            row = {
                "xi": xi,
                "alpha_requested": a,
                "gamma_requested": g,
                "seed": s,
                "alpha_effective": info["alpha_eff"],
                "gamma_effective": g,   # gamma is not touched by the stability rescale
                "g_min": info["g_min"],
                "g_max": info["g_max"],
                "rho_max_realised": info["rho_max_realised"],
            }
            row.update({f"R2_{k}": v for k, v in r2.items()})
            rows.append(row)
        if verbose:
            blk = rows[-len(list(seeds)):]
            msg = "  ".join(f"{k.split('-')[-1]}={np.nanmean([r['R2_'+k] for r in blk]):.3f}"
                            for k in names)
            print(f"[mix] xi={xi:.2f}  alpha={a:.3f} gamma={g:.3f}   {msg}")

    df = pd.DataFrame(rows)
    if path:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df.to_csv(path, index=False)
        if verbose:
            print(f"[cache] wrote {path}")
    return df


# ---------------------------------------------------------------------------
# Robustness check: coupling driver + an UNRELATED shared-gain nuisance
# ---------------------------------------------------------------------------
# Every simulation above (`simulate_gain`, the mixture path) uses the SAME
# latent d_t for both the coupling channel and the gain channel, so covariance
# edges are allowed to "cheat": some of what makes them recover d_t well is
# that the gain itself is d_t-shaped. This section asks a stricter question:
# if a *different*, independent process z_t drives a shared gain while d_t
# alone drives coupling, and recovery is scored only against d_t, does
# correlation's algebraic gain-invariance actually protect it in practice?

DEFAULT_NUISANCE_PERIOD = 733   # samples; no small-integer ratio with CANON's tvp_period=2000


def make_nuisance_process(d: np.ndarray, seed: int, period: int, amp: float,
                          filter_order: int = 4) -> tuple[np.ndarray, float]:
    """Build a nuisance process z, independent of driver `d`, same length as `d`.

    Not a phase-shifted copy of `d` -- z is *stochastic*, not periodic:
    Gaussian white noise, zero-phase low-pass filtered (Butterworth,
    `filtfilt` so there is no group delay and the output stays length `T`)
    with a cutoff at `1/period` cycles/sample. `period` sets how slowly z
    varies; the default (733) has no small-integer ratio with CANON's
    `tvp_period` (2000), so the two processes do not share frequency content
    even coincidentally.

    Independence by construction is not enough on a finite sample -- a random
    draw can still have nonzero sample correlation with `d` by chance. `z` is
    therefore explicitly residualised: the OLS projection of (mean-centred) z
    onto (mean-centred) `d` is subtracted, which makes the sample covariance
    (and hence Pearson correlation) between the residual and `d` exactly zero
    up to floating-point error, not merely small. The residual is then
    rescaled to have peak amplitude `amp`.

    Returns `(z, corr_d_z)`; `corr_d_z` should print as ~1e-15, not just
    "small", if the residualisation worked.
    """
    T = len(d)
    rng = np.random.default_rng(seed + 90210)   # independent stream; does not touch the
                                                 # legacy global RNG `simulate_var_based_on_C` uses
    noise = rng.normal(size=T)
    cutoff = 1.0 / period
    b, a = butter(filter_order, cutoff, btype="low")
    z_raw = filtfilt(b, a, noise)

    d_c = d - d.mean()
    z_c = z_raw - z_raw.mean()
    beta = np.dot(z_c, d_c) / np.dot(d_c, d_c)
    z_orth = z_c - beta * d_c    # residual is exactly orthogonal to d_c by OLS construction

    peak = np.max(np.abs(z_orth))
    z = (amp / peak) * z_orth if peak > 0 else z_orth
    corr_d_z = float(np.corrcoef(z, d)[0, 1])
    return z, corr_d_z


def simulate_coupling_with_nuisance_gain(cfg: dict, seed: int, alpha_fixed: float,
                                         gamma: float, nuisance_period: int | None = None,
                                         nuisance_amp: float | None = None,
                                         coupling: bool = True):
    """Coupling driven by d_t; an independent, orthogonalised z_t drives shared gain.

        xtilde_t = (A0 + alpha_fixed * d_t * C_base) xtilde_{t-1} + eta_t
        x_t      = (1 + gamma * z_t) * xtilde_t

    `alpha_fixed` is passed straight through as `alpha` to
    `simulate_var_based_on_C` (the project's existing pre-scaling coupling
    parameter) and is NOT swept here -- the coupling channel's strength stays
    constant while `gamma` varies, unlike the xi-mixture path, precisely
    so covariance can't be helped by coupling changing too. Stability
    (`max_t rho(A_t) < 1`) reuses the same global-rescale machinery every
    other simulator in this module relies on; gain positivity
    (`1+gamma*z_t > 0`) is enforced the same way `simulate_gain` enforces it
    for `d_t`, just against `nuisance_amp` instead of `tvp_amp`.

    Returns `(X, d_target, z_gain, info)`. `d_target` is what recovery must
    always be scored against.
    """
    N = cfg["N"]
    T = cfg["T"]

    if nuisance_period is None:
        nuisance_period = DEFAULT_NUISANCE_PERIOD
    if nuisance_amp is None:
        nuisance_amp = cfg["tvp_amp"]     # same useful range as the existing driver

    if gamma * nuisance_amp >= 1.0:
        raise ValueError(
            f"gamma*nuisance_amp = {gamma*nuisance_amp:.3f} >= 1 would drive the "
            "gain non-positive; keep it below 1."
        )

    C_base = build_C(cfg, seed) if coupling else np.zeros((N, N))

    X0, d, A0, alpha_eff, C, rho_trace = simulate_var_based_on_C(
        N=N, T=T, Sigma=cfg["Sigma"],
        tvp_amp=cfg["tvp_amp"], tvp_period=cfg["tvp_period"], seed=seed,
        target_rho0=cfg["target_rho0"], alpha=alpha_fixed,
        rho_max=cfg["rho_max"], tvp_phase_rad=cfg["tvp_phase_rad"],
        A_0_is_zero=cfg["A_0_is_zero"], C_base=C_base,
    )

    z, corr_d_z = make_nuisance_process(d, seed=seed, period=nuisance_period, amp=nuisance_amp)

    g = 1.0 + gamma * z
    X = X0 * g[:, None]

    info = dict(alpha_effective=alpha_eff, gamma=gamma,
               g_min=float(g.min()), g_max=float(g.max()),
               rho_max_realised=float(rho_trace.max()),
               corr_d_z=corr_d_z, nuisance_period=nuisance_period,
               nuisance_amp=nuisance_amp)
    return X, d, z, info


def run_nuisance_gain_sweep(gammas, alpha_fixed: float, seeds=range(8),
                            cfg: dict | None = None, names=None,
                            nuisance_period: int | None = None,
                            nuisance_amp: float | None = None,
                            cache: str | None = None, overwrite: bool = False,
                            verbose: bool = True) -> pd.DataFrame:
    """Sweep nuisance gain strength `gamma`; every feature is scored against `d_target` only.

    `alpha_fixed` is held constant across the whole sweep (see
    `simulate_coupling_with_nuisance_gain`), so any decline in a feature's
    R^2 as gamma grows is attributable only to the unrelated nuisance gain
    contaminating that feature, not to coupling itself weakening.
    """
    cfg = GAIN_CFG if cfg is None else cfg
    names = FEATURES_3 if names is None else names

    path = os.path.join(RESULTS_DIR, cache) if cache else None
    if path and os.path.exists(path) and not overwrite:
        if verbose:
            print(f"[cache] loading {path}")
        return pd.read_csv(path)

    rows = []
    for g in gammas:
        for s in seeds:
            X, d, z, info = simulate_coupling_with_nuisance_gain(
                cfg, seed=s, alpha_fixed=alpha_fixed, gamma=g,
                nuisance_period=nuisance_period, nuisance_amp=nuisance_amp,
            )
            r2 = evaluate_features(X, d, cfg, names=names)
            row = {
                "seed": s,
                "gamma": g,
                "alpha_effective": info["alpha_effective"],
                "corr_d_z": info["corr_d_z"],
                "g_min": info["g_min"],
                "g_max": info["g_max"],
                "rho_max_realised": info["rho_max_realised"],
            }
            row.update({f"R2_{k}": v for k, v in r2.items()})
            rows.append(row)
        if verbose:
            blk = rows[-len(list(seeds)):]
            msg = "  ".join(f"{k.split('-')[-1]}={np.nanmean([r['R2_'+k] for r in blk]):.3f}"
                            for k in names)
            mean_corr_dz = np.mean([r["corr_d_z"] for r in blk])
            print(f"[nuisance] gamma={g:.3f}  mean|corr(d,z)|~{abs(mean_corr_dz):.2e}   {msg}")

    df = pd.DataFrame(rows)
    if path:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df.to_csv(path, index=False)
        if verbose:
            print(f"[cache] wrote {path}")
    return df
