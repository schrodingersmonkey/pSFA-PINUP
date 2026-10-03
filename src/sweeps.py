"""
Sweep engine for the p-SFA VAR simulations.

This module owns *everything except plotting*: the VAR parameter regime,
the simulate -> edge-features -> SFA -> metrics path for a single parameter
point, the seed-averaged sweep loop, and on-disk caching of results.

Plotting lives in the notebook.

All simulation and inference is delegated to the original PINUP source
(`src.VAR`, `src.edge_pipeline`) so numbers are reproducible against the
submitted manuscript.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.VAR import (
    build_C_manual,
    simulate_var_based_on_C,
    simulate_var_based_on_C_with_noise__in_TVP,
)
from src.edge_pipeline import run_sfa_pipeline_windowed
from src.sfa import affine_align, pearson_r, r2_score, spearman_rho

RESULTS_DIR = str(Path(__file__).resolve().parent.parent / "outputs" / "results" / "var")


# ---------------------------------------------------------------------------
# 1. Canonical operating regime
# ---------------------------------------------------------------------------

CANON = dict(
    # --- generative model -------------------------------------------------
    N=6,                     # number of nodes
    T=10_000,                # number of samples
    Sigma=0.1,               # innovation (observation) noise variance
    tvp_amp=1.0,             # a_TVP: driver amplitude
    tvp_period=2_000,        # P: driver period      -> P/T = 0.5
    tvp_phase_rad=0.0,       # phase offset
    target_rho0=0.8,         # spectral radius of the baseline A0
    alpha=0.9,               # modulation strength (pre-scaling)
    rho_max=0.8,             # cap on max_t rho(A_t); realised value is 0.95*rho_max
    A_0_is_zero=False,       # A0 = 0 triggers the d^2 identifiability issue
    tvp_noise_std=0.0,       # sigma_d: dynamical noise injected into the driver
    # --- C pattern --------------------------------------------------------
    C_entry_dist="uniform",
    C_dist_scale=1.0,
    C_include_diagonal=False,
    C_symmetric=True,
    C_frac=1.0,              # fraction of nodes inside the modulated subnetwork
    # --- p-SFA pipeline ---------------------------------------------------
    windowing_mode="rolling",
    node_win=1,
    edge_win=601,            # W: centred rolling window on the edge features
)


def cfg_with(**overrides) -> dict:
    """Copy of CANON with `overrides` applied. Unknown keys raise."""
    unknown = set(overrides) - set(CANON)
    if unknown:
        raise KeyError(f"Unknown config key(s): {sorted(unknown)}")
    cfg = deepcopy(CANON)
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# 2. Modulation pattern C
# ---------------------------------------------------------------------------

def build_C(cfg: dict, seed: int) -> np.ndarray:
    """Un-scaled modulation pattern, optionally restricted to a subnetwork.

    `C_frac` < 1 keeps only a randomly chosen ceil(frac*N) nodes; all edges
    touching an excluded node are zeroed. This is the localisation experiment
    (Fig. 13 of the manuscript).
    """
    N = cfg["N"]
    C = build_C_manual(
        N=N,
        rng_seed=seed,
        entry_dist=cfg["C_entry_dist"],
        dist_scale=cfg["C_dist_scale"],
        include_diagonal=cfg["C_include_diagonal"],
        symmetric=cfg["C_symmetric"],
    )

    frac = float(cfg["C_frac"])
    if frac >= 1.0:
        return C

    k = max(2, int(np.ceil(frac * N)))
    rng = np.random.default_rng(seed + 4242)
    active = rng.choice(N, size=k, replace=False)
    mask = np.zeros((N, N), dtype=bool)
    mask[np.ix_(active, active)] = True
    return C * mask


# ---------------------------------------------------------------------------
# 3. Simulation
# ---------------------------------------------------------------------------

def simulate(cfg: dict, seed: int):
    """Run the appropriate VAR generator for `cfg`.

    Returns (X, d, info) where info carries realised stability diagnostics.
    """
    C_base = build_C(cfg, seed)

    common = dict(
        N=cfg["N"],
        T=cfg["T"],
        Sigma=cfg["Sigma"],
        tvp_amp=cfg["tvp_amp"],
        tvp_period=cfg["tvp_period"],
        seed=seed,
        target_rho0=cfg["target_rho0"],
        alpha=cfg["alpha"],
        rho_max=cfg["rho_max"],
        tvp_phase_rad=cfg["tvp_phase_rad"],
        A_0_is_zero=cfg["A_0_is_zero"],
        C_base=C_base,
    )

    if cfg["tvp_noise_std"] > 0:
        X, d, A0, alpha_eff, C, rho_trace = simulate_var_based_on_C_with_noise__in_TVP(
            **common, tvp_noise_std=cfg["tvp_noise_std"]
        )
    else:
        X, d, A0, alpha_eff, C, rho_trace = simulate_var_based_on_C(**common)

    from src.util import spectral_radius

    info = dict(
        alpha_eff=alpha_eff,
        rho_A0=spectral_radius(A0),          # realised, post global scaling
        rho_max_realised=float(rho_trace.max()),
        rho_mean_realised=float(rho_trace.mean()),
    )
    return X, d, info


# ---------------------------------------------------------------------------
# 4. Inference + metrics for a single parameter point
# ---------------------------------------------------------------------------

def evaluate(X, d, cfg: dict, features: str = "edges") -> dict:
    """p-SFA (features='edges') or the node-SFA baseline (features='nodes')."""
    res = run_sfa_pipeline_windowed(
        X, d,
        features=features,
        windowing_mode=cfg["windowing_mode"],
        node_win=cfg["node_win"],
        edge_win=cfg["edge_win"],
        plots=False,
    )
    return res


def run_point(cfg: dict, seed: int, features=("edges", "nodes"),
              n_null: int = 0, generator: str = "coupling") -> dict:
    """Simulate once and evaluate under each feature representation.

    `n_null` > 0 additionally computes a circular-shift surrogate null for the
    edge representation, returning the 95th percentile of the null R^2.
    """
    if generator == "coupling":
        X, d, info = simulate(cfg, seed)
    else:
        raise ValueError(f"Unknown generator: {generator}")

    row = dict(seed=seed, generator=generator, **info)

    for f in features:
        res = evaluate(X, d, cfg, features=f)
        tag = "edge" if f.startswith("edges") else "node"
        row[f"R2_{tag}"] = res["R2"]
        row[f"rP_{tag}"] = res["rP"]
        row[f"rS_{tag}"] = res["rS"]

        if n_null > 0 and tag == "edge":
            null = null_r2_driverfree(cfg, n_rep=n_null, seed=seed)
            row["R2_edge_null_p95"] = float(np.percentile(null, 95))
            row["R2_edge_null_mean"] = float(np.mean(null))

    return row


def null_r2_driverfree(cfg: dict, n_rep: int = 20, seed: int = 0) -> np.ndarray:
    """Driver-free null: the R^2 floor produced by endogenous slowness alone.

    Simulates the *same* system with a_TVP = 0, so A_t = A0 is constant and the
    process is genuinely stationary. p-SFA is then run as usual and its slowest
    component is scored against the sinusoid that would have been the driver.

    Anything above zero here is p-SFA manufacturing a driver-shaped trajectory
    out of intrinsic autocorrelation. This is the null that matters for the
    spectral-radius sweep, where long-memory dynamics produce slow edge
    plateaus with no TVP present at all.
    """
    from src.util import make_driver

    null_cfg = deepcopy(cfg)
    null_cfg["tvp_amp"] = 0.0
    null_cfg["tvp_noise_std"] = 0.0

    out = np.empty(n_rep)
    for k in range(n_rep):
        s = seed * 1000 + k
        X, _, _ = simulate(null_cfg, seed=s)
        res = evaluate(X, np.zeros(cfg["T"]), cfg, features="edges")
        # Score against the driver this system *would* have had.
        d_ref = make_driver(cfg["T"], cfg["tvp_amp"], cfg["tvp_period"],
                            phase=cfg["tvp_phase_rad"])
        d_ref = _match_length(d_ref, res["y_sfa"], cfg)
        y_fit, _, _ = affine_align(res["y_sfa"], d_ref)
        out[k] = r2_score(d_ref, y_fit)
    return out


def _match_length(d_full: np.ndarray, y: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply the same windowing-induced truncation the pipeline applied to d."""
    from src.util import moving_average_centered_any

    if cfg["windowing_mode"] == "rolling":
        d = moving_average_centered_any(d_full, max(1, int(cfg["node_win"])))
        d = moving_average_centered_any(d, max(1, int(cfg["edge_win"])))
    else:
        tau, tau_e = max(1, int(cfg["node_win"])), max(1, int(cfg["edge_win"]))
        Tb = (len(d_full) // tau) * tau
        d = d_full[:Tb].reshape(Tb // tau, tau).mean(axis=1)
        Te = (len(d) // tau_e) * tau_e
        d = d[:Te].reshape(Te // tau_e, tau_e).mean(axis=1)
    if len(d) != len(y):                      # defensive trim
        n = min(len(d), len(y))
        d = d[:n]
    return d


def null_r2_circshift(y_sfa: np.ndarray, d_aligned: np.ndarray, n_shift: int,
                      seed: int = 0) -> np.ndarray:
    """Circular-shift surrogate null. NOT appropriate for a periodic driver.

    A shifted sinusoid still correlates with the original as cos(2*pi*s/P), so
    this null has a mean R^2 near 0.5 and a 95th percentile near the observed
    value regardless of how good the recovery is. Retained only for reference /
    for use with aperiodic drivers.
    """
    T = len(d_aligned)
    rng = np.random.default_rng(seed + 777)
    shifts = rng.integers(int(0.05 * T), int(0.95 * T), size=n_shift)
    out = np.empty(n_shift)
    for k, s in enumerate(shifts):
        d_shift = np.roll(d_aligned, s)
        y_fit, _, _ = affine_align(y_sfa, d_shift)
        out[k] = r2_score(d_shift, y_fit)
    return out


# ---------------------------------------------------------------------------
# 5. Sweep loop + caching
# ---------------------------------------------------------------------------

def run_sweep(param: str,
              values,
              seeds=range(10),
              base: dict | None = None,
              features=("edges", "nodes"),
              n_null: int = 0,
              generator: str = "coupling",
              derive=None,
              cache: str | None = None,
              overwrite: bool = False,
              verbose: bool = True) -> pd.DataFrame:
    """Sweep one config key across `values`, repeating over `seeds`.

    Parameters
    ----------
    param    : config key to vary (must exist in CANON).
    values   : iterable of values for that key.
    derive   : optional callable(cfg, value) -> cfg, applied after setting
               `param`. Use it to couple a second parameter to the swept one
               (e.g. locking the edge window to P/4 in the period sweep).
    cache    : filename under results/. Reloaded instead of recomputed unless
               `overwrite=True`.

    Returns a tidy DataFrame: one row per (value, seed).
    """
    base = deepcopy(CANON) if base is None else deepcopy(base)

    path = os.path.join(RESULTS_DIR, cache) if cache else None
    if path and os.path.exists(path) and not overwrite:
        if verbose:
            print(f"[cache] loading {path}")
        return pd.read_csv(path)

    rows = []
    for v in values:
        cfg = deepcopy(base)
        cfg[param] = v
        if derive is not None:
            cfg = derive(cfg, v)
        for s in seeds:
            row = run_point(cfg, seed=s, features=features,
                            n_null=n_null, generator=generator)
            row[param] = v
            row["edge_win"] = cfg["edge_win"]
            rows.append(row)
        if verbose:
            got = [r["R2_edge"] for r in rows[-len(list(seeds)):]
                   if "R2_edge" in r]
            msg = f"  mean R2_edge = {np.mean(got):.3f}" if got else ""
            print(f"[sweep] {param} = {v}{msg}")

    df = pd.DataFrame(rows)
    if path:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df.to_csv(path, index=False)
        if verbose:
            print(f"[cache] wrote {path}")
    return df


# ---------------------------------------------------------------------------
# 5b. Multi-feature variant
# ---------------------------------------------------------------------------
# Same sweeps, but every feature representation in src.features is evaluated on
# the *identical* simulated X for a given (parameter, seed). Any difference
# between features is therefore attributable to the representation alone, with
# no additional sampling variability.

def component0_scores(F: np.ndarray, d_aligned: np.ndarray, method: str = "sfa",
                       n_components: int = 5, return_component: bool = False):
    """R^2 (affine-aligned) and |Pearson r| of the first/slowest component
    against the driver.

    method="sfa" -> sksfa.SFA slowest component (component 0).
    method="pca" -> sklearn PCA first component (most variance).

    Generalises `sfa_c0_r2` to also support PCA and to report |r| alongside
    R^2, for direct SFA-vs-PCA comparison on the same feature matrix. Used by
    both the VAR sweeps (via `sfa_c0_r2` below) and the Neuropixels
    application notebook, so the two are scored by the same estimator.

    `return_component=True` additionally returns the raw (pre-alignment)
    component as a third value. R^2 doesn't need this -- for a single linear
    component, R^2 from the affine-aligned fit is exactly the squared Pearson
    correlation, so the score is identical whether or not alignment ever
    happens. The raw component is only useful for plotting the recovered
    signal against the driver, which needs its own `affine_align` call at
    the point of plotting to get a sensibly-scaled line -- not bundled into
    every scoring call here.
    """
    if not np.all(np.isfinite(F)):
        return (np.nan, np.nan, None) if return_component else (np.nan, np.nan)
    k = min(n_components, F.shape[1])
    if method == "sfa":
        from sksfa import SFA
        y = SFA(n_components=k, fill_mode="zero").fit_transform(F)[:, 0]
    elif method == "pca":
        from sklearn.decomposition import PCA
        y = PCA(n_components=k).fit_transform(F)[:, 0]
    else:
        raise ValueError(f"Unknown method '{method}', expected 'sfa' or 'pca'")
    y_fit, _, _ = affine_align(y, d_aligned)
    R2 = r2_score(d_aligned, y_fit)
    absr = abs(pearson_r(d_aligned, y_fit))
    if return_component:
        return R2, absr, y
    return R2, absr


def sfa_c0_r2(F: np.ndarray, d_aligned: np.ndarray, n_components: int = 5) -> float:
    """R^2 of the slowest SFA component against the driver.

    Mirrors `_c0` in correlation_testing.ipynb so simulation and Neuropixels
    numbers are produced by the same estimator.
    """
    R2, _ = component0_scores(F, d_aligned, method="sfa", n_components=n_components)
    return R2


def evaluate_features(X: np.ndarray, d: np.ndarray, cfg: dict,
                      names=None, n_components: int = 5) -> dict:
    """R^2 for each named feature representation on one simulated dataset."""
    from src.features import ALL_FEATURES, build_feature
    from src.features import _ma

    if cfg["windowing_mode"] != "rolling":
        raise ValueError("The multi-feature path assumes rolling windows.")

    names = ALL_FEATURES if names is None else names
    W = cfg["edge_win"]

    # node-level pre-smoothing, applied identically to X and to the driver
    Xn = _ma(X, cfg["node_win"])
    d_sm = _ma(_ma(d, cfg["node_win"]), W)

    out = {}
    for nm in names:
        try:
            F = build_feature(nm, Xn, W)
        except Exception:
            out[nm] = np.nan
            continue
        n = min(len(d_sm), F.shape[0])
        out[nm] = sfa_c0_r2(F[:n], d_sm[:n], n_components)
    return out


def run_point_features(cfg: dict, seed: int, names=None,
                       generator: str = "coupling") -> dict:
    if generator == "coupling":
        X, d, info = simulate(cfg, seed)
    else:
        raise ValueError(f"Unknown generator: {generator}")

    row = dict(seed=seed, generator=generator, **info)
    for nm, r2 in evaluate_features(X, d, cfg, names=names).items():
        row[f"R2_{nm}"] = r2
    return row


def run_sweep_features(param: str,
                       values,
                       seeds=range(10),
                       base: dict | None = None,
                       names=None,
                       generator: str = "coupling",
                       derive=None,
                       cache: str | None = None,
                       overwrite: bool = False,
                       verbose: bool = True) -> pd.DataFrame:
    """`run_sweep`, but scoring every feature representation per simulation."""
    from src.features import ALL_FEATURES

    base = deepcopy(CANON) if base is None else deepcopy(base)
    names = ALL_FEATURES if names is None else names

    path = os.path.join(RESULTS_DIR, cache) if cache else None
    if path and os.path.exists(path) and not overwrite:
        if verbose:
            print(f"[cache] loading {path}")
        return pd.read_csv(path)

    seeds = list(seeds)
    rows = []
    for v in values:
        cfg = deepcopy(base)
        cfg[param] = v
        if derive is not None:
            cfg = derive(cfg, v)
        for s in seeds:
            row = run_point_features(cfg, seed=s, names=names, generator=generator)
            row[param] = v
            row["edge_win"] = cfg["edge_win"]
            rows.append(row)
        if verbose:
            blk = rows[-len(seeds):]
            best = {nm: np.nanmean([r[f"R2_{nm}"] for r in blk]) for nm in names}
            top = max(best, key=lambda k: best[k])
            print(f"[sweep] {param} = {v}   best: {top} = {best[top]:.3f}")

    df = pd.DataFrame(rows)
    if path:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df.to_csv(path, index=False)
        if verbose:
            print(f"[cache] wrote {path}")
    return df


def summarise(df: pd.DataFrame, param: str, metric: str = "R2_edge") -> pd.DataFrame:
    """Mean / SD / n of `metric` grouped by the swept parameter."""
    g = df.groupby(param)[metric]
    return pd.DataFrame({
        param: g.mean().index,
        "mean": g.mean().values,
        "sd": g.std().values,
        "n": g.count().values,
    })


# ---------------------------------------------------------------------------
# 6. Derive helpers for coupled parameters
# ---------------------------------------------------------------------------

def lock_window_to_period(ratio: float = 4.0):
    """derive= callable that sets edge_win = P/ratio (odd), as in Fig. 4B.

    Keeps each window a locally quasi-stationary segment of the driver, so the
    period sweep measures the driver's intrinsic timescale rather than a
    window artefact.
    """
    def _derive(cfg, value):
        W = int(round(cfg["tvp_period"] / ratio))
        cfg["edge_win"] = max(3, W + (W + 1) % 2)   # force odd, >= 3
        return cfg
    return _derive
