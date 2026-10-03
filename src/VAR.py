import numpy as np
import matplotlib.pyplot as plt
from src.util import *

def simulate_var_low_rank(
    N: int,
    T: int,
    Sigma: float,
    tvp_amp: float,
    tvp_period: int,
    seed: int,
    target_rho0: float,
    alpha: float,
    rho_max: float,
    symmetric_uv: bool,
    tvp_phase_rad: float,
    A_0_is_zero: bool,
):
    """
    Simulate VAR(1) with A(t) = A0 + alpha_eff * d(t) * u v^T, innovations ~ N(0, Sigma I).
    Returns: X, d, A0, alpha_eff, rho_trace
    """
    set_seed(seed)

    # Driver
    d = make_driver(T, tvp_amp, tvp_period, phase=tvp_phase_rad)

    #if A_0 =0, we only inject the TVP (no other lagged dynamics)
    #else, we initialise with a spectral radius of target_rho0

    if A_0_is_zero:
        A0 = np.zeros((N, N))
    else:
        A0_raw = np.random.normal(loc=0.0, scale=1.0, size=(N, N))
        A0 = scale_to_spectral_radius(A0_raw, target_rho0)
        

    # Rank-1 directions u,v
    if symmetric_uv:
        u = orthonormal_columns(N, 1)[:, 0]
        v = u.copy()
    else:
        U = orthonormal_columns(N, 2)
        u, v = U[:, 0], U[:, 1]

    print()
    # Enforce stability over time ⇒ choose alpha_eff
    rho0 = spectral_radius(A0)
    max_extra = rho_max - rho0
    if max_extra <= 0:
        # create headroom
        A0 = scale_to_spectral_radius(A0, rho_max * 0.8)
        rho0 = spectral_radius(A0)
        max_extra = rho_max - rho0

    alpha_eff = alpha
    if tvp_amp > 0:
        alpha_eff = min(alpha, max_extra / (tvp_amp + 1e-12))
        alpha_eff *= 0.95  # margin

    # Simulate
    X = np.zeros((T, N), float)
    x_prev = np.zeros(N, float)
    rho_trace = np.empty(T, float)

    L_eta = np.sqrt(Sigma) * np.eye(N)

    for t in range(T):
        At = A0 + alpha_eff * d[t] * np.outer(u, v)
        # cache rho for diagnostics; if too slow, sample sparsely
        rho_trace[t] = spectral_radius(At)
        eta_t = L_eta @ np.random.normal(size=N)
        x_t = At @ x_prev + eta_t
        X[t] = x_t
        x_prev = x_t

    return X, d, A0, alpha_eff, rho_trace



def build_C_manual(
    N: int,
    *,
    rng_seed: int | None = None,
    entry_dist: str = "normal",   # "normal" | "uniform" | "uniform_pos"
    dist_scale: float = 1.0,
    include_diagonal: bool = True,
    symmetric: bool = True,   
) -> np.ndarray:
    """
    Build an UN-SCALED symmetric coefficient pattern C_base (no zeros by design).
    Fills the full NxN (upper triangle) and mirrors to enforce symmetry.

    entry_dist options (only these three):
      - "normal"      -> N(0, dist_scale^2)          (signed)
      - "uniform"     -> U(-dist_scale, +dist_scale) (signed)
      - "uniform_pos" -> U(0, +dist_scale)           (nonnegative)
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    if entry_dist not in {"normal", "uniform", "uniform_pos"}:
        raise ValueError(f"Unsupported entry_dist: {entry_dist}")

    def draw():
        if entry_dist == "normal":
            return np.random.normal(0.0, dist_scale)
        if entry_dist == "uniform":
            return np.random.uniform(-dist_scale, +dist_scale)
        # entry_dist == "uniform_pos"
        return np.random.uniform(0.0, +dist_scale)

    C = np.zeros((N, N), dtype=float)

    if symmetric:
        # Fill upper triangle and mirror
        for i in range(N):
            j0 = i if include_diagonal else i + 1
            for j in range(j0, N):
                val = draw()
                C[i, j] = val
                C[j, i] = val
    else:
        # Fill every entry independently
        for i in range(N):
            for j in range(N):
                if i == j and not include_diagonal:
                    continue
                C[i, j] = draw()

    return C


def simulate_var_based_on_C(
    N: int,
    T: int,
    Sigma: float,
    tvp_amp: float,
    tvp_period: int,
    seed: int,
    target_rho0: float,
    alpha: float,                # requested modulation strength (pre-scaling)
    rho_max: float,
    tvp_phase_rad: float,
    A_0_is_zero: bool,
    C_base: np.ndarray,          # UN-SCALED pattern (from builder)
    safety_margin: float = 0.95, # cushion against numerical slop
    plot_rho_trace: bool = False,
    plot_heatmap: bool = False,
    plot_time_series: bool = False,
):
    """
    Simulate VAR(1) with:
        A(t) = A0 + d(t) * C,    where C = alpha_eff * C_base
    and we perform a single global scale s on BOTH A0 and C so that
        max_{t} rho( A0*s + d(t)*C*s ) <= rho_max.

    Returns
    -------
    X : (T, N) simulated time series
    d : (T,) driver
    A0 : (N, N) *after global scaling*
    alpha_eff : float, effective modulation (after global scaling)
    C : (N, N) = alpha_eff * C_base  (after global scaling)
    rho_trace : (T,) spectral radius of A(t) per time
    """
    set_seed(seed)

    # 1) Driver
    d = make_driver(T, tvp_amp, tvp_period, phase=tvp_phase_rad)

    # 2) Baseline A0 (not scaled yet)
    if A_0_is_zero:
        A0 = np.zeros((N, N))
    else:
        A0_raw = np.random.normal(loc=0.0, scale=1.0, size=(N, N))
        A0 = scale_to_spectral_radius(A0_raw, target_rho0) #target rho0 is the intial spectral radius of A. 

    # 3) Time-invariant TVP matrix before global scaling
    #    (alpha is the requested pre-scale strength)
    #C_base = build_C_manual(N, rng_seed=seed, entry_dist="uniform", dist_scale=1.0, include_diagonal=False, symmetric = True)

    C = alpha * C_base
    alpha_eff = alpha  # we will need to adjust alpha so we can control the spectral radius 

    # 4) One-shot GLOBAL scaling to enforce max_t rho(A(t)) <= rho_max
    #    For a bounded driver with extrema at ±tvp_amp, the worst case is attained at those points.
    if tvp_amp > 0:
        rho_plus  = spectral_radius(A0 + (+tvp_amp) * C)
        rho_minus = spectral_radius(A0 + (-tvp_amp) * C)
        rho_worst = max(rho_plus, rho_minus)
    else:
        rho_worst = spectral_radius(A0)
    
    #s is a global shrinkage factor
    if rho_worst > 0:
        s = min(1.0, (rho_max * safety_margin) / rho_worst)
    else:
        s = 1.0

    A0 *= s
    C  *= s
    alpha_eff *= s  # book-keeping: effective modulation after global scale

    # 5) Simulate
    X = np.zeros((T, N), dtype=float)
    x_prev = np.zeros(N, dtype=float)
    rho_trace = np.empty(T, dtype=float)
    L_eta = np.sqrt(Sigma) * np.eye(N)

    for t in range(T):
        At = A0 + d[t] * C
        rho_trace[t] = spectral_radius(At)
        eta_t = L_eta @ np.random.normal(size=N)
        x_t = At @ x_prev + eta_t
        X[t] = x_t
        x_prev = x_t

    if plot_rho_trace:
        plot_rho_trace_figure(rho_trace=rho_trace, rho_max=rho_max, rho0=target_rho0)
    
    if plot_heatmap:
        make_heatmap(C)

    if plot_time_series:
        Z = global_zscore(X)
        idx = downsample_idx(len(d), 6000)
        t = np.arange(len(d))[idx]

        plt.figure(figsize=(11, 5))
        for i in range(min(N, 8)):
            plt.plot(t, Z[idx, i], lw=0.9, label=f"node{i}")
        plt.title("Z-scored Node Time Series (VAR Simulation)")
        plt.xlabel("Time")
        plt.ylabel("Z-score")
        plt.legend(ncol=min(N, 4), fontsize=8)
        plt.tight_layout()
        plt.show()

    return X, d, A0, alpha_eff, C, rho_trace


def simulate_var_based_on_C_with_noise__in_TVP(
    N: int,
    T: int,
    Sigma: float,
    tvp_amp: float,
    tvp_period: int,
    seed: int,
    target_rho0: float,
    alpha: float,                # requested modulation strength (pre-scaling)
    rho_max: float,
    tvp_phase_rad: float,
    A_0_is_zero: bool,
    C_base: np.ndarray,          # UN-SCALED pattern (from builder)
    safety_margin: float = 0.95, # cushion against numerical slop
    tvp_noise_std: float = 0.0,  # <--- NEW: std of additive noise on driver
    plot_rho_trace: bool = False,
    plot_heatmap: bool = False,
    plot_time_series: bool = False,
):
    """
    Simulate VAR(1) with:
        A(t) = A0 + d(t) * C,  where C = alpha_eff * C_base
    and optionally add Gaussian noise to the driver:
        d(t) = sin(2πt / period) + ε_t,  ε_t ~ N(0, tvp_noise_std²)

    Returns
    -------
    X : (T, N) simulated time series
    d : (T,) driver (possibly noisy)
    A0 : (N, N) baseline (after scaling)
    alpha_eff : float, effective modulation (after scaling)
    C : (N, N) modulation matrix (after scaling)
    rho_trace : (T,) spectral radius of A(t) per time step
    """
    set_seed(seed)

    # 1) Base driver
    d = make_driver(T, tvp_amp, tvp_period, phase=tvp_phase_rad)

    # 2) Add noise to driver (optional)
    if tvp_noise_std > 0:
        rng = np.random.default_rng(seed + 999)
        d_noise = rng.normal(0, tvp_noise_std, size=T)
        d = d + d_noise

        # Normalise to keep within [-tvp_amp, +tvp_amp]
        d = (d - np.mean(d))
        d_std = np.std(d)
        if d_std > 1e-6:
            d /= d_std
        d *= tvp_amp
        d = np.clip(d, -tvp_amp, tvp_amp)

    # 3) Baseline A0
    if A_0_is_zero:
        A0 = np.zeros((N, N))
    else:
        A0_raw = np.random.normal(0.0, 1.0, size=(N, N))
        A0 = scale_to_spectral_radius(A0_raw, target_rho0)

    # 4) Modulation matrix C
    C = alpha * C_base
    alpha_eff = alpha

    # 5) Global scaling to enforce max_t rho(A(t)) <= rho_max
    if tvp_amp > 0:
        rho_plus = spectral_radius(A0 + (+tvp_amp) * C)
        rho_minus = spectral_radius(A0 + (-tvp_amp) * C)
        rho_worst = max(rho_plus, rho_minus)
    else:
        rho_worst = spectral_radius(A0)

    s = min(1.0, (rho_max * safety_margin) / rho_worst) if rho_worst > 0 else 1.0
    A0 *= s
    C *= s
    alpha_eff *= s

    # 6) Simulate VAR
    X = np.zeros((T, N), dtype=float)
    x_prev = np.zeros(N, dtype=float)
    rho_trace = np.empty(T, dtype=float)
    L_eta = np.sqrt(Sigma) * np.eye(N)

    for t in range(T):
        At = A0 + d[t] * C
        rho_trace[t] = spectral_radius(At)
        eta_t = L_eta @ np.random.normal(size=N)
        x_t = At @ x_prev + eta_t
        X[t] = x_t
        x_prev = x_t

    # 7) Diagnostics / plots
    if plot_rho_trace:
        plot_rho_trace_figure(rho_trace=rho_trace, rho_max=rho_max, rho0=target_rho0)
    if plot_heatmap:
        make_heatmap(C)
    if plot_time_series:
        Z = global_zscore(X)
        idx = downsample_idx(len(d), 6000)
        t = np.arange(len(d))[idx]
        plt.figure(figsize=(11, 5))
        for i in range(min(N, 8)):
            plt.plot(t, Z[idx, i], lw=0.9, label=f"node{i}")
        plt.title("Z-scored Node Time Series (VAR Simulation)")
        plt.xlabel("Time")
        plt.ylabel("Z-score")
        plt.legend(ncol=min(N, 4), fontsize=8)
        plt.tight_layout()
        plt.show()

    return X, d, A0, alpha_eff, C, rho_trace

