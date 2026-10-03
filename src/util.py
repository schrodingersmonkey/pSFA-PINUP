import os
import time
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def set_seed(seed: int):
    """Seed NumPy's global RNG so simulations are reproducible."""
    np.random.seed(seed)


def make_driver(T: int, amp: float, period: int, phase: float = 60.0) -> np.ndarray:
    """Build a sinusoidal driver signal: amp * sin(2*pi*t/period + phase)."""
    t = np.arange(T, dtype=float)
    return amp * np.sin(2.0 * np.pi * t / float(period) + phase)

def orthonormal_columns(n: int, r: int) -> np.ndarray:
    """Draw r random mutually orthonormal columns in R^n (via QR)."""
    A = np.random.normal(size=(n, r))
    Q, _ = np.linalg.qr(A)
    return Q[:, :r]

def spectral_radius(A: np.ndarray) -> float:
    """Largest absolute eigenvalue of A — its spectral radius."""
    return float(np.max(np.abs(np.linalg.eigvals(A))))

def scale_to_spectral_radius(A: np.ndarray, target_rho: float) -> np.ndarray:
    """Rescale A so its spectral radius equals target_rho."""
    rho = spectral_radius(A)
    if rho == 0:
        return A
    return (target_rho / rho) * A

def global_zscore(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Z-score each column of X using its mean/std over the full series."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd


def mk_out_dir(out_root: str | None) -> str:
    """Create and return an output directory, defaulting to a timestamped 'runs/' folder."""
    if out_root is None or out_root.strip() == "":
        out_root = os.path.join("runs", time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_root, exist_ok=True)
    return out_root


def downsample_idx(T: int, max_points: int = 10000) -> np.ndarray:
    """Evenly spaced indices into range(T), capped at roughly max_points."""
    step = max(1, T // max_points)
    return np.arange(T)[::step]


def make_heatmap(C):
    """Plot matrix C as an annotated heatmap, with each value labelled in its cell."""
    C = np.array(C)

    # If C is 1D, reshape to (N,1) for consistent display
    if C.ndim == 1:
        C = C[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(C, cmap=cm.Greens)
    ax.set_axis_off()

    # annotate each cell
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            val = f"{C[i, j]:.4g}"
            ax.text(j, i, val, ha="center", va="center")

    fig.tight_layout()
    plt.show()

def plot_rho_trace_figure(rho_trace, rho_max, rho0):
    """Plot the spectral radius trajectory rho(A(t)) against its stability bounds."""
    plt.figure(figsize=(10,4))
    plt.plot(rho_trace, lw=1.5, label="ρ(A(t)) over time")
    plt.axhline(1.0, color="red", ls="--", lw=1, label="stability boundary (ρ=1)")
    plt.axhline(rho_max, color="green", ls="--", lw=1, label="rho max cap")
    plt.axhline(rho0, color="orange", ls="--", lw=1, label="Initial max rho")
    plt.xlabel("time step")
    plt.ylabel("spectral radius ρ(A(t))")
    plt.title("Spectral radius trajectory during simulation")
    plt.legend()
    plt.tight_layout()
    plt.show()


def moving_average_centered_any(Y, W):
    """Centered moving average over W samples; accepts 1-D or 2-D input and forces W odd."""
    import numpy as np
    if W % 2 == 0:
        W += 1
    Y = np.asarray(Y)
    if Y.ndim == 1:
        return moving_average_centered(Y[:, None], W)[:, 0]
    elif Y.ndim == 2:
        return moving_average_centered(Y, W)
    else:
        raise ValueError("Y must be 1D or 2D")
    
def moving_average_centered(X: np.ndarray, W: int) -> np.ndarray:
    """Centered moving average over an odd window W, applied independently per column."""
    if W <= 1:
        return X
    if W % 2 == 0:
        raise ValueError("Smoothing window W must be odd for zero-phase centering.")
    T, M = X.shape
    h = np.ones(W, float) / W #each element in kernel = 1/w
    Y = np.empty((T - W + 1, M), float)
    for m in range(M):
        Y[:, m] = np.convolve(X[:, m], h, mode="valid")
    return Y

