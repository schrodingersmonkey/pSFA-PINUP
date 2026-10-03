import numpy as np


def build_edges(Z: np.ndarray):
    T, N = Z.shape
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    M = len(pairs)
    E = np.empty((T, M), float)
    for k, (i, j) in enumerate(pairs):
        E[:, k] = Z[:, i] * Z[:, j]
    return E, pairs


def build_lagged_edges(X: np.ndarray, lag: int = 1, directed: bool = True, include_auto: bool = False):
        """
        Returns E (T-lag, M) where column k is X[:, i] at time t times X[:, j] at time t-lag.
        If directed=True, includes ordered pairs (i, j). Set include_auto=True to include i==j.
        """
        T, N = X.shape
        if directed:
            pairs = [(i, j) for i in range(N) for j in range(N) if include_auto or i != j]
        else:
            # Undirected doesn't make much sense for lagged edges; fallback to upper-triangular w/out diag.
            pairs = [(i, j) for i in range(N) for j in range(i+1, N)]

        M = len(pairs)
        E = np.empty((T - lag, M), dtype=float)
        for k, (i, j) in enumerate(pairs):
            # note the shift: current t uses X[t, i] with X[t-lag, j]
            E[:, k] = X[lag:, i] * X[:-lag, j]
        return E, pairs



def build_non_linear_edges(Z: np.ndarray):
    """
    Build pairwise nonlinear edges using f(z) = sin(z) * sqrt(|z|).
    
    Args:
        Z: (T, N) node time series matrix
    
    Returns:
        E: (T, M) matrix of edge time series
        pairs: list of (i, j) node index pairs
    """
    T, N = Z.shape
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    M = len(pairs)
    E = np.empty((T, M), dtype=float)

    # Apply non-linear transform first
    #F = np.sin(Z) * np.sqrt(np.abs(Z))

    F = np.square(Z) + 100

    for k, (i, j) in enumerate(pairs):
        E[:, k] = F[:, i] * F[:, j]

    return E, pairs
