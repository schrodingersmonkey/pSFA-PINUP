import numpy as np

def affine_align(y: np.ndarray, target: np.ndarray):
    Y = np.column_stack([y, np.ones_like(y)])
    a, b = np.linalg.lstsq(Y, target, rcond=None)[0]
    return a*y + b, a, b


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    return 1.0 - ss_res/(ss_tot + 1e-12)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(); y = y - y.mean()
    return float(np.dot(x, y) / (np.linalg.norm(x)*np.linalg.norm(y) + 1e-12))


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, a.size+1, dtype=float)
    v = a[order]
    start = 0
    for i in range(1, a.size+1):
        if i == a.size or v[i] != v[start]:
            avg = (start + 1 + i)/2.0
            ranks[order[start:i]] = avg
            start = i
    return ranks

def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_r(rankdata(x), rankdata(y))


def fit_sfa_blocks(blocks: list[np.ndarray], n_components: int = 3, eps: float = 1e-9):
    """Block-aware linear SFA: signal covariance over ALL samples, but derivative
    covariance only from WITHIN-block timestep pairs, so the seam between two
    concatenated blocks (e.g. different subjects' recordings) never contributes a
    spurious 'fast' derivative."""
    F = np.vstack(blocks)
    mu = F.mean(axis=0)
    Fc = F - mu
    C = np.atleast_2d(np.cov(Fc, rowvar=False)) + eps * np.eye(Fc.shape[1])
    evals, evecs = np.linalg.eigh(C)
    evals = np.maximum(evals, eps)
    S = evecs @ np.diag(1.0 / np.sqrt(evals))
    Z = Fc @ S
    dZ_parts, idx = [], 0
    for b in blocks:
        n = b.shape[0]
        dZ_parts.append(np.diff(Z[idx:idx + n], axis=0))
        idx += n
    dZ = np.vstack(dZ_parts)
    Cd = np.atleast_2d(np.cov(dZ, rowvar=False))
    Cd = 0.5 * (Cd + Cd.T)
    dvals, dvecs = np.linalg.eigh(Cd)
    order = np.argsort(dvals)
    dvecs = dvecs[:, order]
    dvals = dvals[order]
    Wfull = S @ dvecs
    k = min(n_components, Wfull.shape[1])
    return dict(W=Wfull[:, :k].T, mu=mu, deltas=dvals[:k])


def global_fit(F_list: list[np.ndarray], n_components: int = 3):
    res = fit_sfa_blocks(F_list, n_components=n_components)
    return dict(W=res["W"], mu=res["mu"], best=0, deltas=res["deltas"])


def loso_cv(F_dict: dict, d_dict: dict, n_components: int = 3):
    """Leave-one-subject-out: fit on N-1, freeze weights, apply cold to the
    held-out subject. R2 = pearson_r**2 directly -- no affine_align (sign of
    the raw projection doesn't matter once squared)."""
    sids = list(F_dict.keys())
    out = {}
    for held in sids:
        tr_F = [F_dict[s] for s in sids if s != held]
        fit = global_fit(tr_F, n_components)
        w, mu = fit["W"][fit["best"]], fit["mu"]
        y = (F_dict[held] - mu) @ w
        r = pearson_r(y, d_dict[held])
        out[held] = r ** 2
    return out
