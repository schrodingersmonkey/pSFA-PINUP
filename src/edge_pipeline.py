import numpy as np
import matplotlib.pyplot as plt
from sksfa import SFA
from sklearn.decomposition import PCA


from src.util import *
from src.sfa import *
from src.pearson_edge import *


def run_sfa_pipeline_windowed(
    X: np.ndarray,
    d: np.ndarray,
    *,
    features: str = "edges",            # "nodes" | "edges" | "nodes+lag1" | "edges_lag1"
    windowing_mode: str = "rolling",      # "block" | "rolling"
    node_win: int = 1,
    edge_win: int = 1,
    plots: bool = False,
    max_plot_edges: int = 12,
    savefig: bool = False,
    save_prefix: str = "figure"
):
    """
    Runs the SFA pipeline with node- and edge-level windowing,
    and generates LaTeX-rendered figures .
    """

    # ------------------------------------------------------
    # Matplotlib global LaTeX settings for journal quality.
    # NOTE (code_pSFA): guarded behind `plots` so that headless sweep runs do
    # not force text.usetex=True globally. Numerics are unaffected.
    # ------------------------------------------------------
    if plots:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "pgf.texsystem": "pdflatex",
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{siunitx}",
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "figure.autolayout": True,
            "lines.linewidth": 1.1,
        })

    # ------------------------------------------------------
    # 1) NODE-LEVEL WINDOWING
    # ------------------------------------------------------
    T, N = X.shape
    if d.shape[0] != T:
        raise ValueError("Length mismatch X vs d")

    if windowing_mode == "block":
        tau = max(1, int(node_win))
        if tau > 1:
            Tb = (T // tau) * tau
            Xn = X[:Tb].reshape(Tb // tau, tau, N).mean(axis=1)
            dn = d[:Tb].reshape(Tb // tau, tau).mean(axis=1)
        else:
            Xn, dn = X.copy(), d.copy()
    elif windowing_mode == "rolling":
        W = max(1, int(node_win))
        if W > 1:
            Xn = moving_average_centered_any(X, W)
            dn = moving_average_centered_any(d, W)
        else:
            Xn, dn = X.copy(), d.copy()
    else:
        raise ValueError("windowing_mode must be 'block' or 'rolling'")

    Z = Xn  # already smoothed version

    # ------------------------------------------------------
    # 2) FEATURE CONSTRUCTION
    # ------------------------------------------------------
    pairs = None
    if features == "nodes":
        F = Z
        db = dn
    elif features == "nodes+lag1":
        Zlag = np.pad(Z[:-1], ((1, 0), (0, 0)))
        F = np.hstack([Z, Zlag])
        db = dn
    elif features == "edges":
        E_raw, pairs = build_edges(Z)
        F = E_raw - E_raw.mean(axis=0, keepdims=True)
        db = dn
    elif features in ("edges_lag1", "lagged_edges"):
        E_raw, pairs = build_lagged_edges(Z, lag=1, directed=True, include_auto=False)
        F = E_raw - E_raw.mean(axis=0, keepdims=True)
        db = dn[1:]
    else:
        raise ValueError("Invalid feature type.")

    # ------------------------------------------------------
    # 3) EDGE/FEATURE-LEVEL WINDOWING
    # ------------------------------------------------------
    if windowing_mode == "block":
        tau_e = max(1, int(edge_win))
        if tau_e > 1:
            Te = (F.shape[0] // tau_e) * tau_e
            F  = F[:Te].reshape(Te // tau_e, tau_e, F.shape[1]).mean(axis=1)
            db = db[:Te].reshape(Te // tau_e, tau_e).mean(axis=1)
        d_aligned = db
    else:
        W_e = max(1, int(edge_win))
        if W_e > 1:
            F  = moving_average_centered_any(F, W_e)
            db = moving_average_centered_any(db, W_e)
        d_aligned = db

    # ------------------------------------------------------
    # 4) SFA + ALIGNMENT
    # ------------------------------------------------------
    sfa = SFA(n_components=1)
    y_sfa = sfa.fit_transform(F)[:, 0]
    y_fit, a, b = affine_align(y_sfa, d_aligned)
    R2 = r2_score(d_aligned, y_fit)
    rP = pearson_r(d_aligned, y_fit)
    rS = spearman_rho(d_aligned, y_fit)

    # ------------------------------------------------------
    # 5) VISUALISATION (LaTeX rendered, publication ready)
    # ------------------------------------------------------
    if plots:
        idx = downsample_idx(len(d_aligned), 6000)
        t = np.arange(len(d_aligned))[idx]

        # --- Nodes ---
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(min(N, 8)):
            ax.plot(t, Z[idx, i], lw=1.0, label=fr"$x_{{{i}}}(t)$")
        ax.set_title(r"\textbf{Z-scored Node Time Series}")
        ax.set_xlabel(r"\textit{Time (samples)}")
        ax.set_ylabel(r"\textit{Amplitude}")
        ax.legend(ncol=min(N, 4))
        if savefig: fig.savefig(f"{save_prefix}_nodes.pdf", bbox_inches="tight")

        # --- Edges ---
        if features.startswith("edges") and pairs is not None:
            K_total = F.shape[1]
            K_edges = min(max_plot_edges, K_total)
            fig, ax = plt.subplots(figsize=(10, 4))
            for k in range(K_edges):
                i, j = pairs[k]
                ax.plot(t, F[idx, k], lw=0.8, label=fr"$x_{{{i}}}x_{{{j}}}$")
            if K_edges <= 12:
                ax.legend(ncol=3)
            ax.set_title(fr"\textbf{{Edge Time Series}} "
                         fr"({K_edges}/{K_total})")
            ax.set_xlabel(r"\textit{Time (samples)}")
            ax.set_ylabel(r"\textit{Edge value}")
            if savefig: fig.savefig(f"{save_prefix}_edges.pdf", bbox_inches="tight")

        # --- Unaligned comparison (twin axes) ---
        tt = np.arange(len(d_aligned))
        step = max(1, len(d_aligned) // 6000)
        fig_raw, ax_raw1 = plt.subplots(figsize=(10, 4))
        ax_raw2 = ax_raw1.twinx()
        ax_raw1.plot(tt[::step], d_aligned[::step], color="steelblue", lw=1.5, label=r"Driver $d(t)$")
        ax_raw2.plot(tt[::step], y_sfa[::step],     color="orange",    lw=1.2, label=r"SFA raw $y(t)$")
        ax_raw1.set_xlabel(r"\textit{Time (samples)}")
        ax_raw1.set_ylabel(r"Driver $d(t)$", color="steelblue")
        ax_raw2.set_ylabel(r"SFA output (unit variance)", color="orange")
        lines1, labs1 = ax_raw1.get_legend_handles_labels()
        lines2, labs2 = ax_raw2.get_legend_handles_labels()
        ax_raw1.legend(lines1 + lines2, labs1 + labs2)
        ax_raw1.set_title(r"\textbf{Unaligned: driver vs SFA raw output}")
        if savefig: fig_raw.savefig(f"{save_prefix}_unaligned.pdf", bbox_inches="tight")

        # --- Driver and SFA output (affine aligned) ---
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(tt[::step], d_aligned[::step], lw=1.5, label=r"Driver $d(t)$")
        ax.plot(tt[::step], y_fit[::step], lw=1.2, label=r"SFA output $\hat{d}(t)$")
        ax.set_title(
            fr"{windowing_mode} | $W_n={node_win}$, $W_e={edge_win}$ | "
            fr"$R^2={R2:.3f}$, $r={rP:.3f}$, $\rho={rS:.3f}$"
        )
        ax.set_xlabel(r"\textit{Time (samples)}")
        ax.legend()
        if savefig: fig.savefig(f"{save_prefix}_alignment.pdf", bbox_inches="tight")

    return dict(
        y_sfa=y_sfa, y_fit=y_fit, d_aligned=d_aligned,
        R2=R2, rP=rP, rS=rS,
        windowing_mode=windowing_mode, node_win=node_win, edge_win=edge_win,
        features=features
    )


def run_sfa_pipeline_edge_weights(
    X: np.ndarray,
    d: np.ndarray,
    *,
    features: str = "edges",
    windowing_mode: str = "block",
    node_win: int = 1,
    edge_win: int = 1,
    n_components: int = 3,
    node_labels: list | None = None,
    plots: bool = False,
    savefig: bool = False,
    save_prefix: str = "figure",
    pca_weights: bool = False,
) -> dict:
    """
    Same pipeline as `run_sfa_pipeline_windowed`, but also recovers *which*
    input features (edges, if features='edges') actually drive the slowest
    component, instead of only returning the resulting scalar time series.

    Answers "which specific node-pairs is p-SFA actually using?". Use
    it to build an N x N edge-importance heatmap (via `W[0, :]` 

    Mechanism: fits sksfa.SFA with fill_mode='zero' (required for
    `sfa.affine_parameters()` to work), which returns W (k, M) and b (k,)
    such that `F @ W.T + b` is exactly `sfa.transform(F)` — i.e. the actual
    linear map from input features to each SFA component, in the ORIGINAL
    (unwhitened) feature space. `W[0, :]` is the weight vector for the
    slowest component.

    `n_components` only controls how many weight vectors/components `W` and
    `y_sfa_all` contain (verified: component 0 of each is bit-identical for
    n_components=1 vs. 3 vs. 5 — SFA ranks every component by its own
    slowness objective independent of k, same as in `run_sfa_pipeline_windowed`).
    `delta_vals` is always sksfa's full slowness-eigenvalue spectrum regardless
    of `n_components`, so it's always available to check that component 0 is
    a clear winner over the rest rather than a near-tie. It is NOT a search:
    scoring (R2/rP/rS) is always computed for component 0 only — there is no
    post-hoc "pick whichever component correlates best with the driver" step,
    since that would be cherry-picking after seeing the answer. The same
    applies to the optional PCA comparison below: PC 0 (top variance) is
    scored, not whichever PC happens to correlate best.

    If pca_weights=True, PCA is also fit on the same feature matrix F, so
    `pca_V[0, :]` (PCA's top-variance loading) can be compared directly
    against `W[0, :]` (SFA's slowest-component loading) — the same feature
    space, ranked by two different criteria.

    Returns
    -------
    y_sfa_all       : (T'', k) raw SFA outputs for all k fitted components
    W               : (k, M) weight matrix — W[i, :] = weights for component i
    b               : (k,) bias vector from affine_parameters()
    pairs           : list[(i, j)] or None
    d_aligned       : (T'',) driver after windowing
    delta_vals      : (k,) slowness eigenvalues for all k components (diagnostic only)
    components      : single-entry list [{component: 0, y_fit, R2, rP, rS}] — the slowest component
    best_component  : always 0 (kept for return-shape stability; nothing is "best-of-k" here)
    N               : number of nodes
    pca_V           : (k, M) PCA loadings — V[i, :] = loadings for PC i  (only if pca_weights=True)
    pca_components  : single-entry list, same schema as `components`     (only if pca_weights=True)
    pca_best        : always 0 if pca_weights else None                  (only if pca_weights=True)
    """
    T, N = X.shape
    if d.shape[0] != T:
        raise ValueError("Length mismatch X vs d")

    # ── 1) Node windowing ─────────────────────────────────────────────────
    if windowing_mode == "block":
        tau = max(1, int(node_win))
        if tau > 1:
            Tb = (T // tau) * tau
            Xn = X[:Tb].reshape(Tb // tau, tau, N).mean(axis=1)
            dn = d[:Tb].reshape(Tb // tau, tau).mean(axis=1)
        else:
            Xn, dn = X.copy(), d.copy()
    elif windowing_mode == "rolling":
        Wn = max(1, int(node_win))
        if Wn > 1:
            Xn = moving_average_centered_any(X, Wn)
            dn = moving_average_centered_any(d, Wn)
        else:
            Xn, dn = X.copy(), d.copy()
    else:
        raise ValueError("windowing_mode must be 'block' or 'rolling'")

    Z = Xn

    # ── 2) Feature construction ───────────────────────────────────────────
    pairs = None
    if features == "nodes":
        F, db = Z, dn
    elif features == "nodes+lag1":
        Zlag = np.pad(Z[:-1], ((1, 0), (0, 0)))
        F, db = np.hstack([Z, Zlag]), dn
    elif features == "edges":
        E_raw, pairs = build_edges(Z)
        F = E_raw - E_raw.mean(axis=0, keepdims=True)
        db = dn
    elif features in ("edges_lag1", "lagged_edges"):
        E_raw, pairs = build_lagged_edges(Z, lag=1, directed=True, include_auto=False)
        F = E_raw - E_raw.mean(axis=0, keepdims=True)
        db = dn[1:]
    else:
        raise ValueError("Invalid feature type.")

    # ── 3) Edge/feature windowing ─────────────────────────────────────────
    if windowing_mode == "block":
        tau_e = max(1, int(edge_win))
        if tau_e > 1:
            Te = (F.shape[0] // tau_e) * tau_e
            F  = F[:Te].reshape(Te // tau_e, tau_e, F.shape[1]).mean(axis=1)
            db = db[:Te].reshape(Te // tau_e, tau_e).mean(axis=1)
        d_aligned = db
    else:
        We = max(1, int(edge_win))
        if We > 1:
            F  = moving_average_centered_any(F, We)
            db = moving_average_centered_any(db, We)
        d_aligned = db

    # ── 4) SFA via sksfa with weight extraction via affine_parameters() ───
    # fill_mode="zero" is required for affine_parameters() to work.
    # Output is bit-for-bit identical to SFA(fill_mode="noise") when there
    # are no trivial (near-zero-variance) features, which is always the case
    # for well-conditioned edge matrices.
    k = min(n_components, F.shape[1])
    sfa = SFA(n_components=k, fill_mode="zero")
    y_sfa_all = sfa.fit_transform(F)          # (T'', k), slowest component first
    W_mat, b_mat = sfa.affine_parameters()    # W_mat: (k, M),  b_mat: (k,)
    # Verify: np.allclose(F @ W_mat.T + b_mat, y_sfa_all) should be True

    # ── 5) Alignment metrics for the slowest component only ────────────────
    # Component 0 is always the slowest direction regardless of k (see
    # docstring), so there is nothing to search for: no "best of k" selection.
    y_fit, _, _ = affine_align(y_sfa_all[:, 0], d_aligned)
    best = 0
    components = [dict(
        component=0,
        y_fit=y_fit,
        R2=r2_score(d_aligned, y_fit),
        rP=pearson_r(d_aligned, y_fit),
        rS=spearman_rho(d_aligned, y_fit),
    )]

    # ── 6) PCA weights (optional) ─────────────────────────────────────────
    pca_V = None
    pca_components_list = None
    pca_best = None
    if pca_weights:
        pca = PCA(n_components=k)
        y_pca_all = pca.fit_transform(F)          # (T'', k), most-variance first
        pca_V = pca.components_                   # (k, M), V[i, :] = loadings for PC i

        # PC 0 (top variance) is scored, not whichever PC correlates best.
        pca_y_fit, _, _ = affine_align(y_pca_all[:, 0], d_aligned)
        pca_best = 0
        pca_components_list = [dict(
            component=0,
            y_fit=pca_y_fit,
            R2=r2_score(d_aligned, pca_y_fit),
            rP=pearson_r(d_aligned, pca_y_fit),
            rS=spearman_rho(d_aligned, pca_y_fit),
        )]

    # ── 7) Plots ──────────────────────────────────────────────────────────
    if plots:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 150,
            "figure.autolayout": True,
        })

        labs = node_labels or [str(i) for i in range(N)]

        # (a) N×N weight heatmap for slowest component
        if features == "edges" and pairs is not None:
            w0 = np.abs(W_mat[0, :])             # slowest component weights
            W_grid = np.zeros((N, N))
            for m, (i, j) in enumerate(pairs):
                W_grid[i, j] = w0[m]
                W_grid[j, i] = w0[m]

            fig, ax = plt.subplots(figsize=(max(5, N * 0.55), max(4, N * 0.5)))
            im = ax.imshow(W_grid, cmap="YlOrRd")
            plt.colorbar(im, ax=ax, label=r"$|w_k|$")
            ax.set_xticks(range(N))
            ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=9)
            ax.set_yticks(range(N))
            ax.set_yticklabels(labs, fontsize=9)
            ax.set_title(r"\textbf{Edge SFA weights} -- slowest component $|w_k|$")
            if savefig:
                fig.savefig(f"{save_prefix}_weight_heatmap.pdf", bbox_inches="tight")

        # (b) Top-20 edges sorted by |w| — bar chart
        if pairs is not None:
            w0 = np.abs(W_mat[0, :])
            top_n = min(20, len(pairs))
            sort_idx = np.argsort(w0)[::-1][:top_n]
            edge_labs = [f"{labs[pairs[m][0]]} x {labs[pairs[m][1]]}" for m in sort_idx]

            fig, ax = plt.subplots(figsize=(max(8, top_n * 0.6), 4))
            ax.bar(range(top_n), w0[sort_idx], color="steelblue")
            ax.set_xticks(range(top_n))
            ax.set_xticklabels(edge_labs, rotation=45, ha="right")
            ax.set_ylabel(r"$|w_k|$ (SFA weight)")
            ax.set_title(r"\textbf{Top edges by SFA weight} -- slowest component")
            if savefig:
                fig.savefig(f"{save_prefix}_weight_bar.pdf", bbox_inches="tight")

        # (c) Slowest-component alignment vs driver
        c = components[best]
        tt = np.arange(len(d_aligned))
        step = max(1, len(d_aligned) // 6000)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(tt[::step], d_aligned[::step], lw=1.5, label=r"Driver $d(t)$")
        ax.plot(tt[::step], c["y_fit"][::step], lw=1.2,
                label=fr"Component {best} $\hat{{d}}(t)$")
        ax.set_title(
            fr"\textbf{{Component {best}}} $|$ "
            fr"$R^2={c['R2']:.3f}$, $r={c['rP']:.3f}$, $\rho={c['rS']:.3f}$"
        )
        ax.set_xlabel(r"\textit{Time (samples)}")
        ax.legend()
        if savefig:
            fig.savefig(f"{save_prefix}_alignment.pdf", bbox_inches="tight")

        # (d) PCA plots — mirror of (a) and (b) using pca_V
        if pca_weights and pca_V is not None and pairs is not None:
            v0 = np.abs(pca_V[0, :])             # PC1 loadings

            if features == "edges":
                V_grid = np.zeros((N, N))
                for m, (i, j) in enumerate(pairs):
                    V_grid[i, j] = v0[m]
                    V_grid[j, i] = v0[m]

                fig, ax = plt.subplots(figsize=(max(5, N * 0.55), max(4, N * 0.5)))
                im = ax.imshow(V_grid, cmap="YlOrRd")
                plt.colorbar(im, ax=ax, label=r"$|v_k|$")
                ax.set_xticks(range(N))
                ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=9)
                ax.set_yticks(range(N))
                ax.set_yticklabels(labs, fontsize=9)
                ax.set_title(r"\textbf{Edge PCA loadings} -- PC1 $|v_k|$")
                if savefig:
                    fig.savefig(f"{save_prefix}_pca_weight_heatmap.pdf", bbox_inches="tight")

            top_n = min(20, len(pairs))
            sort_idx = np.argsort(v0)[::-1][:top_n]
            edge_labs = [f"{labs[pairs[m][0]]} x {labs[pairs[m][1]]}" for m in sort_idx]

            fig, ax = plt.subplots(figsize=(max(8, top_n * 0.6), 4))
            ax.bar(range(top_n), v0[sort_idx], color="darkorange")
            ax.set_xticks(range(top_n))
            ax.set_xticklabels(edge_labs, rotation=45, ha="right")
            ax.set_ylabel(r"$|v_k|$ (PCA loading)")
            ax.set_title(r"\textbf{Top edges by PCA loading} -- PC1")
            if savefig:
                fig.savefig(f"{save_prefix}_pca_weight_bar.pdf", bbox_inches="tight")

            cp = pca_components_list[pca_best]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(tt[::step], d_aligned[::step], lw=1.5, label=r"Driver $d(t)$")
            ax.plot(tt[::step], cp["y_fit"][::step], lw=1.2,
                    label=fr"PC {pca_best} $\hat{{d}}(t)$")
            ax.set_title(
                fr"\textbf{{PC {pca_best}}} $|$ "
                fr"$R^2={cp['R2']:.3f}$, $r={cp['rP']:.3f}$, $\rho={cp['rS']:.3f}$"
            )
            ax.set_xlabel(r"\textit{Time (samples)}")
            ax.legend()
            if savefig:
                fig.savefig(f"{save_prefix}_pca_alignment.pdf", bbox_inches="tight")

    return dict(
        y_sfa_all=y_sfa_all,
        W=W_mat,
        b=b_mat,
        pairs=pairs,
        d_aligned=d_aligned,
        delta_vals=sfa.delta_values_,
        components=components,
        best_component=best,
        N=N,
        pca_V=pca_V,
        pca_components=pca_components_list,
        pca_best=pca_best,
    )
