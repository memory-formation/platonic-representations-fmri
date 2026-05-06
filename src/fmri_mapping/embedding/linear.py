

import numpy as np
import pandas as pd


from .ops import split_betas_by_subject_index


def fit_session_residualizer(
    X_train: np.ndarray,
    C_train: np.ndarray,
    lam: float = 1e-3,
):
    """
    Fit ridge-regularized confound regression on train only.
    Returns B so that X_res = X - C @ B.
    """
    C_train = C_train.astype(np.float32)
    X_train = X_train.astype(np.float32)

    CtC = C_train.T @ C_train
    reg = lam * np.eye(C_train.shape[1], dtype=np.float32)
    B = np.linalg.solve(CtC + reg, C_train.T @ X_train)  # (n_confounds, n_features)
    return B


def apply_session_residualizer(
    X: np.ndarray, C: np.ndarray, B: np.ndarray
) -> np.ndarray:
    C = C.astype(np.float32)
    X = X.astype(np.float32)
    return X - C @ B

def corr_cols(A, B, eps=1e-8):
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    num = np.sum(A * B, axis=0)
    den = np.sqrt(np.sum(A * A, axis=0) * np.sum(B * B, axis=0) + eps)
    return num / den

def voxel_reliability_loo(
    beta1: np.ndarray,
    beta2: np.ndarray,
    beta3: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute per-voxel reliability using leave-one-out correlations.

    Parameters
    ----------0.949	0.796533	1.388
    beta1, beta2, beta3 : (n_stimuli, n_voxels)
        Betas from the three repetitions (TRAIN set only).
    eps : float
        Numerical stability constant.

    Returns
    -------
    reliability : (n_voxels,)
        Mean leave-one-out Pearson correlation per voxel.
    """

    r1 = corr_cols(beta1, (beta2 + beta3) / 2, eps=eps)
    r2 = corr_cols(beta2, (beta1 + beta3) / 2, eps=eps)
    r3 = corr_cols(beta3, (beta1 + beta2) / 2, eps=eps)

    return (r1 + r2 + r3) / 3


def _get_mcca_matrices(mcca, views=(0, 1, 2)):
    """
    Extract (W_v, mu_v) from an mvlearn MCCA object for the given views.
    Returns:
        Ws: list of (d, k) numpy arrays
        mus: list of (d,) numpy arrays (zeros if means_[v] is None)
    """
    Ws = []
    mus = []
    for v in views:
        W = np.asarray(mcca.loadings_[v])
        mu = mcca.means_[v]
        if mu is None:
            mu = np.zeros(W.shape[0], dtype=W.dtype)
        else:
            mu = np.asarray(mu, dtype=W.dtype)
        Ws.append(W)
        mus.append(mu)
    return Ws, mus


def distill_mcca_projector(
    X_train: np.ndarray,
    mcca,
    views: tuple[int, ...] = (0, 1, 2),
    target: str = "avg_embeddings",  # "avg_embeddings" | "single_view" | "avg_projector"
    view_for_single: int = 0,
    lam_ridge: float = 1e-2,
    fit_intercept: bool = True,
):
    """
    Distill MCCA into a single linear projector from X -> Z_target (ridge regression).

    Parameters
    ----------
    X_train : (n_samples, d)
        Training inputs in the SAME space as MCCA loadings (e.g., PCA space).
    mcca : fitted mvlearn.embed.MCCA
    views : which MCCA views to use.
    target : how to define the target latent:
        - "avg_embeddings": Z = mean_v ( (X - mu_v) @ W_v )   (recommended)
        - "single_view":    Z = (X - mu_view) @ W_view        (baseline)
        - "avg_projector":  Z = (X - mu_bar) @ W_bar          (explicit)
    view_for_single : used if target == "single_view"
    lam_ridge : ridge penalty (lambda)
    fit_intercept : if True, learns an intercept b so Z ≈ X W + b

    Returns
    -------
    W_star : (d, k)
    b_star : (k,)  (zeros if fit_intercept=False)
    Z_target : (n_samples, k)  (returned for diagnostics)
    """
    # --- extract matrices ---
    Ws, mus = _get_mcca_matrices(mcca, views=views)

    # --- build target Z ---
    if target == "avg_embeddings":
        Zs = [(X_train - mu[None, :]) @ W for W, mu in zip(Ws, mus)]
        Z_target = np.mean(np.stack(Zs, axis=0), axis=0)
    elif target == "single_view":
        # find index of view_for_single in `views` if present, else extract directly
        if view_for_single in views:
            idx = list(views).index(view_for_single)
            W, mu = Ws[idx], mus[idx]
        else:
            W, mu = (
                _get_mcca_matrices(mcca, views=(view_for_single,))[0][0],
                _get_mcca_matrices(mcca, views=(view_for_single,))[1][0],
            )
        Z_target = (X_train - mu[None, :]) @ W
    elif target == "avg_projector":
        W_bar = np.mean(np.stack(Ws, axis=0), axis=0)
        mu_bar = np.mean(np.stack(mus, axis=0), axis=0)
        Z_target = (X_train - mu_bar[None, :]) @ W_bar
    else:
        raise ValueError(
            "target must be one of: 'avg_embeddings', 'single_view', 'avg_projector'"
        )

    # --- ridge fit: Z ≈ X W + b ---
    X = X_train.astype(np.float64, copy=False)
    Z = Z_target.astype(np.float64, copy=False)

    if fit_intercept:
        X_mean = X.mean(axis=0, keepdims=True)
        Z_mean = Z.mean(axis=0, keepdims=True)
        Xc = X - X_mean
        Zc = Z - Z_mean
    else:
        X_mean = None
        Z_mean = None
        Xc = X
        Zc = Z

    d = Xc.shape[1]
    XtX = Xc.T @ Xc
    W_star = np.linalg.solve(XtX + lam_ridge * np.eye(d), Xc.T @ Zc)  # (d, k)

    if fit_intercept:
        b_star = (Z_mean - X_mean @ W_star).ravel()  # (k,)
    else:
        b_star = np.zeros(Z.shape[1], dtype=W_star.dtype)

    return W_star.astype(np.float32), b_star.astype(np.float32)


def apply_distilled_projector(
    X: np.ndarray, W_star: np.ndarray, b_star: np.ndarray | None = None
) -> np.ndarray:
    """
    Apply distilled projector:
        Z = X @ W_star + b_star
    """
    Z = X @ W_star
    if b_star is not None:
        Z = Z + b_star[None, :]
    return Z


def estimate_sample_weights(X_train, df_repetitions_train, train_mask, eps=1e-8):
    X1, X2, X3 = split_betas_by_subject_index(
        X_train,
        df_repetitions_train,
        inpute="zeros",  # Said zero but row are inputed by nan
    )
    sample_corrs = np.zeros(X1.shape[0])
    sample_count = np.zeros(X1.shape[0])
    for A, B in [(X1, X2), (X1, X3), (X2, X3)]:

        mask = ~np.isnan(A).any(axis=1) & ~np.isnan(B).any(axis=1)
        corrs = corr_cols(A[mask].T, B[mask].T)
        sample_corrs[mask] += corrs
        sample_count[mask] += 1

    sample_weights = sample_corrs / (sample_count + eps)
    W_weights = np.full(X_train.shape[0], np.nan)

    subject_indexes = []
    weights = []
    for i in range(3):
        mask = df_repetitions_train[f"subject_index_{i+1}"].notna().to_numpy(dtype=bool)
        subject_index = (
            df_repetitions_train[f"subject_index_{i+1}"]
            .values[mask]
            .astype(int)
            .tolist()
        )
        subject_indexes.extend(subject_index)
        weights.append(sample_weights[mask])
    weights = np.concatenate(weights)

    # We have to inpute nans (only one single repetition)
    W_weights[subject_indexes] = weights
    W_weights = W_weights[train_mask]  # Only keep train samples 
    W_weights = np.where(np.isnan(W_weights), np.nanmedian(W_weights), W_weights)

    assert len(W_weights) == train_mask.sum(), "Sample weights length must match number of samples"
    assert np.isnan(W_weights).sum() == 0, "Sample weights must not contain nans after imputation"

    return W_weights


def compute_voxel_reliability_v2(
    betas: np.ndarray,
    df_train: pd.DataFrame,
    additional_index: int = 0,
    different_sessions: bool = True,
    eps: float = 1e-8,
    query: str = "",
) -> np.ndarray:
    df = df_train.copy()
    df = df[["nsd_id", "subject_index", "session", "repetition"]]
    # merge on nsd_id=nsd_id, subject_index!=subject_index
    df = df.merge(df, on="nsd_id", suffixes=("_1", "_2"), how="inner")
    df["diff_index"] = df["subject_index_1"] - df["subject_index_2"]
    df = df.query("diff_index > 0")  # Avoid including the same pair twice
    # Not same session
    if different_sessions:
        df = df.query("session_1 != session_2")
    # Additional diff_index
    if additional_index > 0:
        df = df.query("diff_index > @additional_index")

    if query:
        df = df.query(query)

    subject_index_1 = df["subject_index_1"].values
    subject_index_2 = df["subject_index_2"].values
    betas_1 = betas[subject_index_1]
    betas_2 = betas[subject_index_2]

    r = corr_cols(betas_1, betas_2, eps=eps)
    return r

