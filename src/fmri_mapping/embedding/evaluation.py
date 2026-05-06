import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from .ops import split_repetitions, split_betas_by_subject_index


def accuracy_cosine_similarity(
    x_true: torch.Tensor,
    x_pred: torch.Tensor,
    return_similarity_matrix: bool = False,
):
    """
    Returns:
        accuracy: top-1 retrieval accuracy
        cosine: mean diagonal cosine similarity
        mean_rank: mean rank of the correct match (1 = best)
        similarities (optional): full similarity matrix
    """

    x_true_norm = torch.nn.functional.normalize(x_true, dim=1)
    x_pred_norm = torch.nn.functional.normalize(x_pred, dim=1)

    # Mean cosine similarity (mean diagonal)
    cosine = torch.sum(x_true_norm * x_pred_norm, dim=1).mean().item()

    # Similarity matrix
    similarities = x_pred_norm @ x_true_norm.T  # (n, n)

    n = similarities.shape[0]
    device = similarities.device
    target = torch.arange(n, device=device)

    # Top-1 accuracy
    top1 = torch.argmax(similarities, dim=1)
    accuracy = (top1 == target).float().mean().item()

    # ---- Mean rank ----
    # Sort similarities in descending order
    sorted_indices = torch.argsort(similarities, dim=1, descending=True)

    # Rank of the correct index (1-based)
    ranks = (sorted_indices == target[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    mean_rank = ranks.float().mean().item()

    if return_similarity_matrix:
        return accuracy, cosine, mean_rank, similarities

    return accuracy, cosine, mean_rank


def compute_recall_curves(view_x, view_y, full_curves: bool = True):
    view_x = torch.tensor(view_x, device="cuda")
    view_y = torch.tensor(view_y, device="cuda")
    n_components = view_x.shape[1]
    assert view_y.shape[1] == n_components

    metrics = []
    start = 1 if full_curves else n_components

    for k in range(start, n_components + 1):
        accuracy, cosine, mean_rank = accuracy_cosine_similarity(
            view_x[:, :k], view_y[:, :k]
        )
        metrics.append(
            {
                "n_components": k,
                "accuracy": accuracy,
                "cosine": cosine,
                "mean_rank": mean_rank,
            }
        )

    del view_x, view_y
    torch.cuda.empty_cache()
    return pd.DataFrame(metrics)


def plot_metric_curves(metrics: pd.DataFrame, hue: str = None, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    (ax1, ax2, ax3) = axes

    if title:
        fig.suptitle(title)

    sns.lineplot(data=metrics, x="n_components", y="accuracy", hue=hue, ax=ax1)
    ax1.set_title(f"Retrieval Accuracy vs Number of MCCA Components - V1")
    ax1.set_ylabel("Accuracy")

    sns.lineplot(data=metrics, x="n_components", y="cosine", hue=hue, ax=ax2)
    ax2.set_title("Mean Cosine Similarity")
    ax2.set_ylabel("Cosine Similarity")

    sns.lineplot(data=metrics, x="n_components", y="mean_rank", hue=hue, ax=ax3)
    ax3.set_title("Mean Rank")
    ax3.set_ylabel("Mean Rank")

    for ax in (ax1, ax2, ax3):
        sns.despine(ax=ax)
        ax.set_xlabel("Number of MCCA Components")
        ax.set_xlim(1, metrics["n_components"].max())

    return fig, axes


def get_metrics_combs(X1_view, X2_view, X3_view, full_curves: bool = True, **kwargs):
    metrics = []
    for comb in [
        ("1-2", X1_view, X2_view),
        ("1-3", X1_view, X3_view),
        ("2-3", X2_view, X3_view),
    ]:
        comb_name, X1_v, X2_v = comb

        # Skip those pairs where one view is empty with nans
        X1_v_nan = np.isnan(X1_v).any(axis=1)  # Check if any row is all NaNs n_trials
        X2_v_nan = np.isnan(X2_v).any(axis=1)  # Check if any row is all NaNs n_trials

        sample_mask = ~(
            X1_v_nan | X2_v_nan
        )  # mask of samples where neither view is all NaNs

        metric = compute_recall_curves(
            X1_v[sample_mask, :], X2_v[sample_mask, :], full_curves=full_curves
        )
        n_samples = sample_mask.sum()
        metric["normalized_mean_rank"] = (metric["mean_rank"] - 1) / (n_samples - 1)

        metric["combination"] = comb_name
        metric["n_samples"] = n_samples
        for arg in kwargs:
            metric[arg] = kwargs[arg]

        metrics.append(metric)
    return pd.concat(metrics, axis=0)


def compute_metric_curves(
    Z: np.ndarray,
    subject: int,
    shuffle_indexes: bool = False,
    inpute="zeros",
    full_curves: bool = True,
    **kwargs,
):
    df_repetitions = split_repetitions(subject=subject, shuffle_indexes=shuffle_indexes)
    df_repetitions_train = df_repetitions.query("not shared")
    df_repetitions_test = df_repetitions.query("shared")

    metrics = []
    for subset, df_subset in [
        ("train", df_repetitions_train),
        ("test", df_repetitions_test),
    ]:
        if len(df_subset) == 0:
            continue
        Z1, Z2, Z3 = split_betas_by_subject_index(Z, df_subset, inpute=inpute)
        metric = get_metrics_combs(
            Z1,
            Z2,
            Z3,
            subject=subject,
            subset=subset,
            full_curves=full_curves,
            **kwargs,
        )
        metrics.append(metric)

    df_metric = pd.concat(metrics)
    return df_metric


def avg_metric_combinations(
    df_metric: pd.DataFrame,
    avg_cols=["mean_rank", "accuracy", "cosine", "normalized_mean_rank", "n_samples"],
    agg_cols=["combination"],
    sort_cols=["mean_rank"],
) -> pd.DataFrame:

    agg = {col: "mean" for col in avg_cols}
    columns = df_metric.columns.tolist()
    groupby = [col for col in columns if col not in avg_cols and col not in agg_cols]

    return (
        df_metric.groupby(groupby)
        .aggregate(agg)
        .reset_index()
        .sort_values(sort_cols)
        .reset_index(drop=True)
    )


from torch import nn


@torch.no_grad()
def apply_batched_model(
    model: nn.Module, Z: torch.Tensor, batch_size: int = 4096
) -> torch.Tensor:
    model.eval()
    outs = []
    for i in range(0, Z.shape[0], batch_size):
        outs.append(model(Z[i : i + batch_size]))
    return torch.cat(outs, dim=0)




@torch.no_grad()
def compute_rsa(Z_i: torch.tensor, Z_j: torch.tensor, first_metric: str = "pearson", second_metric: str = "pearson") -> dict:

    # Compute RDMs
    if first_metric == "pearson":
        rdm_i = 1 - torch.corrcoef(Z_i)
        rdm_j = 1 - torch.corrcoef(Z_j)
    elif first_metric == "euclidean":
        rdm_i = torch.cdist(Z_i, Z_i)
        rdm_j = torch.cdist(Z_j, Z_j)
    elif first_metric == "cosine":
        rdm_i = 1 - torch.nn.functional.cosine_similarity(Z_i.unsqueeze(1), Z_i.unsqueeze(0), dim=-1)
        rdm_j = 1 - torch.nn.functional.cosine_similarity(Z_j.unsqueeze(1), Z_j.unsqueeze(0), dim=-1)

    # Obtain the upper triangular part of the RDMs

    rows, cols = torch.triu_indices(rdm_i.shape[0], rdm_i.shape[1], offset=1, device=rdm_i.device)
    rdm_i_upper = rdm_i[rows, cols]
    rdm_j_upper = rdm_j[rows, cols]

    
    # Compute the correlation between the upper triangular parts of the RDMs
    if second_metric == "pearson":
        rsa = torch.corrcoef(torch.stack([rdm_i_upper, rdm_j_upper]))[0, 1].item()
    elif second_metric == "spearman":
        rdm_i_upper = rdm_i_upper.cpu().numpy()
        rdm_j_upper = rdm_j_upper.cpu().numpy()
        rsa = pd.Series(rdm_i_upper).corr(pd.Series(rdm_j_upper), method="spearman")
    elif second_metric == "euclidean":
        rsa = torch.norm(rdm_i_upper - rdm_j_upper).item()
    elif second_metric == "cosine":
        rsa = torch.nn.functional.cosine_similarity(rdm_i_upper, rdm_j_upper, dim=0).item()

    return rsa
