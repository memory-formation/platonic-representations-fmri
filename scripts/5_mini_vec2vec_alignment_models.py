import torch
import numpy as np
import argparse
from pathlib import Path
import numpy as np
from tqdm import trange, tqdm
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import quadratic_assignment
from sklearn.cluster import KMeans
import os
import torch
import torch.nn.functional as F
from typing import Literal
from fmri_mapping.embedding.evaluation import accuracy_cosine_similarity, compute_rsa
from sklearn.decomposition import PCA

# from fmri_mapping.embedding.spherical_kmeans import SphericalKMeans

from dmf.alerts import alert, send_message


def cos_sim_matrix(X, Y):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y)
    X_norm = X / X.norm(dim=-1, keepdim=True)
    Y_norm = Y / Y.norm(dim=-1, keepdim=True)
    return X_norm @ Y_norm.T


import torch
import torch.nn.functional as F


def get_csls_sim(src_emb: torch.Tensor, tgt_emb: torch.Tensor, k: int = 10):
    """
    Compute Cross-domain Similarity Local Scaling (CSLS) similarity matrix.

    Parameters
    ----------
    src_emb : torch.Tensor
        Source embeddings of shape (N_src, D)
    tgt_emb : torch.Tensor
        Target embeddings of shape (N_tgt, D)
    k : int
        Number of nearest neighbors used to estimate local density

    Returns
    -------
    csls_sim : torch.Tensor
        CSLS similarity matrix of shape (N_src, N_tgt)

    Notes
    -----
    CSLS is:
        2 * cos(x, y) - r_src(x) - r_tgt(y)

    where:
        r_src(x) = average cosine similarity of x to its top-k neighbors in target
        r_tgt(y) = average cosine similarity of y to its top-k neighbors in source
    """
    # Normalize embeddings so dot product = cosine similarity
    src_norm = F.normalize(src_emb, dim=1)
    tgt_norm = F.normalize(tgt_emb, dim=1)

    # Cosine similarity matrix
    cosine_sim = src_norm @ tgt_norm.T  # (N_src, N_tgt)

    # Source-side local scaling:
    # for each source point, mean similarity to its top-k target neighbors
    k_src = min(k, cosine_sim.shape[1])
    sim_src_k, _ = torch.topk(cosine_sim, k=k_src, dim=1)
    r_src = sim_src_k.mean(dim=1)  # (N_src,)

    # Target-side local scaling:
    # for each target point, mean similarity to its top-k source neighbors
    k_tgt = min(k, cosine_sim.shape[0])
    sim_tgt_k, _ = torch.topk(cosine_sim, k=k_tgt, dim=0)
    r_tgt = sim_tgt_k.mean(dim=0)  # (N_tgt,)

    # CSLS correction
    csls_sim = 2 * cosine_sim - r_src.unsqueeze(1) - r_tgt.unsqueeze(0)

    return csls_sim


def tensor(x):
    return torch.tensor(x).float()


def N(X, dim=-1, **kwargs):
    return F.normalize(X, dim=dim, **kwargs)


def sim(X, Y, center: bool = True):
    X, Y = tensor(X), tensor(Y)
    if center:
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
    return X @ Y.T


def train_orthogonal_linear(X, Y):
    solution, _ = orthogonal_procrustes(X, Y)
    return tensor(solution)


def eval_score(X_eval, Y_eval, W, backward=False):
    if backward:
        return torch.cosine_similarity(X_eval, Y_eval @ W.T, dim=-1).mean()

    return torch.cosine_similarity(X_eval @ W, Y_eval, dim=-1).mean()


def aligned_centroids(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    n_runs: int = 50,
    n_clusters: int = 50,
    method: Literal["2opt", "faq"] = "2opt",
    subsample: int = None,
    center_kernels: bool = True,
):
    """Align cluster centroids between two datasets using quadratic assignment.
    This function performs K-means clustering on two input tensors and then finds
    an optimal alignment between the resulting cluster centroids using quadratic
    assignment problem (QAP) solvers.
    Args:
        X_train (torch.Tensor): Input data tensor of shape (n_samples, n_features).
        Y_train (torch.Tensor): Target data tensor of shape (n_samples, n_features).
        n_runs (int, optional): Number of times to run QAP solver to find better
            solutions. Defaults to 300.
        n_clusters (int, optional): Number of clusters for K-means. Defaults to 50.
        method (Literal["2opt", "faq"], optional): QAP solver method to use.
            "2opt" for 2-opt heuristic or "faq" for FAQ algorithm. Defaults to "2opt".
        subsample (int, optional): If specified, randomly subsample both input tensors
            to this size before clustering. Defaults to None (use all data).
        center_kernels (bool, optional): Whether to center the kernel matrices before
            QAP. Defaults to True.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of (centers1, centers2) where:
            - centers1: Cluster centroids from X_train (shape: n_clusters, n_features)
            - centers2: Cluster centroids from Y_train, aligned to centers1
              (shape: n_clusters, n_features)
    Notes:
        - The function runs QAP multiple times and keeps the solution with the
          highest objective value (maximization).
        - Kernel matrices are computed as similarity matrices between centroids.
        - The alignment permutation is applied to centers2 to match centers1.
    """
    options = {"maximize": True}  # , 'P0': 'randomized'}
    if subsample is not None:
        X_train, Y_train = (
            X_train[torch.randperm(len(X_train))[:subsample]],
            Y_train[torch.randperm(len(Y_train))[:subsample]],
        )

    clusterer1 = KMeans(n_clusters=n_clusters)
    # clusterer1 = SphericalKMeans(n_clusters=n_clusters, n_jobs=-1)
    clusterer1.fit(X_train)
    clusterer2 = KMeans(n_clusters=n_clusters)
    # clusterer2 = SphericalKMeans(n_clusters=n_clusters)
    clusterer2.fit(Y_train)
    centers1, centers2 = clusterer1.cluster_centers_, clusterer2.cluster_centers_
    kernel1 = sim(centers1, centers1, center=center_kernels).float()
    kernel2 = sim(centers2, centers2, center=center_kernels).float()

    quad = None
    # need to re-run the QAP a few times because it's not very good at finding the global optimum
    for i in trange(n_runs, leave=False):
        new_quad = quadratic_assignment(
            kernel1, kernel2, method=method, options=options
        )
        if quad is None or quad.fun < new_quad.fun:
            quad = new_quad
    centers2 = centers2[quad.col_ind]
    return tensor(centers1), tensor(centers2)


def alpha_schedule(i, initial=0.5, minimun=0.1, start=2000, end=4000) -> float:
    # From 0 to start -> initial value
    # From start to end -> linear decay to minimum
    # From end onwards -> minimum value
    if i < start:
        return initial
    elif i < end:
        return initial - (initial - minimun) * ((i - start) / (end - start))
    else:
        return minimun


def process_subjects(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor = None,
    Y_test: torch.Tensor = None,
    n_runs_aligned_centroids: int = 50,
    top_k: int = 32,
    n_clusters: int = 64,
    n_runs: int = 30,
    method: str = "2opt",
    n_runs_refinement: int = 5000,
    subsample_refinement: int = 4000,
    alpha: float = 0.3,
    top_k_post_refinement: int = 50,
    normalize_embeddings: bool = True,
    similarity_metric: Literal["cosine", "csls"] = "cosine",
    save_every: int = 50,
):

    get_sim = get_csls_sim if similarity_metric == "csls" else cos_sim_matrix

    if normalize_embeddings:
        X_train = N(X_train)
        Y_train = N(Y_train)
    # Stage 1: Match anchors (centroids) using a mini-vec2vec approach
    all_centers1, all_centers2 = [], []
    for _ in trange(n_runs_aligned_centroids, leave=False, desc="Matching centroids"):
        centers1, centers2 = aligned_centroids(
            X_train, Y_train, n_clusters=n_clusters, n_runs=n_runs, method=method
        )
        all_centers1.append(centers1)
        all_centers2.append(centers2)

    all_centers1 = torch.cat(all_centers1, dim=0)
    all_centers2 = torch.cat(all_centers2, dim=0)

    sim1 = get_sim(X_train, all_centers1)
    sim2 = get_sim(Y_train, all_centers2)
    sim_similarity = get_sim(sim1, sim2)

    top_similar = sim_similarity.topk(dim=-1, k=top_k).indices

    coefs = torch.ones(top_k) / top_k  # N(1 / (1 + torch.arange(k))**.5, p=1) #
    Y_matched = Y_train[top_similar].transpose(-1, -2) @ coefs
    W = train_orthogonal_linear(X_train, Y_matched)

    # Copy for saving the initial alignment before refinement
    W_initial = W.clone()

    # accuracy, cosine, mean_rank = accuracy_cosine_similarity(X_test @ W, Y_test)
    # print("First stage", accuracy, cosine, mean_rank)
    W_history = [W.clone()]
    for i in (
        pbar := trange(n_runs_refinement, leave=False, desc="Iterative refinement")
    ):
        sample_points = X_train[torch.randperm(len(X_train))[:subsample_refinement]]
        sample_similarities = get_sim(sample_points @ W, Y_train)
        neighbors = sample_similarities.topk(dim=-1, k=top_k_post_refinement).indices
        sample_matched = Y_train[neighbors].mean(dim=1)
        W_new = train_orthogonal_linear(sample_points, sample_matched)
        effective_alpha = alpha_schedule(
            i, initial=alpha, minimun=0.1, start=1500, end=2500
        )
        W_new = (1 - effective_alpha) * W + effective_alpha * W_new
        # Project to orthogonal using SVD to prevent drift

        try:
            U, _, Vt = torch.svd(W_new)
            W_new = U @ Vt.T
            # Compute the diff between W and W_new to monitor convergence (optional)
            diff = torch.norm(W - W_new).item()
            W = W_new
        except Exception as e:
            print(f"SVD failed at iteration {i} with error: {e}. Skipping update.")

        if X_test is not None and Y_test is not None:
            _, cosine, mean_rank = accuracy_cosine_similarity(X_test @ W, Y_test)
            pbar.set_postfix(
                {
                    "diff": f"{diff:.3f}",
                    "cos": f"{cosine:.3f}",
                    "rank": f"{mean_rank:.3f}",
                    "alpha": f"{effective_alpha:.2f}",
                }
            )
        else:
            pbar.set_postfix({"diff": f"{diff:.4f}", "alpha": f"{effective_alpha:.2f}"})
        # eval_score_current = eval_score(X_train, Y_train, W)
        # pbar.set_postfix({"eval_score": f"{eval_score_current:.4f}"})
        if (i + 1) % save_every == 0:
            W_history.append(W.clone())

    W_history = torch.stack(W_history, dim=0)
    metadata = {
        "n_runs_aligned_centroids": n_runs_aligned_centroids,
        "n_clusters": n_clusters,
        "method": method,
        "n_runs_refinement": n_runs_refinement,
        "subsample_refinement": subsample_refinement,
        "alpha": alpha,
        "top_k_post_refinement": top_k_post_refinement,
    }

    pck = {"W_initial": W_initial, "W": W, "metadata": metadata, "W_history": W_history}
    return pck


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate language embeddings for NSD stimuli."
    )
    parser.add_argument(
        "--subject_x",
        type=int,
        default=None,
        required=False,
        help="Source subject to align",
    )
    parser.add_argument(
        "--subject_y",
        type=int,
        default=None,
        required=False,
        help="Target subject to align",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=None,
        required=False,
        help=(
            "List of subjects to align. If subject_x and subject_y are provided, "
            "this argument is ignored."
        ),
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="mlp_embeddings_standarized",
        help="Path to the folder containing the standardized embeddings",
    )
    parser.add_argument(
        "--input_template",
        type=str,
        default="avg_ws_mlp_v1_128_768_sub-{subject:02d}.pt",
        help="Template for input files. Should contain {subject} placeholder.",
    )

    parser.add_argument(
        "--output_template",
        type=str,
        default="mini-vec2vec-alignment-model-{model_name}-sub-{subject:02d}_{mean_rank:.2f}.pt",
        help="Template for output files. Should contain {subject_x} and {subject_y} placeholders.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="final_alignment_models",
        help="Path to the folder to save the alignments",
    )
    parser.add_argument(
        "--n_runs_aligned_centroids",
        type=int,
        default=50,
        help="Number of runs for aligned centroids stage",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=20,
        help="Number of clusters for aligned centroids stage",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=30,
        help="Number of runs for aligned centroids stage",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="2opt",
        help="Method for quadratic assignment in aligned centroids stage",
    )
    parser.add_argument(
        "--n_runs_refinement",
        type=int,
        default=1000,
        help="Number of runs for iterative refinement stage",
    )
    parser.add_argument(
        "--subsample_refinement",
        type=int,
        default=4000,
        help="Number of samples to use for each run in iterative refinement stage",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Interpolation factor between current solution and new solution in iterative refinement stage",
    )
    parser.add_argument(
        "--top_k_post_refinement",
        type=int,
        default=20,
        help="Number of neighbors to use for matching after refinement",
    )
    parser.add_argument(
        "--similarity_metric",
        type=str,
        default="csls",
        choices=["cosine", "csls"],
        help="Similarity metric to use for matching (cosine or csls)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=32,
        help="Number of neighbors to use for initial matching",
    )
    # parser.add_argument(
    #     "--normalize_embeddings",
    #     action="store_true",
    #     help="Whether to normalize embeddings before alignment",
    # )

    args = parser.parse_args()

    args.input_folder = Path(args.input_folder)
    assert (
        args.input_folder.exists()
    ), f"Input folder {args.input_folder} does not exist"

    args.output_folder = Path(args.output_folder)
    args.output_folder.mkdir(parents=True, exist_ok=True)

    # Format as a list of pairs of subjects to align
    if args.subjects is None:
        args.subjects = list(range(1, 8 + 1))

    return args

@alert
def main():
    args = parse_args()

    model_templates = [
        "<path-to-models>/vit_small_patch14_dinov2.lvd142m_pool-cls.pt",
        "<path-to-models>/vit_large_patch16_224.augreg_in21k_pool-cls.pt",
        "<path-to-models>//vit_base_patch16_clip_224.laion2b_pool-cls.pt",
        "<path-to-models>/nsd_language_embeddings_all-mini.pt",
    ]


    for model_template in model_templates:
        model_name = Path(model_template).stem
        model_features = torch.load(model_template)["feats"][:, -1, :].float() # Take the last layer
        all_indices = list(range(model_features.shape[0]))




        for subject in (pbar := tqdm(args.subjects)):                
            pbar.set_description(f"subject=sub-{subject} Model={model_name[:30]}")
            input_file = args.input_folder / args.input_template.format(subject=subject)
            assert input_file.exists(), f"Input file {input_file} does not exist"

            pck_source = torch.load(input_file)
            X_train = pck_source["Z_train"]
            X_test = pck_source["Z_test"]
            X_labels_test = pck_source["labels_test"]
            X_labels_test = pck_source["labels_train"]

            # Make a list of the indices of the test stimuli that are in the subject
            X_all_labels = set(torch.cat([X_labels_test, X_labels_test], dim=0).numpy().tolist())
            model_train_indices = [i for i in all_indices if i not in X_all_labels]
            # Shuffle the train indices to avoid any bias in the order
            np.random.shuffle(model_train_indices)


            # Full dimension
            Y_train = model_features[model_train_indices]
            Y_test = model_features[X_labels_test]

            # Project to same dimensionality space with PCA
            pca = PCA(n_components=X_train.shape[1])
            Y_train = torch.from_numpy(pca.fit_transform(Y_train)).float()
            Y_test = torch.from_numpy(pca.transform(Y_test)).float()

           
            min_n_test = 515
            X_test = X_test[:min_n_test]
            X_labels_test = X_labels_test[:min_n_test]
            Y_test = Y_test[:min_n_test]

            pck = process_subjects(
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test,
                n_runs_aligned_centroids=args.n_runs_aligned_centroids,
                n_clusters=args.n_clusters,
                n_runs=args.n_runs,
                method=args.method,
                top_k=args.top_k,
                n_runs_refinement=args.n_runs_refinement,
                subsample_refinement=args.subsample_refinement,
                alpha=args.alpha,
                top_k_post_refinement=args.top_k_post_refinement,
                normalize_embeddings=True,
                similarity_metric=args.similarity_metric,
            )
            pck["subject"] = subject
            pck["model"] = model_name

            accuracy, cosine, mean_rank = accuracy_cosine_similarity(
                X_test @ pck["W"], Y_test
            )

            rsa = compute_rsa(X_test @ pck["W"], Y_test)

            pck["evaluation"] = {
                "accuracy": accuracy,
                "cosine": cosine,
                "mean_rank": mean_rank,
                "n_test": len(X_test),
                "rsa": rsa,
            }

        output_file = args.output_folder / args.output_template.format(
            subject=subject, model=model_name, mean_rank=mean_rank
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)

        torch.save(pck, output_file)
        print(
            f"Finished alignment for sub-{subject} and model-{model_name}. "
            f"Accuracy: {accuracy:.4f}, Cosine: {cosine:.4f}, Mean Rank: {mean_rank:.2f}"
        )
        send_message(
            f"Finished alignment for sub-{subject} and model-{model_name}. "
            f"Accuracy: {accuracy:.4f}, Cosine: {cosine:.4f}, Mean Rank: {mean_rank:.2f}"
        )


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    main()
