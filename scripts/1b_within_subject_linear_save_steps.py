import argparse
from typing import Literal
import torch
import os
import pandas as pd
import numpy as np
import gc
from pathlib import Path
from sklearn.decomposition import PCA
from mvlearn.embed import MCCA

from tqdm import tqdm

from fmri_mapping.embedding.evaluation import (
    compute_metric_curves,
    avg_metric_combinations,
)

from fmri_mapping.embedding.linear import (
    fit_session_residualizer,
    voxel_reliability_loo,
    distill_mcca_projector,
    estimate_sample_weights,
)
from fmri_mapping.embedding.ops import (
    get_subject_betas,
    get_confound_matrix,
    split_betas_by_subject_index,
    split_repetitions,
)

from fmri_mapping.io.nsd import get_resource
from fmri_mapping.embedding.evaluation import compute_metric_curves



def compute_linear_within_subject_pipeline(
    subject: int,
    betas: np.ndarray,
    df_stimuli: pd.DataFrame,
    df_repetitions_train: pd.DataFrame,
    confound_reg: float = 0.0,
    reliability_eps: float = 1e-6,
    n_components_pca: int = 512,
    n_components_mcca: int = 128,
    reg_mcca: float = 0.1,
    mcca_inpute="mean",
    mcca_distill_reg: float = 0.5,
    progress: bool = True,
    apply_sample_weights: bool = True,
):
    """Compute within-subject linear embedding pipeline."""
    pbar = tqdm(
        total=5,
        desc=f"Subject {subject}",
        disable=not progress,
        position=1,
        leave=False,
    )

    train_mask = (df_stimuli.shared == False).to_numpy(
        dtype=bool
    )  # Should be all true. But for safety.

    pbar.set_description(f"Residualizing confounds")
    # Residualize betas with session_run confound
    C = get_confound_matrix(subject)  # n_trials x n_confounds
    B = fit_session_residualizer(
        betas[train_mask, :], C[train_mask, :], lam=confound_reg
    )




    steps_folder = Path("steps")
    steps_folder.mkdir(exist_ok=True)
    
    betas_res = betas - C @ B  # n_trials x n_voxels


    np.save(steps_folder / f"subject-{subject:02d}_betas_prec.npy", betas)

    np.save(steps_folder / f"subject-{subject:02d}_betas_res.npy", betas_res)

    del betas
    gc.collect()
    pbar.update(1)


    #Compute reliability mask
    W_rel = voxel_reliability_loo(
        *split_betas_by_subject_index(betas_res, df_repetitions_train, inpute=False)
    )



    W_rel = np.clip(W_rel, 0.0, None) + reliability_eps  # n_voxels
    # betas_rel = W_rel[None, :] * betas_res  # n_trials x n_voxels
    betas_rel = betas_res
    del betas_res
    gc.collect()
    pbar.update(1)

    np.save(steps_folder / f"subject-{subject:02d}_betas_rel.npy", betas_rel)

    # # Apply PCA
    pbar.set_description(f"Fitting PCA")
    pca = PCA(n_components=n_components_pca, random_state=42)

    if apply_sample_weights:
        W_sample = estimate_sample_weights(betas_rel, df_repetitions_train, train_mask)
        W_sample = np.clip(W_sample, 1e-3, None)  # Avoid zero weights
        W_sample = np.sqrt(W_sample)  # Convert to sample weights for PCA
        pca.fit(W_sample[:, None] * betas_rel[train_mask, :])  # Fit on train
    else:
        pca.fit(betas_rel[train_mask, :])  # Fit on train

    W_pca = pca.components_.T  # n_voxels x n_pca_components
    mu_pca = pca.mean_  # n_voxels

    # Apply PCA
    betas_pca = (betas_rel - mu_pca[None, :]) @ W_pca  # n_trials x n_pca_components

    np.save(steps_folder / f"subject-{subject:02d}_betas_pca-no-rel.npy", betas_pca)
    del betas_rel
    gc.collect()
    pbar.update(1)

    # Fit MCCA
    pbar.set_description(f"Fitting MCCA")
    mcca = MCCA(n_components=n_components_mcca, regs=reg_mcca)
    X1, X2, X3 = split_betas_by_subject_index(
        betas_pca, df_repetitions_train, inpute=mcca_inpute
    )  # Fit on train
    mcca.fit([X1, X2, X3])
    pbar.update(1)

    # Distill a unique projector with ridge regression
    pbar.set_description(f"Distilling MCCA projector")
    W_mcca, b_mcca = distill_mcca_projector(
        betas_pca[train_mask, :], mcca, lam_ridge=mcca_distill_reg
    )

    # Inference on all data
    Z = (betas_pca @ W_mcca) + b_mcca[None, :]  # n_trials x n_mcca_components
    matrices = {
        "C": C,  # n_trials x n_confounds
        "B": B,  # n_confounds x n_voxels
        "W_rel": W_rel,  # n_voxels
        "W_pca": W_pca,  # n_voxels x n_pca_components
        "mu_pca": mu_pca,  # n_voxels
        "W_mcca": W_mcca,  # n_pca_components x n_mcca_components
        "b_mcca": b_mcca,  # n_mcca_components
    }
    pbar.update(1)

    return Z, matrices


def process_subject(
    subject: int,
    rois: list[int],
    n_components_pca: int = 512,
    n_components_mcca: int = 128,
    reliability_eps: float = 1e-6,
    reg_mcca: float = 0.1,
    mcca_distill_reg: float = 0.5,
    confound_reg: float = 0.0,
    mcca_inpute: Literal["mean", "zeros"] | None = "mean",
    quantile: float = 0.004,
    compute_curves: bool = True,
    betas_subfolder: str = "betas",
    scaling: float = 300.0,
    low_memory: bool = False,
    apply_sample_weights: bool = True,
):
    """Process a single subject within-subject linear embedding pipeline."""
    # # Get data
    df_stimuli = get_resource("stimulus").query("subject == @subject and exists")
    df_repetitions = split_repetitions(subject=subject, shuffle_indexes=True)
    df_repetitions_train = df_repetitions.query("not shared")
    # Get subject betas
    betas, qcuts = get_subject_betas(
        subject=subject,
        roi=rois,
        low_memory=low_memory,
        q=quantile,
        return_quantiles=True,
        betas_subfolder=betas_subfolder,
        scaling=scaling,
    )  # n_trials x n_voxels

    Z, matrices = compute_linear_within_subject_pipeline(
        subject=subject,
        betas=betas,
        df_stimuli=df_stimuli,
        df_repetitions_train=df_repetitions_train,
        confound_reg=confound_reg,
        reliability_eps=reliability_eps,
        n_components_pca=n_components_pca,
        n_components_mcca=n_components_mcca,
        reg_mcca=reg_mcca,
        mcca_inpute=mcca_inpute,
        mcca_distill_reg=mcca_distill_reg,
        apply_sample_weights=apply_sample_weights,
    )

    df_metric = compute_metric_curves(Z, subject=subject, full_curves=False)

    # Get curves for diagnosis when increasing dimensions
    if compute_curves:
        df_metric_curves = compute_metric_curves(Z, subject=subject)
    else:
        df_metric_curves = df_metric

    df_metric_agg = avg_metric_combinations(df_metric)

    matrices["qcuts"] = qcuts  # n_voxels x 2

    return Z, matrices, df_metric_curves, df_metric_agg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Within-subject linear embedding pipeline"
    )
    # Subjects -> List of integers with subjects to process
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="List of subjects to process (NSD subjects IDs)",
    )

    # Rois -> List of integers with ROIs to process
    parser.add_argument(
        "--rois",
        type=int,
        nargs="+",
        default=[0],  # [1, 4, 5, 6, 181, 184, 185, 186],  # V1-V4 LH + RH
        help="List of ROIs to process (NSD ROI IDs)",
    )

    # n_components_pca
    parser.add_argument(
        "--n_components_pca",
        type=str,
        default=768,
        help="Number of PCA components",
    )
    # n_components_mcca
    parser.add_argument(
        "--n_components_mcca",
        type=int,
        default=128,
        help="Number of MCCA components",
    )
    # reg_mcca
    parser.add_argument(
        "--reg_mcca",
        type=float,
        default=0.1,
        help="Regularization for MCCA",
    )
    # mcca_distill_reg
    parser.add_argument(
        "--mcca_distill_reg",
        type=float,
        default=0.5,
        help="Ridge regularization for MCCA distillation",
    )
    # confound_reg
    parser.add_argument(
        "--confound_reg",
        type=float,
        default=0.0,
        help="Ridge regularization for confound residualization",
    )
    # reliability_eps
    parser.add_argument(
        "--reliability_eps",
        type=float,
        default=1e-6,
        help="Numerical stability for voxel reliability",
    )
    # mcca_inpute
    parser.add_argument(
        "--mcca_inpute",
        type=str,
        choices=["mean", "zeros", "none"],
        default="mean",
        help="Inpute strategy for missing repetitions in MCCA",
    )

    parser.add_argument(
        "--quantile",
        type=float,
        default=0.004,
        help="Quantile for beta clipping",
    )
    parser.add_argument(
        "--scaling", type=float, default=300.0, help="Scaling factor for betas"
    )

    # output_folder
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./linear_embeddings",
        help="Output folder to save results",
    )

    parser.add_argument(
        "--filename",
        type=str,
        default="ws_linear_{version}_{n_components_pca}_{n_components}_sub-{subject:02d}.pt",
        help="Filename to save results",
    )
    parser.add_argument(
        "--betas_subfolder",
        type=str,
        default="betas",
        help="Subfolder within the NSD betas directory where the ROI .npy files are located",
    )

    parser.add_argument(
        "--print_eval",
        action="store_true",
        help="Whether to print evaluation results to console",
    )
    parser.add_argument(
        "--low_memory",
        action="store_true",
        help="Whether to use low memory quantile computation for beta clipping",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version string to include in output filename",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="Whether to skip saving the results to disk",
    )

    args = parser.parse_args()



    if str(args.n_components_pca).isdigit():
        args.n_components_pca = int(args.n_components_pca)

    if args.subjects is None:
        subjects = get_resource("stimulus").subject.unique().tolist()
        subjects.sort()
        args.subjects = subjects

    if args.mcca_inpute == "none":
        args.mcca_inpute = None

    return args


def main():
    args = parse_args()

    subjects = args.subjects
    rois = args.rois
    filename = args.filename
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    parameters = dict(
        n_components_pca=args.n_components_pca,
        n_components_mcca=args.n_components_mcca,
        reliability_eps=args.reliability_eps,
        reg_mcca=args.reg_mcca,
        mcca_distill_reg=args.mcca_distill_reg,
        confound_reg=args.confound_reg,
        mcca_inpute=args.mcca_inpute,
        compute_curves=False,
        quantile=args.quantile,
        betas_subfolder=args.betas_subfolder,
        scaling=args.scaling,
        low_memory=args.low_memory,
        apply_sample_weights=True,
    )

    for subject in (pbar := tqdm(subjects, desc="Processing subjects")):
        pbar.set_postfix(subject=subject)
        Z, matrices, df_metric_curves, df_metric_agg = process_subject(
            subject=subject,
            rois=rois,
            **parameters,
        )
        payload = {
            "subject": subject,
            "rois": rois,  # list[int]
            "Z": torch.from_numpy(Z),  # (N, D) tensor
            "matrices": {k: torch.from_numpy(v) for k, v in matrices.items()},
            "metrics": {
                "curves": df_metric_curves.to_dict(orient="records"),
                "agg": df_metric_agg.to_dict(orient="records"),
            },
            "parameters": parameters,
        }
        if not args.skip_save:
            filename_subject = filename.format(
                version=args.version,
                n_components_pca=args.n_components_pca,
                n_components=args.n_components_mcca,
                subject=subject,
            )
            filename_subject = output_folder / filename_subject
            torch.save(payload, filename_subject)

        if args.print_eval:
            string = (
                f"\n\nSubject {subject}:\n {df_metric_agg.to_string(index=False)}\n"
            )
            print(string)


if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
