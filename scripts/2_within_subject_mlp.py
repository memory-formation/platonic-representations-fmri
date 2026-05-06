import os
import argparse
from datetime import datetime
from pathlib import Path
import warnings


import numpy as np
import torch
from tqdm import trange, tqdm
from torch import nn

from fmri_mapping.io.nsd import get_resource
from fmri_mapping.utils.logger import WandbLogger
from fmri_mapping.embedding.ops import (
    split_betas_by_subject_index,
    split_repetitions,
    to_tensor,
)
from fmri_mapping.embedding.evaluation import (
    compute_metric_curves,
    avg_metric_combinations,
    accuracy_cosine_similarity,
    apply_batched_model,
)

from fmri_mapping.embedding.losses import (
    info_nce_symmetric,
    geometry_cosine_reg,
    pull_cosine,
    circle_loss,
    vsp_loss,
    loo_repeat_denoise_loss,
)
from fmri_mapping.embedding.mlp import ResidualMLP
from fmri_mapping.utils import get_device


WANDB_MODE = "online"  # "offline", "online" or "disabled"
WANDB_PROJECT = "fmri-mapping-encoder"
os.environ["WANDB_SILENT"] = "true"


EVALUATION_BATCH_SIZE = 8192


def compute_losses(
    Z1,
    Z2,
    lambda_nce: float = 1.0,
    temperature: float = 0.09,
    lambda_geom: float = 0.1,
    lambda_pull: float = 0.5,
    lambda_identity_preserve: float = 0.0,
    lambda_circle: float = 0.0,
    lambda_vsp: float = 0.0,
    lambda_mse: float = 0.0,
):
    total_loss = torch.tensor(0.0, device=Z1.device)
    stats = {}

    if lambda_nce > 0:
        loss_info_nce = info_nce_symmetric(Z1, Z2, temperature=temperature)
        total_loss = total_loss + lambda_nce * loss_info_nce
        stats["loss_info_nce"] = loss_info_nce.item()
    if lambda_geom > 0:
        loss_cosine = geometry_cosine_reg(Z1, Z2)
        total_loss = total_loss + lambda_geom * loss_cosine
        stats["loss_geom_cosine"] = loss_cosine.item()
    if lambda_pull > 0:
        loss_pull_cosine = pull_cosine(Z1, Z2)
        total_loss = total_loss + lambda_pull * loss_pull_cosine
        stats["loss_pull_cosine"] = loss_pull_cosine.item()
    if lambda_identity_preserve > 0:
        loss_identity = 0.5 * (pull_cosine(Z1, Z1) + pull_cosine(Z2, Z2))
        total_loss = total_loss + lambda_identity_preserve * loss_identity
        stats["loss_identity_preserve"] = loss_identity.item()
    if lambda_circle > 0:
        loss_circle = circle_loss(Z1, Z2)
        total_loss = total_loss + lambda_circle * loss_circle
        stats["loss_circle"] = loss_circle.item()
    if lambda_vsp > 0:
        loss_vsp = vsp_loss(Z1, Z2)
        total_loss = total_loss + lambda_vsp * loss_vsp
        stats["loss_vsp"] = loss_vsp.item()
    if lambda_mse > 0:
        loss_mse = nn.functional.mse_loss(Z1, Z2, reduction="sum")
        total_loss = total_loss + lambda_mse * loss_mse
        stats["loss_mse"] = loss_mse.item()

    return total_loss, stats


def _add_keys_prefix(d: dict, prefix: str) -> dict:
    return {f"{prefix}.{k}": v for k, v in d.items()}


def train_mlp_triplets(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    Z3: torch.Tensor,
    Z1t: torch.Tensor,
    Z2t: torch.Tensor,  # <-- test views
    model: nn.Module,
    decoder_model: nn.Module,
    accuracy_fn,  # <-- accuracy_cosine_similarity
    temperature: float = 0.07,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    steps: int = 2000,
    eval_every: int = 200,
    device: str = "cuda",
    seed: int = 42,
    lambda_nce: float = 1.0,
    lambda_geom: float = 0.01,
    lambda_pull: float = 0.0,
    lambda_identity_preserve: float = 0.0,
    lambda_circle: float = 0.0,
    lambda_vsp: float = 0.0,
    lambda_loo: float = 0.0,
    lambda_mse: float = 0.1,
    milestones: list[int] = [1300, 1400, 1500, 1600, 1700, 1800, 1900],
    gamma: float = 0.3,
    logger: WandbLogger = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    assert Z1.shape == Z2.shape == Z3.shape
    N, D = Z1.shape

    model = model.to(device)
    decoder_model = decoder_model.to(device)
    model.train()
    decoder_model.train()

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(decoder_model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=milestones,  # can add more, e.g. [1500, 1800]
        gamma=gamma,  # lr <- lr * 0.1 at each milestone
    )

    Z1 = Z1.to(device)
    Z2 = Z2.to(device)
    Z3 = Z3.to(device)
    Z1t = Z1t.to(device)
    Z2t = Z2t.to(device)
    pairs = [(Z1, Z2), (Z1, Z3), (Z2, Z3)]
    stats = []

    postfix = {}
    postfix["loss"] = f"{0:.3f}"
    postfix["alpha"] = f"{model.alpha.item():.2f}"
    postfix["acc"] = f"{0:.3f}"
    postfix["rank"] = f"{0:.2f}"

    model.train()
    loss_kwargs = dict(
        lambda_nce=lambda_nce,
        lambda_geom=lambda_geom,
        lambda_pull=lambda_pull,
        lambda_identity_preserve=lambda_identity_preserve,
        lambda_circle=lambda_circle,
        lambda_vsp=lambda_vsp,
        lambda_mse=lambda_mse,
        temperature=temperature,
    )

    for step in (pbar := trange(steps, desc="Training MLP", position=1, leave=False)):
        opt.zero_grad()

        idx = torch.randint(0, N, (batch_size,), device=device)
        Zal, Zbl, Zcl = model(Z1[idx]), model(Z2[idx]), model(Z3[idx])
        loss = 0.0
        for Zal, Zbl in [(Zal, Zbl), (Zal, Zcl), (Zbl, Zcl)]:
            encode_losses, encode_stats = compute_losses(Zal, Zbl, **loss_kwargs)
            logger.logkvs(encode_stats)
            loss = loss + encode_losses

        loss = loss / len(pairs)

        if lambda_loo > 0:
            loss_loo = loo_repeat_denoise_loss(Z1[idx], Z2[idx], Z3[idx])
            loss = loss + lambda_loo * loss_loo
            logger.logkv("loss_loo", loss_loo.item())

        logger.logkvs({"loss_total": loss.item(), "alpha": model.alpha.item()})

        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        logger.logkv("grad_norm", norm.item())
        opt.step()
        scheduler.step()
        logger.dumpkvs()
        postfix["loss"] = f"{loss.item():.3f}"
        postfix["alpha"] = f"{model.alpha.item():.2f}"
        pbar.set_postfix(postfix)

        # ---- evaluation ----
        if step % eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                Z1t_hat = model(Z1t)
                Z2t_hat = model(Z2t)

                acc, cosine, mean_rank = accuracy_fn(Z1t_hat, Z2t_hat)

            model.train()

            stat = {
                "step": step,
                "test_accuracy": acc,
                "test_cosine": cosine,
                "test_mean_rank": mean_rank,
                "alpha": model.alpha.item(),
            }
            postfix["acc"] = f"{acc:.3f}"
            postfix["rank"] = f"{mean_rank:.2f}"
            pbar.set_postfix(postfix)
            stats.append(stat)
            logger.logkvs(stat)
            logger.dumpkvs(force=True)

    return model, stats


def process_subject_mlp(
    subject: int,
    Z: torch.Tensor,
    model_parameters: dict,
    training_parameters: dict,
    output_filename: Path,
    device: str = "cuda",
    dtype=torch.float,
    train_inpute: bool = "mean",
    print_eval: bool = False,
    skip_save: bool = False,
):
    df_repetitions = split_repetitions(
        subject=subject,
        shuffle_indexes=True,
        how="outer",
    )
    df_repetitions_train = df_repetitions.query("not shared")
    df_repetitions_test = df_repetitions.query("shared")

    # Prepare data
    Z_numpy = Z.cpu().numpy().astype(np.float32)
    Z1, Z2, Z3 = split_betas_by_subject_index(
        Z_numpy, df_repetitions_train, inpute=train_inpute
    )
    Z1, Z2, Z3 = to_tensor(Z1, Z2, Z3, device=device, dtype=dtype)

    Z1t, Z2t, _ = split_betas_by_subject_index(
        Z_numpy, df_repetitions_test, inpute=False
    )
    Z1t, Z2t = to_tensor(Z1t, Z2t, device=device, dtype=dtype)

    # Define model
    model = ResidualMLP(dtype=dtype, **model_parameters)
    decoder_model = ResidualMLP(dtype=dtype, **model_parameters)

    logger = WandbLogger(
        name=f"stage_b_subject_{subject}",
        log_frequency=10,
        mode=WANDB_MODE,
        project=WANDB_PROJECT,
        config={
            "subject": subject,
            **model_parameters,
            **training_parameters,
        },
    )
    # Train model
    model, stats = train_mlp_triplets(
        Z1,
        Z2,
        Z3,
        Z1t,
        Z2t,
        accuracy_fn=accuracy_cosine_similarity,
        model=model,
        decoder_model=decoder_model,
        device=device,
        logger=logger,
        **training_parameters,
    )
    logger.finish()

    Z = Z.to(device, dtype=dtype)
    Z_hat = apply_batched_model(model, Z, batch_size=EVALUATION_BATCH_SIZE)
    Z_hat = Z_hat.cpu()

    Z_hat_numpy = Z_hat.numpy().astype(np.float32)

    df_metric = compute_metric_curves(Z_hat_numpy, subject=subject, full_curves=False)
    df_metric_agg = avg_metric_combinations(df_metric)
    df_metric_curves = compute_metric_curves(
        Z_hat_numpy, subject=subject, full_curves=True
    )
    results = {
        "aggregated_metrics": df_metric_agg.to_dict(orient="records"),
        "metric_curves": df_metric_curves.to_dict(orient="records"),
    }

    if not skip_save:
        d_in = model_parameters["d_in"]
        save_mlp_checkpoint(
            model=model,
            model_parameters=model_parameters,
            Z=Z_hat,
            training_parameters=training_parameters,
            subject=subject,
            d_in=d_in,
            filename=output_filename,
            stats=stats,
            results=results,
        )

    if print_eval:
        string = f"\n\nSubject {subject}:\n{df_metric_agg.to_string(index=False)}\n"
        print(string)


def save_mlp_checkpoint(
    model: nn.Module,
    model_parameters: dict,
    training_parameters: dict,
    Z: torch.Tensor,
    subject: int,
    d_in: int,
    filename: Path,
    stats: list = None,
    results: dict = None,
):

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_parameters": model_parameters,
        "Z": Z,
        "training_parameters": training_parameters,
        "subject": subject,
        "d_in": d_in,
        "timestamp": datetime.now().isoformat(),
        "training_stats": stats,
        "evaluation_results": results,
    }

    torch.save(checkpoint, filename)


def parse_args():

    parser = argparse.ArgumentParser(description="Within-subject MLP embedding")
    # Sujects (list of integers +)
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="Subject(s) to process (e.g., --subject 1 2 3). If not provided, process all.",
    )
    # Folder of linear embeddings
    parser.add_argument(
        "--linear_embeddings_dir",
        type=str,
        default="./linear_embeddings",
        help="Directory containing linear embeddings.",
    )
    # Embeddings template
    parser.add_argument(
        "--embeddings_template",
        type=str,
        default="ws_linear_v1_768_128_sub-{subject:02d}.pt",
        help="Template for linear embeddings files including {subject:02d} placeholder.",
    )
    # output_dir
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./mlp_embeddings",
        help="Directory to save MLP embeddings and results.",
    )
    parser.add_argument(
        "--output_template",
        type=str,
        default="ws_mlp_{version}_{d_in}_{d_hidden}_sub-{subject:02d}.pt",
        help="Template for MLP embeddings files including {subject:02d} placeholder.",
    )

    # Model parameters
    parser.add_argument(
        "--d_in",
        type=int,
        help="Dimensionality of input linear embeddings. If none is inferred from files.",
    )
    # d_hidden
    parser.add_argument(
        "--d_hidden",
        type=int,
        default=768,
        help="Dimensionality of hidden layers.",
    )
    # depth
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Number of hidden layers.",
    )
    # initial alpha
    parser.add_argument(
        "--initial_alpha",
        type=float,
        default=0.25,
        help="Initial value of alpha parameter.",
    )
    # alpha trainable
    parser.add_argument(
        "--fixed_alpha",
        action="store_true",
        help="If set, alpha parameter is fixed (not trainable).",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for MLP.",
    )

    # training parameters
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda' or 'cpu'). If none, inferred automatically.",
    )
    # temperature
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.09,
        help="Temperature for InfoNCE loss.",
    )
    # lr
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    # weight decay
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for optimizer.",
    )
    # batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for training.",
    )
    # steps
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of training steps.",
    )
    # eval every
    parser.add_argument(
        "--eval_every",
        type=int,
        default=200,
        help="Evaluate every N steps.",
    )
    # lambda pull
    parser.add_argument(
        "--lambda_pull",
        type=float,
        default=0.5,
        help="Weight for pull cosine loss.",
    )
    # lambda geom
    parser.add_argument(
        "--lambda_geom",
        type=float,
        default=0.1,
        help="Weight for geometric cosine loss.",
    )
    parser.add_argument(
        "--lambda_identity_preserve",
        type=float,
        default=0.0,
        help="Weight for identity preservation loss.",
    )
    parser.add_argument(
        "--lambda_nce",
        type=float,
        default=1.0,
        help="Weight for InfoNCE loss.",
    )
    parser.add_argument(
        "--lambda_vsp",
        type=float,
        default=0.0,
        help="Weight for VSP loss.",
    )
    parser.add_argument(
        "--lambda_loo",
        type=float,
        default=0.0,
        help="Weight for leave-one-out repeat denoising loss.",
    )
    parser.add_argument(
        "--lambda_mse",
        type=float,
        default=0.05,
        help="Weight for MSE loss.",
    )
    # lambda_circle
    parser.add_argument(
        "--lambda_circle",
        type=float,
        default=0.0,
        help="Weight for circle loss.",
    )
    parser.add_argument(
        "--print_eval",
        action="store_true",
        help="Whether to print evaluation results to console",
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
        help="If set, skip saving the model checkpoint.",
    )

    args = parser.parse_args()

    if args.subjects is None:
        subjects = get_resource("stimulus").subject.unique().tolist()
        subjects.sort()
        args.subjects = subjects

    args.linear_embeddings_dir = Path(args.linear_embeddings_dir)
    assert (
        args.linear_embeddings_dir.exists()
    ), f"Directory {args.linear_embeddings_dir} does not exist."

    for subject in args.subjects:
        filename = args.embeddings_template.format(subject=subject)
        filepath = args.linear_embeddings_dir / filename
        assert filepath.exists(), f"File {filepath} does not exist."

    return args


def main():
    os.chdir(Path(__file__).parent)
    args = parse_args()

    device = args.device if args.device is not None else get_device()
    model_parameters = dict(
        d_hidden=args.d_hidden,
        depth=args.depth,
        alpha_trainable=not args.fixed_alpha,
        initial_alpha=args.initial_alpha,
        d_in=args.d_in,  # Could be None
        dropout=args.dropout,
    )

    training_parameters = dict(
        temperature=args.temperature,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_every=args.eval_every,
        lambda_pull=args.lambda_pull,
        lambda_geom=args.lambda_geom,
        lambda_identity_preserve=args.lambda_identity_preserve,
        lambda_circle=args.lambda_circle,
        lambda_vsp=args.lambda_vsp,
        lambda_nce=args.lambda_nce,
        lambda_loo=args.lambda_loo,
        lambda_mse=args.lambda_mse,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject in tqdm(args.subjects, position=0, desc="Subjects"):
        filename = args.embeddings_template.format(subject=subject)
        filepath = args.linear_embeddings_dir / filename
        pckg = torch.load(filepath)
        assert pckg["subject"] == subject, f"Subject mismatch in file {filepath}"
        Z = pckg["Z"]
        if args.d_in is None:
            d_in = Z.shape[1]
            model_parameters["d_in"] = d_in
        else:
            d_in = args.d_in
            if Z.shape[1] != d_in:
                warnings.warn(
                    f"Input dimensionality mismatch for subject {subject}: "
                    f"expected {d_in}, got {Z.shape[1]}. Truncating as needed."
                )

                Z = Z[:, :d_in]
        output_template = args.output_template
        output_filename = output_dir / output_template.format(
            version=args.version,
            d_in=d_in,
            d_hidden=args.d_hidden,
            subject=subject,
        )

        process_subject_mlp(
            subject=subject,
            Z=Z,
            model_parameters=model_parameters,
            training_parameters=training_parameters,
            device=device,
            output_filename=output_filename,
            dtype=torch.float,
            train_inpute="mean",
            print_eval=args.print_eval,
            skip_save=args.skip_save,
        )


if __name__ == "__main__":
    main()
