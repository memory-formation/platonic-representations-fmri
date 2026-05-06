import argparse
import numpy as np
import torch

import os
from pathlib import Path

from tqdm import tqdm
import torch
import pandas as pd

from fmri_mapping.io.nsd import get_resource, get_subject_behavioural
from fmri_mapping.embedding.evaluation import split_repetitions


"""
# Generate standardized embeddings for encoder already refined model
python standarize_embeddings.py --input_folder mlp_embeddings \
    --input_template "ws_mlp_v1_128_768_sub-{subject:02d}.pt" \
    --output_folder mlp_embeddings_standarized \
    --output_template "trial_ws_mlp_v1_128_768_sub-{subject:02d}.pt"

# Generate standardized embeddings for encoder already refined model, averaged repetitions
python standarize_embeddings.py --input_folder mlp_embeddings \
    --input_template "ws_mlp_v1_128_768_sub-{subject:02d}.pt" \
    --output_folder mlp_embeddings_standarized \
    --output_template "avg_ws_mlp_v1_128_768_sub-{subject:02d}.pt"
"""


@torch.inference_mode()
def pandas_to_torch_dict(
    df: pd.DataFrame, columns, prefix: str = "", suffix: str = ""
) -> dict:
    """
    Convert a pandas DataFrame to a dictionary of torch tensors.
    """
    if columns is None:
        columns = df.columns
    return {
        f"{prefix}{col}{suffix}": torch.tensor(df[col].values, device="cpu")
        for col in columns
    }


def process_subject(
    subject: int,
    input_file: Path,
    output_file: Path,
):
    pck = torch.load(input_file)
    Z = pck["Z"]
    df = get_resource("stimulus").query("subject == @subject and exists")
    df_beh = get_subject_behavioural(subject)
    df_beh["shared"] = df["shared"].values
    assert len(Z) == len(
        df
    ), "Embeddings and stimulus dataframe must have the same length"
    assert len(df) == len(
        df_beh
    ), "Stimulus and behavioural dataframe must have the same length"
    df_train = df.query("not shared")
    df_test = df.query("shared")

    output_pck = {}
    output_pck["Z_train"] = Z[df_train.subject_index.values]

    train_main_vars = pandas_to_torch_dict(
        df_train.rename(columns={"nsd_id": "labels", "repetition": "repetitions"}),
        columns=["labels", "repetitions"],
        suffix="_train",
    )
    output_pck.update(train_main_vars)

    output_pck["Z_test"] = Z[df_test.subject_index.values]
    test_main_vars = pandas_to_torch_dict(
        df_test.rename(columns={"nsd_id": "labels", "repetition": "repetitions"}),
        columns=["labels", "repetitions"],
        suffix="_test",
    )
    output_pck.update(test_main_vars)

    df_beh = df_beh[["SESSION", "session_run", "RT", "ISCORRECT", "ISOLD", "shared"]]
    df_beh = df_beh.rename(
        columns={
            "SESSION": "session",
            "RT": "rt",
            "ISCORRECT": "is_correct",
            "ISOLD": "is_old",
        }
    )
    df_beh.is_correct = df_beh.is_correct.astype(bool)
    df_beh.is_old = df_beh.is_old.astype("int64").astype("boolean")
    df_beh_train = df_beh.query("not shared")
    df_beh_test = df_beh.query("shared")
    metadata_train = pandas_to_torch_dict(
        df_beh_train,
        columns=["session", "session_run", "rt", "is_correct", "is_old"],
    )
    metadata_train["subject"] = torch.tensor(subject, device="cpu")
    coco_id_train = metadata_test = pandas_to_torch_dict(
        df_train,
        columns=["coco_id"],
    )
    metadata_train["coco_id"] = coco_id_train["coco_id"]

    metadata_test = pandas_to_torch_dict(
        df_beh_test,
        columns=["session", "session_run", "rt", "is_correct", "is_old"],
    )

    metadata_test["subject"] = torch.tensor(subject, device="cpu")
    coco_id_test = pandas_to_torch_dict(
        df_test,
        columns=["coco_id"],
    )
    metadata_test["coco_id"] = coco_id_test["coco_id"]
    output_pck["metadata_train"] = metadata_train
    output_pck["metadata_test"] = metadata_test

    torch.save(output_pck, output_file)


def average_embeddings(Z, df_repetitions):
    """
    Average embeddings across repetitions.
    """
    Z_avg = Z[df_repetitions.subject_index_1.values].clone()
    # Where subject_index_2 is not nan, sum
    Z_avg[~df_repetitions.subject_index_2.isna().values] += Z[
        df_repetitions.subject_index_2.dropna().values
    ]
    # Where subject_index_3 is not nan, sum
    Z_avg[~df_repetitions.subject_index_3.isna().values] += Z[
        df_repetitions.subject_index_3.dropna().values
    ]
    # Divide by column exists (that contains the number of repetitions)
    Z_avg /= torch.tensor(df_repetitions.exists.values[:, None])
    labels = torch.tensor(df_repetitions.nsd_id.values)
    n_repetitions = torch.tensor(df_repetitions.exists.values)

    return Z_avg, labels, n_repetitions


def process_subject_average(
    subject: int,
    input_file: Path,
    output_file: Path,
):
    pck = torch.load(input_file)
    Z = pck["Z"]
    df = get_resource("stimulus").query("subject == @subject and exists")
    df_beh = get_subject_behavioural(subject)
    df_beh["shared"] = df["shared"].values
    assert len(Z) == len(
        df
    ), "Embeddings and stimulus dataframe must have the same length"
    assert len(df) == len(
        df_beh
    ), "Stimulus and behavioural dataframe must have the same length"

    df_repetitions = split_repetitions(
        subject=subject, shuffle_indexes=False, min_exists=1
    )
    df_no_reps = df["nsd_id"].drop_duplicates(keep="first").reset_index(drop=True)
    df_repetitions = df_repetitions.set_index("nsd_id").loc[df_no_reps.values].reset_index()
    Z_train, labels_train, n_repetitions_train = average_embeddings(Z, df_repetitions.query("not shared"))
    Z_test, labels_test, n_repetitions_test = average_embeddings(Z, df_repetitions.query("shared"))

    output_pck = {}
    output_pck["Z_train"] = Z_train
    output_pck["labels_train"] = labels_train
    output_pck["n_repetitions_train"] = n_repetitions_train

    output_pck["Z_test"] = Z_test
    output_pck["labels_test"] = labels_test
    output_pck["n_repetitions_test"] = n_repetitions_test

    output_pck["metadata_train"] = {
        "subject": torch.tensor(subject, device="cpu"),
    }
    output_pck["metadata_test"] = {
        "subject": torch.tensor(subject, device="cpu"),
    }
    torch.save(output_pck, output_file)  


def main():
    args = parse_args()

    assert (
        args.input_folder.exists()
    ), f"Input folder {args.input_folder} does not exist"
    args.output_folder.mkdir(parents=True, exist_ok=True)

    for subject in (pbar := tqdm(args.subjects)):

        input_file = args.input_folder / args.input_template.format(subject=subject)
        assert input_file.exists(), f"Input file {input_file} does not exist"
        output_file = args.output_folder / args.output_template.format(subject=subject)
        if output_file.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output file {output_file} already exists. "
                "Use --overwrite to overwrite it."
            )
        pbar.set_description(f"Processing {input_file.name}")
        if not args.average:
            process_subject(
                subject=subject,
                input_file=input_file,
                output_file=output_file,
            )
        else:
            process_subject_average(
                subject=subject,
                input_file=input_file,
                output_file=output_file,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Standardize embeddings")
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing the embeddings",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the folder to save the standardized embeddings",
    )
    parser.add_argument(
        "--input_template",
        type=str,
        default="*.npy",
        help="Template for input files",
    )

    parser.add_argument(
        "--output_template",
        type=str,
        help="Template for output files",
    )
    # List of subjects to process, if not provided process all subjects in the input folder
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        help="List of subject IDs to process",
    )
    parser.add_argument(
        "--average", action="store_true", help="Average embeddings across subjects"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    args = parser.parse_args()
    args.input_folder = Path(args.input_folder)
    args.output_folder = Path(args.output_folder)

    if args.subjects is None:
        df = get_resource("stimulus")
        subjects = df.subject.unique()
        subjects.sort()
        args.subjects = subjects.astype(int).tolist()

    return args


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    main()
