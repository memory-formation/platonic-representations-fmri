"""
Test DCCA within reps
"""

from pathlib import Path
import torch
from mvlearn.embed import DCCA
import numpy as np
from tqdm import tqdm
from fmri_mapping.io.nsd import get_resource, get_subject_roi
from fmri_mapping.embedding.ops import split_repetitions
from fmri_mapping.embedding.evaluation import accuracy_cosine_similarity, compute_rsa
import pandas as pd

DATA_TEMPLATE = "../scripts/steps/subject-{subject:02d}_betas_pca-no-rel.npy"


def process_subject(
    subject: int,
    view_1: int = 1,
    view_2: int = 2,
    view_inpute: int = 3,
    test_stimulus: list[int] = None,
    n_components: int = 128,
    layers: list[int] = [128, 128],
    max_iter: int = 1000,
    print_train_log_info: bool = True,
    **kwargs,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load data
    df_reps = split_repetitions(subject=subject, shuffle_indexes=True, min_exists=2)
    df_reps_train = df_reps.query("shared == False")
    df_reps_test = df_reps.query("shared == True and nsd_id in @test_stimulus")
    X = np.load(DATA_TEMPLATE.format(subject=subject))

    subject_index_1 = df_reps_train[f"subject_index_{view_1}"]
    subject_index_2 = df_reps_train[f"subject_index_{view_2}"]
    subject_index_inpute = df_reps_train[f"subject_index_{view_inpute}"]
    subject_index_1 = subject_index_1.fillna(subject_index_inpute).astype(int)
    subject_index_2 = subject_index_2.fillna(subject_index_inpute).astype(int)

    X_train_1 = X[subject_index_1]
    X_train_2 = X[subject_index_2]
    Xs_train = [X_train_1, X_train_2]

    # Fit DCCA
    dcca = DCCA(
        input_size1=X_train_1.shape[1],
        input_size2=X_train_2.shape[1],
        n_components=n_components,
        layer_sizes1=layers,
        layer_sizes2=layers,
        epoch_num=max_iter,
        device=device,
        print_train_log_info=print_train_log_info,
        **kwargs,
    )
    

    dcca.fit(Xs_train)

    X_transformed = dcca.transform([X, X])
    X_transformed = (X_transformed[0] + X_transformed[1]) / 2

    X_test_1_transformed = X_transformed[
        df_reps_test[f"subject_index_{view_1}"].astype(int)
    ]
    X_test_2_transformed = X_transformed[
        df_reps_test[f"subject_index_{view_2}"].astype(int)
    ]
    X_test_3_transformed = X_transformed[
        df_reps_test[f"subject_index_{view_inpute}"].astype(int)
    ]

    evaluation = []
    test_views = [
        (view_1, torch.tensor(X_test_1_transformed).cuda()),
        (view_2, torch.tensor(X_test_2_transformed).cuda()),
        (view_inpute, torch.tensor(X_test_3_transformed).cuda()),
    ]
    # Iterate all pairs of 2 views
    for view_a, X_test_a in test_views:
        for view_b, X_test_b in test_views:
            if view_a >= view_b:
                continue

            acc, cos, rank = accuracy_cosine_similarity(X_test_a, X_test_b)
            rsa = compute_rsa(X_test_a, X_test_b)
            evaluation.append(
                {
                    "subject": subject,
                    "view_a": view_a,
                    "view_b": view_b,
                    "accuracy": acc,
                    "cosine": cos,
                    "mean_rank": rank,
                    "rsa": rsa,
                    "n_components": n_components,
                    "layers": str(layers),
                    "max_iter": max_iter,
                    "data_template": DATA_TEMPLATE,
                }
            )
            print(evaluation[-1])

    return X_transformed, evaluation


def get_common_nsd_id_stimuli(min_reps: int = 3):
    df = get_resource("stimulus").query(
        "shared and exists and (repetition == @min_reps - 1)"
    )
    df_count_subjects = df.groupby("nsd_id").subject.count().reset_index()
    df_count_subjects = df_count_subjects[
        df_count_subjects.subject == 8
    ]  # All subjects

    return list(df_count_subjects.nsd_id.unique())


from dmf.alerts import alert, send_message


@alert
def main():
    test_stimulus = get_common_nsd_id_stimuli()

    n_components = 128
    folder = Path("dcca_results")
    folder.mkdir(exist_ok=True)

    views = [(1, 2, 3), (1, 3, 2), (2, 3, 1)]
    layers_combs = [[128], [256], [512], [128, 128]]
    max_iter = 1000
    results = []

    pbar = tqdm(total=8 * len(views) * len(layers_combs), desc="Total progress")
    for layers in layers_combs:
        layers_name = "-".join(map(str, layers))
        for view_1, view_2, view_inpute in views:
            views_name = f"{view_1}-{view_2}"
            for subject in range(1, 8 + 1):
                pbar.set_description(
                    f"sub-{subject}, views {views_name}, layers {layers_name}"
                )
                send_message(
                    f"DCCA Processing subject {subject}, views {views_name} with layers {layers_name}"
                )
                try:
                    X, evaluation = process_subject(
                        subject=subject,
                        test_stimulus=test_stimulus,
                        view_1=view_1,
                        view_2=view_2,
                        view_inpute=view_inpute,
                        n_components=n_components,
                        layers=layers,
                        max_iter=max_iter,
                    )
                    results.extend(evaluation)
                    save_file = (
                        folder
                        / f"subject-{subject:02d}_views-{views_name}_layers-{layers_name}_components-{n_components}.npz"
                    )
                    np.save(save_file, X)
                    # Clear GPU memory
                    del X
                    torch.cuda.empty_cache()
                except Exception as e:
                    # If is a keyboard interrupt, raise it to stop the process
                    if isinstance(e, KeyboardInterrupt):
                        raise e
                    send_message(
                        f"Error processing subject {subject}, views {views_name} with layers {layers_name}: {e}"
                    )
                    
                pbar.update(1)

    df = pd.DataFrame(results)
    df.to_parquet(folder / f"evaluation_dcca_components-{n_components}.parquet")


if __name__ == "__main__":
    main()
