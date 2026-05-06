"""
Test DCCA within reps
"""

from pathlib import Path
import torch
from mvlearn.embed import SplitAE
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
    max_iter: int = 1000,
    hidden_size: int = 512,
    num_hidden_layers: int = 2,
    batch_size: int = 256,
    print_train_log_info: bool = False,
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
    dcca = SplitAE(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        embed_size=n_components,
        training_epochs=max_iter,
        batch_size=batch_size,
        print_info=print_train_log_info,
        print_graph=False,
    )

    dcca.fit(Xs_train)
    X_transformed = dcca.transform([X])

    X_transformed = X_transformed[0]

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
                    "hidden_size": hidden_size,
                    "num_hidden_layers": num_hidden_layers,
                    "batch_size": batch_size,
                    "max_iter": max_iter,
                    "data_template": DATA_TEMPLATE,
                }
            )
            print(evaluation)

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
    batch_size = 512
    folder = Path("splitae_results")
    folder.mkdir(exist_ok=True)

    views = [(1, 2, 3), (1, 3, 2), (2, 3, 1)]
    hidden_sizes = [512, 468]  # You can add more hidden sizes to test
    num_hidden_layers_list = [1, 2]  # You can add more layer numbers to test
    max_iter = 3000
    results = []
    layers_name = ""

    pbar = tqdm(
        total=8 * len(views) * len(hidden_sizes) * len(num_hidden_layers_list),
        desc="Total progress",
    )
    for hidden_size in hidden_sizes:
        for num_hidden_layers in num_hidden_layers_list:

            for view_1, view_2, view_inpute in views:
                views_name = f"{view_1}-{view_2}"
                for subject in range(1, 8 + 1):
                    pbar.set_description(
                        f"sub-{subject}, views {views_name}, hidden_size {hidden_size}, num_hidden_layers {num_hidden_layers}"
                    )

                    try:
                        X, evaluation = process_subject(
                            subject=subject,
                            test_stimulus=test_stimulus,
                            view_1=view_1,
                            view_2=view_2,
                            view_inpute=view_inpute,
                            n_components=n_components,
                            hidden_size=hidden_size,
                            num_hidden_layers=num_hidden_layers,
                            batch_size=batch_size,
                            max_iter=max_iter,
                        )

                        save_file = (
                            folder
                            / f"subject-{subject:02d}_views-{views_name}_components-{n_components}_hidden-{hidden_size}_num-layers-{num_hidden_layers}_batch-{batch_size}_iter-{max_iter}.npz"
                        )
                        results.extend(evaluation)
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
    df.to_parquet(folder / f"evaluation_splitae_components-{n_components}.parquet")


if __name__ == "__main__":
    main()
