from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm, trange
from fmri_mapping.io.nsd import get_resource, get_subject_roi
from fmri_mapping.embedding.ops import split_repetitions
from fmri_mapping.embedding.evaluation import accuracy_cosine_similarity, compute_rsa
import pandas as pd

# Sklearn ridge regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from dmf.alerts import alert, send_message

DATA_TEMPLATE = "../scripts/steps/subject-{subject:02d}_betas_pca-no-rel.npy"
MODELS_FOLDERS = "<paths-to-extracted-features>/features/nsd/all"

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_common_nsd_id_stimuli(min_reps: int = 3): # Gets the stimuli shared by all subjects
    df = get_resource("stimulus").query(
        "shared and exists and (repetition == @min_reps - 1)"
    )
    df_count_subjects = df.groupby("nsd_id").subject.count().reset_index()
    df_count_subjects = df_count_subjects[
        df_count_subjects.subject == 8
    ]  # All subjects

    return list(df_count_subjects.nsd_id.unique())


def process_subject_model_regression(
    subject: int,
    Y,
    test_stimulus: list[int],
    alpha: float = 1.0,
    return_transformed: bool = False,
    **kwargs,
):
    results = []
    df_reps = split_repetitions(subject=subject, shuffle_indexes=True, min_exists=2)
    df_reps_train = df_reps.query("shared == False")
    df_reps_test = df_reps.query("shared == True and nsd_id in @test_stimulus")
    X = np.load(DATA_TEMPLATE.format(subject=subject))

    df = get_resource("stimulus").query("subject == @subject and exists")
    train_mask = (df.shared == False).values
    nsd_ids = df.nsd_id.values[train_mask]

    X_train = X[train_mask]
    Y_train = Y[nsd_ids]  # n x d2

    X1_test = X[df_reps_test[f"subject_index_1"]]
    X2_test = X[df_reps_test[f"subject_index_2"]]
    X3_test = X[df_reps_test[f"subject_index_3"]]

    model = Ridge(alpha=alpha)
    model.fit(X_train, Y_train)

    Z_test_1 = model.predict(X1_test)
    Z_test_2 = model.predict(X2_test)
    Z_test_3 = model.predict(X3_test)

    Z_test_1 = torch.from_numpy(Z_test_1)
    Z_test_2 = torch.from_numpy(Z_test_2)
    Z_test_3 = torch.from_numpy(Z_test_3)

    test_views = ((1, Z_test_1), (2, Z_test_2), (3, Z_test_3))

    for view_a, Z_a in test_views:
        for view_b, Z_b in test_views:
            if view_a >= view_b:
                continue

            acc, cos, rank = accuracy_cosine_similarity(Z_a, Z_b)
            rsa = compute_rsa(Z_a, Z_b)

            results.append(
                {
                    "subject": subject,
                    "view_a": view_a,
                    "view_b": view_b,
                    "accuracy": acc,
                    "cosine": cos,
                    "mean_rank": rank,
                    "alpha": alpha,
                    "rsa": rsa,
                    "template": DATA_TEMPLATE,
                    **kwargs,
                }
            )

    if return_transformed:
        X_transormed_all = model.predict(X)
        return results, X_transormed_all

    return results, None


@alert(input=["model_path"])
def process_model(
    model_path: Path,
    test_stimulus: list[int],
    last_layer: bool = False,
    save_files: bool = True,
):
    pck = torch.load(model_path)
    features = pck["feats"].float()
    layers = features.shape[1]
    results = []

    for layer in trange(layers, desc="Processing layers", leave=False):
        if last_layer and layer != layers - 1:
            continue
        Y = features[:, layer, :]
        for subject in trange(1, 8 + 1, leave=False, desc=f"Processing subject"):
            r, X_transformed = process_subject_model_regression(
                subject=subject,
                Y=Y,
                test_stimulus=test_stimulus,
                model_name=model_path.stem,
                layer=layer,
                last_layer=(layer == layers - 1),
                return_transformed=save_files,
            )
            results.extend(r)
            if save_files and X_transformed is not None:
                folder = Path("features_linear")
                folder.mkdir(exist_ok=True)
                filename = f"{model_path.stem}_subject-{subject:02d}_layer-{layer:02d}_features.npy"
                np.save(folder / filename, X_transformed)

    return results


def main():
    test_stimulus = get_common_nsd_id_stimuli()
    models = list(Path(MODELS_FOLDERS).glob("*.pt"))
    # Only those that starts with vit_
    # models = [model for model in models if model.name.startswith("vit_")]
    # models = [model for model in models if "vit_large_patch16_224.augreg_in21k_pool-cls" in model.name]
    models = [model for model in models if "vit_" in model.name]

    results = []


    for model in (pbar := tqdm(models, desc="Processing models")):
        pbar.set_description(model.stem)
        r = process_model(
            model, test_stimulus=test_stimulus, last_layer=True, save_files=True
        )
        results.extend(r)

    df = pd.DataFrame(results)
    filename = "vision_results_models_regression-rel.parquet"
    df.to_parquet(filename)
    send_message("Finished processing models regression", attachment=filename)


if __name__ == "__main__":
    main()
