import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rlhf_utils import (
    extract_feature_matrix,
    fit_ridge_regression,
    load_reward_model,
    predict_ridge,
    save_reward_model,
)


def load_rated_sequences(sample_ids, sample_dir: Path):
    sequences = []
    found_ids = []
    for sample_id in sample_ids:
        npy_path = sample_dir / f"{sample_id}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Missing rated sample sequence {npy_path}. "
                "Expected a .npy sequence file for every rated baseline sample."
            )
        sequences.append(np.load(npy_path, allow_pickle=True))
        found_ids.append(sample_id)
    return found_ids, sequences


def leave_one_out_predictions(X, y, l2):
    preds = []
    n = len(y)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model_i = fit_ridge_regression(X[mask], y[mask], l2=l2)
        pred_i = predict_ridge(model_i, X[i : i + 1])[0]
        preds.append(float(pred_i))
    return np.array(preds, dtype=np.float64)



def main():
    data_dir = Path("data")
    survey_dir = data_dir / "survey_results"
    agg_path = survey_dir / "aggregated_rewards.csv"
    if not agg_path.exists():
        raise FileNotFoundError(f"Missing {agg_path}. Run aggregate_human_rewards.py first.")

    agg = pd.read_csv(agg_path)
    sample_ids = agg["sample_id"].astype(str).tolist()
    sample_dir = data_dir / "generated_samples" / "transformer"

    sample_ids, sequences = load_rated_sequences(sample_ids, sample_dir)
    agg = agg.set_index("sample_id").loc[sample_ids].reset_index()

    X = extract_feature_matrix(sequences)
    y_raw = agg["mean_participant_norm_score"].to_numpy(dtype=np.float64)
    y_mean = float(y_raw.mean())
    y_std = float(y_raw.std())
    if y_std < 1e-8:
        y_std = 1.0
    y = (y_raw - y_mean) / y_std

    l2 = 3.0
    reward_model = fit_ridge_regression(X, y, l2=l2)
    reward_model["target_mean"] = y_mean
    reward_model["target_std"] = y_std
    reward_model["target_kind"] = "participant_normalized_reward_z"

    y_pred = predict_ridge(reward_model, X)
    loo_pred = leave_one_out_predictions(X, y, l2=l2)

    rmse_train = float(np.sqrt(np.mean((y_pred - y) ** 2)))
    rmse_loo = float(np.sqrt(np.mean((loo_pred - y) ** 2)))
    corr_train = float(np.corrcoef(y_pred, y)[0, 1]) if len(y) > 1 else 0.0
    corr_loo = float(np.corrcoef(loo_pred, y)[0, 1]) if len(y) > 1 else 0.0

    model_path = data_dir / "reward_model.npz"
    save_reward_model(model_path, reward_model)

    feature_table = pd.DataFrame(X, columns=reward_model["feature_names"])
    feature_table.insert(0, "sample_id", sample_ids)
    feature_table["target_reward_z"] = y
    feature_table["pred_reward_z_train"] = y_pred
    feature_table["pred_reward_z_loo"] = loo_pred
    feature_table.to_csv(survey_dir / "reward_model_training_table.csv", index=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(y, loo_pred)
    lim_min = min(float(y.min()), float(loo_pred.min())) - 0.25
    lim_max = max(float(y.max()), float(loo_pred.max())) + 0.25
    plt.plot([lim_min, lim_max], [lim_min, lim_max])
    plt.xlabel("True reward z-score")
    plt.ylabel("LOO predicted reward z-score")
    plt.title("Reward model leave-one-out fit")
    plt.tight_layout()
    plt.savefig(survey_dir / "reward_model_loo_scatter.png", dpi=200)
    plt.close()

    info = {
        "model_type": "ridge_regression",
        "inputs": reward_model["feature_names"],
        "target": "z-scored mean participant-normalized reward",
        "l2_regularization": l2,
        "num_rated_samples": int(len(sample_ids)),
        "train_rmse": rmse_train,
        "loo_rmse": rmse_loo,
        "train_correlation": corr_train,
        "loo_correlation": corr_loo,
        "reward_model_path": str(model_path),
    }
    with open(data_dir / "reward_model_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(pd.DataFrame({"sample_id": sample_ids, "target": y, "loo_pred": loo_pred}))
    print(f"Saved reward model to {model_path}")


if __name__ == "__main__":
    main()
