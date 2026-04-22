import json
from pathlib import Path

import numpy as np
import pandas as pd

from rlhf_utils import (
    density,
    empty_step_ratio,
    load_reward_model,
    predict_reward_from_sequences,
    repetitive_pattern_ratio,
)



def load_sequence_set(directory: Path, prefix: str, n: int = 10):
    sequences = []
    ids = []
    for i in range(1, n + 1):
        path = directory / f"{prefix}_{i}.npy"
        if path.exists():
            sequences.append(np.load(path, allow_pickle=True))
            ids.append(f"{prefix}_{i}")
    if not sequences:
        raise FileNotFoundError(f"No files found for {prefix}_*.npy in {directory}")
    return ids, sequences



def diversity_pairwise(sequences) -> float:
    flat = [np.asarray(seq).astype(np.float32).flatten() for seq in sequences]
    if len(flat) <= 1:
        return 0.0
    dists = []
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            dists.append(float(np.mean(flat[i] != flat[j])))
    return float(np.mean(dists))



def summarize_set(model_name: str, sequences, reward_model=None):
    row = {
        "model": model_name,
        "num_samples": len(sequences),
        "density": float(np.mean([density(s) for s in sequences])),
        "empty_step_ratio": float(np.mean([empty_step_ratio(s) for s in sequences])),
        "repetition_ratio": float(np.mean([repetitive_pattern_ratio(s) for s in sequences])),
        "diversity_pairwise": diversity_pairwise(sequences),
    }
    if reward_model is not None:
        row["predicted_reward_z"] = float(np.mean(predict_reward_from_sequences(sequences, reward_model, output_space="z")))
    return row



def aggregate_human_csv(path: Path):
    df = pd.read_csv(path)
    base_cols = ["groove_quality", "coherence", "variety"]
    for col in base_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    numeric_cols = base_cols.copy()
    if "overall_preference" in df.columns:
        df["overall_preference"] = pd.to_numeric(df["overall_preference"], errors="coerce")
        numeric_cols.append("overall_preference")
    df = df.dropna(subset=numeric_cols).copy()

    if "overall_preference" in df.columns:
        df["raw_reward"] = 0.2 * df["groove_quality"] + 0.2 * df["coherence"] + 0.2 * df["variety"] + 0.4 * df["overall_preference"]
    else:
        df["raw_reward"] = (df["groove_quality"] + df["coherence"] + df["variety"]) / 3.0

    summary = {
        "participants": int(df["participant_id"].nunique()),
        "samples": int(df["sample_id"].nunique()),
        "mean_groove": float(df["groove_quality"].mean()),
        "mean_coherence": float(df["coherence"].mean()),
        "mean_variety": float(df["variety"].mean()),
        "mean_raw_reward": float(df["raw_reward"].mean()),
    }
    if "overall_preference" in df.columns:
        summary["mean_overall_preference"] = float(df["overall_preference"].mean())
    return summary



def main():
    data_dir = Path("data")
    survey_dir = data_dir / "survey_results"
    baseline_dir = data_dir / "generated_samples" / "transformer"
    rlhf_dir = data_dir / "generated_samples" / "rlhf"

    reward_model = None
    reward_model_path = data_dir / "reward_model.npz"
    if reward_model_path.exists():
        reward_model = load_reward_model(reward_model_path)

    _, baseline_sequences = load_sequence_set(baseline_dir, "transformer_sample", n=10)
    _, rlhf_sequences = load_sequence_set(rlhf_dir, "rlhf_sample", n=10)

    rows = [
        summarize_set("Transformer (before RLHF)", baseline_sequences, reward_model=reward_model),
        summarize_set("RLHF-tuned Transformer", rlhf_sequences, reward_model=reward_model),
    ]
    comparison_df = pd.DataFrame(rows)

    pre_path = survey_dir / "human_ratings.csv"
    post_path = survey_dir / "human_ratings_post_rlhf.csv"
    ab_path = survey_dir / "ab_preferences.csv"

    summary = {
        "objective_metrics": comparison_df.to_dict(orient="records"),
    }

    if pre_path.exists():
        summary["pre_rlhf_human_summary"] = aggregate_human_csv(pre_path)
    if post_path.exists():
        summary["post_rlhf_human_summary"] = aggregate_human_csv(post_path)
    if ab_path.exists():
        ab_df = pd.read_csv(ab_path)
        if "preferred_model" in ab_df.columns:
            preferred = ab_df["preferred_model"].astype(str).str.lower().str.strip()
            summary["ab_preference"] = {
                "num_votes": int(len(preferred)),
                "rlhf_win_rate": float(np.mean(preferred == "rlhf")),
                "baseline_win_rate": float(np.mean(preferred == "baseline")),
                "tie_rate": float(np.mean(preferred == "tie")),
            }

    out_csv = survey_dir / "before_after_comparison.csv"
    comparison_df.to_csv(out_csv, index=False)
    with open(survey_dir / "before_after_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(comparison_df)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
