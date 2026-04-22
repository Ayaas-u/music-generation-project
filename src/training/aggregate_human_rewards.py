import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_RATING_COLUMNS = ["groove_quality", "coherence", "variety"]
OPTIONAL_RATING_COLUMNS = ["overall_preference"]



def compute_raw_reward(df: pd.DataFrame):
    if "overall_preference" in df.columns:
        formula = "0.2*groove_quality + 0.2*coherence + 0.2*variety + 0.4*overall_preference"
        raw_reward = (
            0.2 * df["groove_quality"]
            + 0.2 * df["coherence"]
            + 0.2 * df["variety"]
            + 0.4 * df["overall_preference"]
        )
    else:
        formula = "(groove_quality + coherence + variety) / 3"
        raw_reward = (df["groove_quality"] + 0.1* df["coherence"] + 0.2*df["variety"]) / 3.0
    return raw_reward, formula



def main():
    survey_dir = Path("data") / "survey_results"
    ratings_path = survey_dir / "human_ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing {ratings_path}. Run prepare_google_form_ratings.py or place a clean human_ratings.csv there.")

    df = pd.read_csv(ratings_path)
    required = ["participant_id", "sample_id", *BASE_RATING_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in human_ratings.csv: {missing}")

    numeric_cols = BASE_RATING_COLUMNS.copy()
    if "overall_preference" in df.columns:
        numeric_cols.append("overall_preference")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols).copy()
    df[numeric_cols] = df[numeric_cols].clip(lower=1, upper=5)
    if "comment" not in df.columns:
        df["comment"] = ""

    df["raw_reward"], reward_formula = compute_raw_reward(df)

    participant_stats = (
        df.groupby("participant_id", as_index=False)
        .agg(participant_mean_raw_reward=("raw_reward", "mean"), participant_std_raw_reward=("raw_reward", "std"))
    )
    participant_stats["participant_std_raw_reward"] = participant_stats["participant_std_raw_reward"].fillna(0.0)

    df = df.merge(participant_stats, on="participant_id", how="left")
    df["participant_norm_reward"] = np.where(
        df["participant_std_raw_reward"] > 1e-8,
        (df["raw_reward"] - df["participant_mean_raw_reward"]) / df["participant_std_raw_reward"],
        0.0,
    )

    agg_spec = {
        "num_raters": ("participant_id", "nunique"),
        "mean_raw_reward": ("raw_reward", "mean"),
        "std_raw_reward": ("raw_reward", "std"),
        "mean_participant_norm_score": ("participant_norm_reward", "mean"),
        "std_participant_norm_score": ("participant_norm_reward", "std"),
        "groove_quality_mean": ("groove_quality", "mean"),
        "coherence_mean": ("coherence", "mean"),
        "variety_mean": ("variety", "mean"),
    }
    if "overall_preference" in df.columns:
        agg_spec["overall_preference_mean"] = ("overall_preference", "mean")

    agg = (
        df.groupby("sample_id", as_index=False)
        .agg(**agg_spec)
        .sort_values("mean_participant_norm_score", ascending=False)
    )

    agg["std_raw_reward"] = agg["std_raw_reward"].fillna(0.0)
    agg["std_participant_norm_score"] = agg["std_participant_norm_score"].fillna(0.0)

    min_score = float(agg["mean_participant_norm_score"].min())
    max_score = float(agg["mean_participant_norm_score"].max())
    if max_score > min_score:
        agg["reward_01"] = (agg["mean_participant_norm_score"] - min_score) / (max_score - min_score)
    else:
        agg["reward_01"] = 0.5

    survey_dir.mkdir(parents=True, exist_ok=True)
    participant_scores_path = survey_dir / "participant_normalized_scores.csv"
    aggregated_path = survey_dir / "aggregated_rewards.csv"
    df.to_csv(participant_scores_path, index=False)
    agg.to_csv(aggregated_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(agg["sample_id"], agg["mean_participant_norm_score"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean participant-normalized reward")
    plt.title("Pre-RLHF human reward by sample")
    plt.tight_layout()
    plt.savefig(survey_dir / "pre_rlhf_reward_barplot.png", dpi=200)
    plt.close()

    summary = {
        "num_participants": int(df["participant_id"].nunique()),
        "num_samples": int(df["sample_id"].nunique()),
        "num_rows": int(len(df)),
        "reward_formula": reward_formula,
        "participant_normalization": "z-score per participant over raw_reward",
        "aggregated_rewards_csv": str(aggregated_path),
        "participant_normalized_scores_csv": str(participant_scores_path),
    }
    with open(survey_dir / "reward_aggregation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(agg)
    print(f"Saved {aggregated_path}")


if __name__ == "__main__":
    main()
