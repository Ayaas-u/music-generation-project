import argparse
import csv
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create a blind A/B manifest for post-RLHF human preference testing.")
    parser.add_argument("--num-pairs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/survey_results/ab_manifest.csv")
    args = parser.parse_args()

    random.seed(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(1, args.num_pairs + 1):
        baseline = f"transformer_sample_{i}.mp3"
        rlhf = f"rlhf_sample_{i}.mp3"
        if random.random() < 0.5:
            label_a, label_b = baseline, rlhf
            answer = "baseline"
        else:
            label_a, label_b = rlhf, baseline
            answer = "rlhf"
        rows.append({
            "pair_id": i,
            "audio_A": label_a,
            "audio_B": label_b,
            "preferred_model_answer_key": answer,
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
