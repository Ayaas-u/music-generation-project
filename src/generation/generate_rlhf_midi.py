import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from rlhf_utils import (
    BOS_TOKEN,
    is_valid,
    load_reward_model,
    load_transformer_model,
    predict_reward_from_sequences,
    save_drum_midi,
    set_seed,
    tokens_to_sequence,
    validity_penalty,
)


def empty_bar_ratio(sequence_6xT, bar_size=16):
    T = sequence_6xT.shape[1]
    bars = 0
    empty = 0

    for start in range(0, T - bar_size + 1, bar_size):
        bar = sequence_6xT[:, start:start + bar_size]
        bars += 1
        if np.sum(bar) == 0:
            empty += 1

    if bars == 0:
        return 1.0
    return empty / bars


def snare_empty_bar_ratio(sequence_6xT, bar_size=16):
    snare = sequence_6xT[1]
    T = snare.shape[0]
    bars = 0
    empty = 0

    for start in range(0, T - bar_size + 1, bar_size):
        bar = snare[start:start + bar_size]
        bars += 1
        if np.sum(bar) == 0:
            empty += 1

    if bars == 0:
        return 1.0
    return empty / bars

def snare_repetition_ratio(sequence_6xT, bar_size=16):
    snare = sequence_6xT[1]
    T = snare.shape[0]
    patterns = []

    for start in range(0, T - bar_size + 1, bar_size):
        patt = tuple(snare[start:start + bar_size].astype(np.int8).tolist())
        patterns.append(patt)

    if not patterns:
        return 1.0

    most_common = max(patterns.count(p) for p in set(patterns))
    return most_common / len(patterns)


def passes_extra_filters(
    sequence_6xT,
    max_empty_bar_ratio=0.20,
    max_kick_empty_bar_ratio=0.35,
    max_kick_repetition_ratio=0.40,
):
    bar_empty = empty_bar_ratio(sequence_6xT)
    snare_empty = snare_empty_bar_ratio(sequence_6xT)
    snare_rep = snare_repetition_ratio(sequence_6xT)

    if bar_empty > max_empty_bar_ratio:
        return False
    if snare_empty > max_kick_empty_bar_ratio:
        return False
    if snare_rep > max_kick_repetition_ratio:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate and save the top RLHF drum samples.")
    parser.add_argument("--num-final-samples", type=int, default=10)
    parser.add_argument("--num-candidates", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--total-steps", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # New stricter filters
    parser.add_argument("--max-empty-bar-ratio", type=float, default=0.20)
    parser.add_argument("--max-kick-empty-bar-ratio", type=float, default=0.35)
    parser.add_argument("--max-kick-repetition-ratio", type=float, default=0.40)

    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path("data")
    out_dir = data_dir / "generated_samples" / "rlhf"
    out_dir.mkdir(parents=True, exist_ok=True)

    reward_model_path = data_dir / "reward_model.npz"
    if not reward_model_path.exists():
        raise FileNotFoundError(f"Missing {reward_model_path}. Run train_reward_model.py first.")
    reward_model = load_reward_model(reward_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_transformer_model(data_dir / "rlhf_model.pth", device=device, train=False)

    candidates = []
    attempts = 0

    while len(candidates) < args.num_candidates and attempts < args.max_attempts:
        attempts += 1
        start = torch.tensor([[BOS_TOKEN]], dtype=torch.long, device=device)

        with torch.no_grad():
            generated = model.generate(
                start,
                max_new_tokens=args.total_steps,
                temperature=args.temperature,
                top_k=args.top_k,
            )

        tokens = generated[0].detach().cpu().numpy()[1:]
        sequence = tokens_to_sequence(tokens)

        # Base validity from your existing pipeline
        valid_base = bool(is_valid(sequence))

        # New kick/empty-bar filters
        valid_extra = passes_extra_filters(
            sequence,
            max_empty_bar_ratio=args.max_empty_bar_ratio,
            max_kick_empty_bar_ratio=args.max_kick_empty_bar_ratio,
            max_kick_repetition_ratio=args.max_kick_repetition_ratio,
        )

        valid = valid_base and valid_extra
        if not valid:
            continue

        predicted_reward_z = float(
            predict_reward_from_sequences([sequence], reward_model, output_space="z")[0]
        )
        penalty = float(validity_penalty(sequence))

        # Extra stats for debugging / selection
        bar_empty = float(empty_bar_ratio(sequence))
        kick_empty = float(snare_empty_bar_ratio(sequence))
        kick_rep = float(snare_repetition_ratio(sequence))

        # Penalize repetitive kick / sparse bars in selection score too
        selection_score = (
            predicted_reward_z
            - 0.5 * penalty
            - 0.5 * bar_empty
            - 0.4 * kick_empty
            - 0.6 * kick_rep
        )

        candidates.append({
            "tokens": tokens,
            "sequence": sequence,
            "predicted_reward_z": predicted_reward_z,
            "validity_penalty": penalty,
            "empty_bar_ratio": bar_empty,
            "kick_empty_bar_ratio": kick_empty,
            "kick_repetition_ratio": kick_rep,
            "valid": valid,
            "selection_score": selection_score,
        })

    if len(candidates) < args.num_final_samples:
        raise RuntimeError(
            f"Only found {len(candidates)} valid candidates after {attempts} attempts. "
            f"Try increasing --max-attempts or loosening the extra filters slightly."
        )

    candidates.sort(key=lambda x: x["selection_score"], reverse=True)
    selected = candidates[: args.num_final_samples]

    metadata_rows = []
    for i, item in enumerate(selected, start=1):
        midi_path = out_dir / f"rlhf_sample_{i}.mid"
        seq_path = out_dir / f"rlhf_sample_{i}.npy"
        token_path = out_dir / f"rlhf_sample_{i}_tokens.npy"

        np.save(seq_path, item["sequence"])
        np.save(token_path, item["tokens"])
        save_drum_midi(item["sequence"], midi_path, tempo=120, velocity=115, step_duration=0.125)

        metadata_rows.append({
            "sample_id": f"rlhf_sample_{i}",
            "midi_file": str(midi_path),
            "sequence_file": str(seq_path),
            "token_file": str(token_path),
            "predicted_reward_z": item["predicted_reward_z"],
            "validity_penalty": item["validity_penalty"],
            "empty_bar_ratio": item["empty_bar_ratio"],
            "kick_empty_bar_ratio": item["kick_empty_bar_ratio"],
            "kick_repetition_ratio": item["kick_repetition_ratio"],
            "selection_score": item["selection_score"],
            "valid": item["valid"],
        })
        print(f"Saved {midi_path}")

    with open(out_dir / "rlhf_generation_candidates.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    summary = {
        "num_candidates_generated": len(candidates),
        "num_selected": len(selected),
        "num_attempts": attempts,
        "generation_steps": args.total_steps,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_empty_bar_ratio": args.max_empty_bar_ratio,
        "max_kick_empty_bar_ratio": args.max_kick_empty_bar_ratio,
        "max_kick_repetition_ratio": args.max_kick_repetition_ratio,
        "source_checkpoint_keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else [],
    }

    with open(data_dir / "rlhf_generation_info.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()