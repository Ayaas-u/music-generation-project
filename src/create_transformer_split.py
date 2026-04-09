import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def main():
    data_path = Path("data/groove_sequences.npy")
    split_dir = Path("data/train_test_split")
    split_dir.mkdir(parents=True, exist_ok=True)

    sequences = np.load(data_path)
    n_samples = len(sequences)
    indices = np.arange(n_samples)

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=42,
        shuffle=True,
    )

    np.save(split_dir / "train_idx.npy", train_idx)
    np.save(split_dir / "val_idx.npy", val_idx)
    np.save(split_dir / "test_idx.npy", test_idx)

    split_info = {
        "dataset_path": str(data_path),
        "total_samples": int(n_samples),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "split_type": "sequence_level_split",
        "random_state": 42,
        "split_ratio": "80/10/10",
        "note": (
            "The final processed dataset was already saved as pre-windowed sequences, "
            "so Task 3 uses a deterministic sequence-level split."
        ),
    }

    with open(split_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print("Split saved successfully.")
    print(f"Total samples: {n_samples}")
    print(f"Train: {len(train_idx)}")
    print(f"Val:   {len(val_idx)}")
    print(f"Test:  {len(test_idx)}")


if __name__ == "__main__":
    main()