import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Import your existing conversion code here
# Adjust this import to match your project
from midi_to_pianoroll import midi_to_drum_pianoroll


DATA_DIR = Path("data/groove")
OUT_DIR = Path("data")
SEQ_LEN = 128
RANDOM_SEED = 42


def segment_sequence(arr, seq_len=128):
    """
    arr shape: (6, T)
    returns: list of windows, each shape (6, seq_len)
    """
    windows = []
    total_steps = arr.shape[1]
    for start in range(0, total_steps - seq_len + 1, seq_len):
        window = arr[:, start:start + seq_len]
        if window.shape[1] == seq_len:
            windows.append(window.astype(np.uint8))
    return windows


def collect_midi_files(root):
    return sorted([p for p in root.rglob("*.mid")])


def main():
    midi_files = collect_midi_files(DATA_DIR)
    print(f"Found {len(midi_files)} MIDI files")

    all_sequences = []
    all_sources = []

    usable_files = 0

    for midi_path in midi_files:
        try:
            # Must return shape (6, T)
            roll = midi_to_drum_pianoroll(str(midi_path))

            if roll is None:
                continue

            if roll.ndim != 2 or roll.shape[0] != 6:
                continue

            windows = segment_sequence(roll, seq_len=SEQ_LEN)
            if len(windows) == 0:
                continue

            usable_files += 1
            for w in windows:
                all_sequences.append(w)
                all_sources.append(str(midi_path))

        except Exception as e:
            print(f"Skipping {midi_path}: {e}")

    sequences = np.stack(all_sequences)  # (N, 6, 128)
    sources = np.array(all_sources)

    print(f"Usable files: {usable_files}")
    print(f"Sequences shape: {sequences.shape}")

    np.save(OUT_DIR / "groove_sequences_with_sources.npy", sequences)
    np.save(OUT_DIR / "groove_sequence_sources.npy", sources)

    with open(OUT_DIR / "groove_dataset_meta.json", "w") as f:
        json.dump({
            "num_midi_files_found": len(midi_files),
            "num_usable_files": usable_files,
            "num_sequences": int(sequences.shape[0]),
            "sequence_shape": list(sequences.shape[1:]),
            "seq_len": SEQ_LEN
        }, f, indent=2)

    print("Saved dataset with source tracking.")


if __name__ == "__main__":
    main()