import json
from pathlib import Path

import numpy as np


def compute_density(samples):
    return float(np.mean(samples))


def compute_diversity(samples):
    patterns = []
    for seq in samples:
        for start in range(0, seq.shape[1] - 15, 16):
            patt = seq[:, start:start+16].astype(np.int8).flatten()
            patterns.append(tuple(patt.tolist()))
    if not patterns:
        return 0.0
    return float(len(set(patterns)) / len(patterns))


def compute_repetition(samples):
    patterns = []
    for seq in samples:
        for start in range(0, seq.shape[1] - 15, 16):
            patt = seq[:, start:start+16].astype(np.int8).flatten()
            patterns.append(tuple(patt.tolist()))
    if not patterns:
        return 0.0
    repeated = len(patterns) - len(set(patterns))
    return float(repeated / len(patterns))


def main():
    data_dir = Path("data")
    seqs = np.load(data_dir / "transformer_generated_sequences.npy", allow_pickle=True)

    chunks = []
    for seq in seqs:
        T = seq.shape[1]
        for start in range(0, T, 128):
            chunk = seq[:, start:start+128]
            if chunk.shape[1] == 128:
                chunks.append(chunk)

    chunks = np.array(chunks, dtype=np.float32)

    results = {
        "num_original_sequences": int(len(seqs)),
        "num_128_chunks_used": int(len(chunks)),
        "density": compute_density(chunks),
        "diversity": compute_diversity(chunks),
        "repetition": compute_repetition(chunks),
    }

    with open(data_dir / "transformer_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()