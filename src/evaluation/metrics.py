import numpy as np


def density(sequence_6xT: np.ndarray) -> float:
    """
    Fraction of active drum hits in the 6 x T binary sequence.
    """
    sequence_6xT = np.asarray(sequence_6xT)
    return float(sequence_6xT.mean())


def repetition_ratio(sequence_6xT: np.ndarray, chunk_size: int = 16) -> float:
    """
    Measures how much a fixed-length rhythmic chunk repeats.
    Higher means more repetitive.
    """
    sequence_6xT = np.asarray(sequence_6xT)
    T = sequence_6xT.shape[1]
    patterns = []

    for start in range(0, T - chunk_size + 1, chunk_size):
        patt = tuple(
            sequence_6xT[:, start:start + chunk_size].astype(np.int8).flatten().tolist()
        )
        patterns.append(patt)

    if not patterns:
        return 1.0

    most_common = max(patterns.count(p) for p in set(patterns))
    return float(most_common / len(patterns))


def pairwise_diversity(sequences: list[np.ndarray]) -> float:
    """
    Average pairwise Hamming-style difference across a list of 6 x T sequences.
    Higher means more diverse outputs.
    """
    if len(sequences) < 2:
        return 0.0

    diffs = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            a = np.asarray(sequences[i])
            b = np.asarray(sequences[j])

            min_T = min(a.shape[1], b.shape[1])
            a = a[:, :min_T]
            b = b[:, :min_T]

            diffs.append(np.mean(a != b))

    return float(np.mean(diffs)) if diffs else 0.0


def empty_step_ratio(sequence_6xT: np.ndarray) -> float:
    """
    Fraction of timesteps where no drum is active.
    """
    sequence_6xT = np.asarray(sequence_6xT)
    empty_steps = np.sum(sequence_6xT.sum(axis=0) == 0)
    return float(empty_steps / sequence_6xT.shape[1])