import numpy as np


DRUM_LABELS = ["Kick", "Snare", "HiHat", "Tom", "Crash", "Ride"]


def drum_hit_distribution(sequence_6xT: np.ndarray) -> dict[str, float]:
    """
    Drum-project adaptation of a pitch histogram.

    Instead of melodic pitch classes, this computes the normalized hit
    distribution across the 6 drum channels:
    Kick, Snare, HiHat, Tom, Crash, Ride.
    """
    sequence_6xT = np.asarray(sequence_6xT)
    counts = sequence_6xT.sum(axis=1).astype(float)
    total = counts.sum()

    if total == 0:
        return {label: 0.0 for label in DRUM_LABELS}

    return {
        label: float(count / total)
        for label, count in zip(DRUM_LABELS, counts)
    }


if __name__ == "__main__":
    print(
        "This file adapts pitch histogram analysis for drum generation by "
        "measuring hit distribution over the 6 drum channels."
    )