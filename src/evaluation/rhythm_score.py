import numpy as np

from metrics import density, repetition_ratio, empty_step_ratio


def rhythm_score(
    sequence_6xT: np.ndarray,
    target_density: float = 0.17,
    density_weight: float = 0.4,
    repetition_weight: float = 0.4,
    emptiness_weight: float = 0.2,
) -> float:
    """
    Simple drum-focused rhythm quality score.

    Higher score is better.
    Rewards density near the target value and penalizes repetition
    and excessive empty steps.
    """
    d = density(sequence_6xT)
    r = repetition_ratio(sequence_6xT)
    e = empty_step_ratio(sequence_6xT)

    density_term = max(0.0, 1.0 - abs(d - target_density) / max(target_density, 1e-8))
    repetition_term = 1.0 - r
    emptiness_term = 1.0 - e

    score = (
        density_weight * density_term
        + repetition_weight * repetition_term
        + emptiness_weight * emptiness_term
    )
    return float(score)