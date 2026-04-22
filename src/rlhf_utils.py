import json
import random
from pathlib import Path

import numpy as np
import pretty_midi
import torch

from transformer import DrumTransformer

try:
    from transformer_tokenizer import tokens_to_sequence as _tokens_to_sequence
except Exception:
    _tokens_to_sequence = None

VOCAB_SIZE = 66
BLOCK_SIZE = 128
BOS_TOKEN = 64
EOS_TOKEN = 65

DRUM_NOTES = {
    0: 36,  # kick
    1: 38,  # snare
    2: 42,  # hihat
    3: 45,  # tom
    4: 49,  # crash
    5: 51,  # ride
}

FEATURE_NAMES = [
    "density",
    "empty_step_ratio",
    "repetition_16",
    "kick_rate",
    "snare_rate",
    "hihat_rate",
    "tom_rate",
    "crash_rate",
    "ride_rate",
    "simultaneous_hits_ratio",
    "pattern_change_rate",
    "strong_beat_density",
    "offbeat_density",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_sequence_shape(sequence_6xT: np.ndarray) -> np.ndarray:
    seq = np.asarray(sequence_6xT)
    if seq.ndim != 2:
        raise ValueError(f"Expected 2D sequence array, got shape {seq.shape}")
    if seq.shape[0] == 6:
        return seq.astype(np.float32)
    if seq.shape[1] == 6:
        return seq.T.astype(np.float32)
    raise ValueError(f"Expected shape (6, T) or (T, 6), got {seq.shape}")


def fallback_tokens_to_sequence(tokens: np.ndarray) -> np.ndarray:
    tokens = np.asarray(tokens).astype(np.int64)
    valid = tokens[(tokens >= 0) & (tokens < 64)]
    seq = np.zeros((6, len(valid)), dtype=np.float32)
    for t, token in enumerate(valid):
        for ch in range(6):
            seq[ch, t] = float((int(token) >> ch) & 1)
    return seq


def tokens_to_sequence(tokens: np.ndarray) -> np.ndarray:
    if _tokens_to_sequence is not None:
        try:
            seq = _tokens_to_sequence(tokens)
            return ensure_sequence_shape(seq)
        except Exception:
            pass
    return fallback_tokens_to_sequence(tokens)


def save_drum_midi(sequence_6xT: np.ndarray, out_path, tempo: int = 120, velocity: int = 115, step_duration: float = 0.125) -> None:
    sequence_6xT = ensure_sequence_shape(sequence_6xT)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    drum = pretty_midi.Instrument(program=0, is_drum=True)

    num_drums, T = sequence_6xT.shape
    if num_drums != 6:
        raise ValueError(f"Expected 6 drum channels, got {num_drums}")

    for t in range(T):
        start = t * step_duration
        end = start + step_duration * 0.9
        for drum_idx in range(6):
            if sequence_6xT[drum_idx, t] > 0.5:
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=DRUM_NOTES[drum_idx],
                    start=start,
                    end=end,
                )
                drum.notes.append(note)

    midi.instruments.append(drum)
    midi.write(str(out_path))


def density(sequence_6xT: np.ndarray) -> float:
    seq = ensure_sequence_shape(sequence_6xT)
    return float(seq.mean())


def empty_step_ratio(sequence_6xT: np.ndarray) -> float:
    seq = ensure_sequence_shape(sequence_6xT)
    empty_steps = np.sum(seq.sum(axis=0) == 0)
    return float(empty_steps / seq.shape[1])


def repetitive_pattern_ratio(sequence_6xT: np.ndarray, chunk_size: int = 16) -> float:
    seq = ensure_sequence_shape(sequence_6xT)
    patterns = []
    T = seq.shape[1]
    for start in range(0, T - chunk_size + 1, chunk_size):
        patt = tuple(seq[:, start:start + chunk_size].astype(np.int8).flatten().tolist())
        patterns.append(patt)
    if not patterns:
        return 1.0
    most_common = max(patterns.count(p) for p in set(patterns))
    return float(most_common / len(patterns))


def simultaneous_hits_ratio(sequence_6xT: np.ndarray) -> float:
    seq = ensure_sequence_shape(sequence_6xT)
    return float(np.mean(seq.sum(axis=0) >= 2))


def pattern_change_rate(sequence_6xT: np.ndarray) -> float:
    seq = ensure_sequence_shape(sequence_6xT)
    if seq.shape[1] <= 1:
        return 0.0
    changes = np.any(seq[:, 1:] != seq[:, :-1], axis=0)
    return float(np.mean(changes))


def strong_beat_density(sequence_6xT: np.ndarray) -> float:
    seq = ensure_sequence_shape(sequence_6xT)
    strong_steps = [i for i in range(seq.shape[1]) if (i % 16) in (0, 4, 8, 12)]
    if not strong_steps:
        return 0.0
    return float(seq[:, strong_steps].mean())


def offbeat_density(sequence_6xT: np.ndarray) -> float:
    seq = ensure_sequence_shape(sequence_6xT)
    off_steps = [i for i in range(seq.shape[1]) if (i % 16) in (2, 6, 10, 14)]
    if not off_steps:
        return 0.0
    return float(seq[:, off_steps].mean())


def extract_reward_features(sequence_6xT: np.ndarray) -> np.ndarray:
    seq = ensure_sequence_shape(sequence_6xT)
    channel_rates = seq.mean(axis=1)
    feats = np.array([
        density(seq),
        empty_step_ratio(seq),
        repetitive_pattern_ratio(seq, chunk_size=16),
        float(channel_rates[0]),
        float(channel_rates[1]),
        float(channel_rates[2]),
        float(channel_rates[3]),
        float(channel_rates[4]),
        float(channel_rates[5]),
        simultaneous_hits_ratio(seq),
        pattern_change_rate(seq),
        strong_beat_density(seq),
        offbeat_density(seq),
    ], dtype=np.float32)
    return feats


def extract_feature_matrix(sequences) -> np.ndarray:
    return np.stack([extract_reward_features(seq) for seq in sequences], axis=0)


def fit_ridge_regression(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> dict:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    Xs = (X - x_mean) / x_std

    y_mean = float(y.mean())
    yc = y - y_mean

    A = Xs.T @ Xs + float(l2) * np.eye(Xs.shape[1], dtype=np.float64)
    b = Xs.T @ yc
    w = np.linalg.solve(A, b)

    return {
        "weights": w.astype(np.float64),
        "bias": y_mean,
        "x_mean": x_mean.astype(np.float64),
        "x_std": x_std.astype(np.float64),
        "l2": float(l2),
        "feature_names": FEATURE_NAMES,
    }


def predict_ridge(model: dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Xs = (X - model["x_mean"]) / model["x_std"]
    return (Xs @ model["weights"] + model["bias"]).astype(np.float64)


def save_reward_model(model_path, model_dict: dict) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        model_path,
        weights=model_dict["weights"],
        bias=np.array([model_dict["bias"]], dtype=np.float64),
        x_mean=model_dict["x_mean"],
        x_std=model_dict["x_std"],
        l2=np.array([model_dict["l2"]], dtype=np.float64),
        feature_names=np.array(model_dict["feature_names"], dtype=object),
        target_mean=np.array([model_dict.get("target_mean", 0.0)], dtype=np.float64),
        target_std=np.array([model_dict.get("target_std", 1.0)], dtype=np.float64),
        target_kind=np.array([model_dict.get("target_kind", "reward_z")], dtype=object),
    )


def load_reward_model(model_path) -> dict:
    payload = np.load(model_path, allow_pickle=True)
    return {
        "weights": payload["weights"].astype(np.float64),
        "bias": float(payload["bias"][0]),
        "x_mean": payload["x_mean"].astype(np.float64),
        "x_std": payload["x_std"].astype(np.float64),
        "l2": float(payload["l2"][0]),
        "feature_names": [str(x) for x in payload["feature_names"].tolist()],
        "target_mean": float(payload["target_mean"][0]),
        "target_std": float(payload["target_std"][0]),
        "target_kind": str(payload["target_kind"][0]),
    }


def predict_reward_from_sequences(sequences, reward_model: dict, output_space: str = "z") -> np.ndarray:
    X = extract_feature_matrix(sequences)
    pred = predict_ridge(reward_model, X)
    if output_space == "raw":
        return pred * reward_model.get("target_std", 1.0) + reward_model.get("target_mean", 0.0)
    return pred


def validity_penalty(sequence_6xT: np.ndarray, min_density: float = 0.10, max_density: float = 0.28) -> float:
    seq = ensure_sequence_shape(sequence_6xT)
    d = density(seq)
    e = empty_step_ratio(seq)
    r = repetitive_pattern_ratio(seq)

    penalty = 0.0
    if d < min_density:
        penalty += (min_density - d) / max(min_density, 1e-8)
    if d > max_density:
        penalty += (d - max_density) / max(max_density, 1e-8)
    if e > 0.45:
        penalty += (e - 0.45) / 0.55
    if r > 0.50:
        penalty += (r - 0.50) / 0.50
    return float(max(0.0, penalty))


def is_valid(sequence_6xT: np.ndarray, min_density: float = 0.10, max_density: float = 0.28) -> bool:
    return validity_penalty(sequence_6xT, min_density=min_density, max_density=max_density) <= 1e-8


def load_transformer_model(checkpoint_path, device, train: bool = False):
    model = DrumTransformer(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    if train:
        model.train()
    else:
        model.eval()
    return model, checkpoint


def write_json(path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
