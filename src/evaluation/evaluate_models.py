import os
import json
import numpy as np


def load_dataset(npy_path):
    data = np.load(npy_path)
    print(f"Loaded {npy_path} with shape: {data.shape}")
    return data


def standardize_shape(data):
    if len(data.shape) != 3:
        raise ValueError(f"Expected 3D array, got shape {data.shape}")

    if data.shape[2] == 6:
        return data

    if data.shape[1] == 6:
        return np.transpose(data, (0, 2, 1))

    raise ValueError(f"Could not identify channel dimension in shape {data.shape}")


def binarize(data, threshold=0.5):
    return (data > threshold).astype(np.int32)


def note_density(sequence):
    return np.sum(sequence) / sequence.size


def rhythm_diversity(sequence):
    step_patterns = [tuple(step.tolist()) for step in sequence]
    unique_patterns = len(set(step_patterns))
    return unique_patterns / len(step_patterns)


def repetition_ratio(sequence, pattern_len=2):
    if len(sequence) < pattern_len:
        return 0.0

    patterns = []
    for i in range(len(sequence) - pattern_len + 1):
        pattern = tuple(sequence[i:i + pattern_len].flatten().tolist())
        patterns.append(pattern)

    total_patterns = len(patterns)
    unique_patterns = len(set(patterns))
    repeated_patterns = total_patterns - unique_patterns

    return repeated_patterns / total_patterns if total_patterns > 0 else 0.0


def evaluate_dataset(data, name="Model"):
    densities = []
    diversities = []
    repetitions = []

    for seq in data:
        densities.append(note_density(seq))
        diversities.append(rhythm_diversity(seq))
        repetitions.append(repetition_ratio(seq))

    return {
        "model": name,
        "avg_density": float(np.mean(densities)),
        "avg_diversity": float(np.mean(diversities)),
        "avg_repetition": float(np.mean(repetitions)),
    }


def generate_random_dataset_like(reference_data, hit_prob=0.18):
    return np.random.binomial(1, hit_prob, size=reference_data.shape).astype(np.int32)


def train_markov_chain(data):
    num_channels = data.shape[2]
    transitions = {}

    for ch in range(num_channels):
        counts = np.ones((2, 2), dtype=np.float64)

        for seq in data:
            for t in range(1, seq.shape[0]):
                prev_state = int(seq[t - 1, ch] > 0)
                curr_state = int(seq[t, ch] > 0)
                counts[prev_state, curr_state] += 1

        probs = counts / counts.sum(axis=1, keepdims=True)
        transitions[ch] = probs

    return transitions


def generate_markov_sequence(transitions, seq_len=128, num_channels=6):
    sequence = np.zeros((seq_len, num_channels), dtype=np.int32)

    for ch in range(num_channels):
        state = np.random.choice([0, 1])
        sequence[0, ch] = state

        for t in range(1, seq_len):
            probs = transitions[ch][state]
            state = np.random.choice([0, 1], p=probs)
            sequence[t, ch] = state

    return sequence


def generate_markov_dataset_like(reference_data):
    num_samples, seq_len, num_channels = reference_data.shape
    transitions = train_markov_chain(reference_data)

    generated = []
    for _ in range(num_samples):
        generated.append(generate_markov_sequence(transitions, seq_len, num_channels))

    return np.array(generated)


def print_results_table(results_list):
    print("\n=== Final Evaluation Results ===")
    print(f"{'Model':<20} {'Density':<12} {'Diversity':<12} {'Repetition':<12}")
    print("-" * 62)
    for r in results_list:
        print(f"{r['model']:<20} {r['avg_density']:<12.4f} {r['avg_diversity']:<12.4f} {r['avg_repetition']:<12.4f}")


def save_results(results_list, out_path="data/final_evaluation_results.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"\nSaved final evaluation results to: {out_path}")


if __name__ == "__main__":
    real_data = load_dataset("data/groove_sequences.npy")
    real_data = standardize_shape(real_data)
    real_data = binarize(real_data, threshold=0.5)

    print(f"Standardized real dataset shape: {real_data.shape}")

    random_data = generate_random_dataset_like(real_data, hit_prob=0.18)
    markov_data = generate_markov_dataset_like(real_data)

    lstm_data = load_dataset("data/lstm_generated_samples.npy")
    lstm_data = standardize_shape(lstm_data)
    lstm_data = binarize(lstm_data, threshold=0.40)
    print(f"Standardized LSTM generated shape: {lstm_data.shape}")

    vae_data = load_dataset("data/vae_generated_samples.npy")
    vae_data = standardize_shape(vae_data)
    vae_data = binarize(vae_data, threshold=0.51)
    print(f"Standardized VAE generated shape: {vae_data.shape}")

    results = [
        evaluate_dataset(real_data, name="Real Dataset"),
        evaluate_dataset(random_data, name="Random Baseline"),
        evaluate_dataset(markov_data, name="Markov Baseline"),
        evaluate_dataset(lstm_data, name="LSTM Autoencoder"),
        evaluate_dataset(vae_data, name="LSTM VAE"),
    ]

    print_results_table(results)
    save_results(results)