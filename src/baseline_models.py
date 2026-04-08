import os
import json
import random
import numpy as np
import mido

# Drum note mapping for 6 channels
DRUM_NOTES = {
    0: 36,  # Kick
    1: 38,  # Snare
    2: 42,  # Hi-Hat Closed
    3: 45,  # Tom
    4: 49,  # Crash
    5: 51,  # Ride
}


def load_dataset(npy_path="data/groove_sequences.npy"):
    data = np.load(npy_path)
    print(f"Loaded dataset shape: {data.shape}")
    return data


def random_drum_pattern(seq_len=16, num_channels=6, hit_prob=0.18):
    pattern = np.random.binomial(1, hit_prob, size=(seq_len, num_channels))
    return pattern.astype(np.int32)


def train_markov_chain(data):
    """
    Learn transition probabilities per drum channel:
    P(x_t | x_(t-1))
    """
    num_channels = data.shape[2]
    transitions = {}

    for ch in range(num_channels):
        # counts[from_state][to_state]
        counts = np.ones((2, 2), dtype=np.float64)  # Laplace smoothing

        for seq in data:
            for t in range(1, seq.shape[0]):
                prev_state = int(seq[t - 1, ch] > 0)
                curr_state = int(seq[t, ch] > 0)
                counts[prev_state, curr_state] += 1

        probs = counts / counts.sum(axis=1, keepdims=True)
        transitions[ch] = probs

    return transitions


def generate_markov_pattern(transitions, seq_len=16, num_channels=6):
    pattern = np.zeros((seq_len, num_channels), dtype=np.int32)

    for ch in range(num_channels):
        state = np.random.choice([0, 1])
        pattern[0, ch] = state

        for t in range(1, seq_len):
            probs = transitions[ch][state]
            state = np.random.choice([0, 1], p=probs)
            pattern[t, ch] = state

    return pattern


def save_markov_model(transitions, path="data/markov_model.json"):
    serializable = {str(k): v.tolist() for k, v in transitions.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def pattern_to_midi(pattern, out_file, ticks_per_step=120, velocity=100):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Channel 9 = MIDI channel 10 for drums
    track.append(mido.Message("program_change", program=0, time=0, channel=9))

    active_notes = []

    for t in range(pattern.shape[0]):
        step_notes = []

        for ch in range(pattern.shape[1]):
            if pattern[t, ch] > 0:
                note = DRUM_NOTES[ch]
                step_notes.append(note)
                track.append(mido.Message(
                    "note_on",
                    note=note,
                    velocity=velocity,
                    time=0,
                    channel=9
                ))

        # Turn off previous notes after one step
        if active_notes:
            first = True
            for note in active_notes:
                track.append(mido.Message(
                    "note_off",
                    note=note,
                    velocity=0,
                    time=ticks_per_step if first else 0,
                    channel=9
                ))
                first = False
        elif step_notes:
            # advance time if needed on first active note-off later
            pass

        active_notes = step_notes

    # Turn off any remaining active notes
    if active_notes:
        first = True
        for note in active_notes:
            track.append(mido.Message(
                "note_off",
                note=note,
                velocity=0,
                time=ticks_per_step if first else 0,
                channel=9
            ))
            first = False

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    mid.save(out_file)
    print(f"Saved MIDI: {out_file}")


def generate_random_samples(num_samples=5, out_dir="data/generated_samples/random"):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(num_samples):
        pattern = random_drum_pattern()
        out_file = os.path.join(out_dir, f"random_sample_{i+1}.mid")
        pattern_to_midi(pattern, out_file)


def generate_markov_samples(transitions, num_samples=5, out_dir="data/generated_samples/markov"):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(num_samples):
        pattern = generate_markov_pattern(transitions)
        out_file = os.path.join(out_dir, f"markov_sample_{i+1}.mid")
        pattern_to_midi(pattern, out_file)


if __name__ == "__main__":
    data = load_dataset("data/groove_sequences.npy")

    print("Generating random baseline samples...")
    generate_random_samples(num_samples=5)

    print("Training Markov chain baseline...")
    transitions = train_markov_chain(data)
    save_markov_model(transitions, "data/markov_model.json")

    print("Generating Markov baseline samples...")
    generate_markov_samples(transitions, num_samples=5)

    print("Done.")