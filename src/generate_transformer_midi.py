import json
from pathlib import Path

import numpy as np
import torch
import pretty_midi

from transformer import DrumTransformer
from transformer_tokenizer import tokens_to_sequence


VOCAB_SIZE = 66
BLOCK_SIZE = 128
BOS_TOKEN = 64

DRUM_NOTES = {
    0: 36,  # kick
    1: 38,  # snare
    2: 42,  # hihat
    3: 45,  # tom
    4: 49,  # crash
    5: 51,  # ride
}


def save_drum_midi(sequence_6xT, out_path, tempo=120, velocity=115, step_duration=0.125):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    drum = pretty_midi.Instrument(program=0, is_drum=True)

    num_drums, T = sequence_6xT.shape
    assert num_drums == 6

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


def generate_one_sample(model, device, total_steps=512, temperature= 1.4 , top_k=5):
    start = torch.tensor([[BOS_TOKEN]], dtype=torch.long, device=device)
    generated = model.generate(
        start,
        max_new_tokens=total_steps,
        temperature=temperature,
        top_k=top_k,
    )
    tokens = generated[0].detach().cpu().numpy()[1:]
    seq = tokens_to_sequence(tokens)
    return tokens, seq


def density(sequence_6xT):
    return float(sequence_6xT.mean())


def empty_step_ratio(sequence_6xT):
    empty_steps = np.sum(sequence_6xT.sum(axis=0) == 0)
    return empty_steps / sequence_6xT.shape[1]


def repetitive_pattern_ratio(sequence_6xT, chunk_size=16):
    patterns = []
    T = sequence_6xT.shape[1]

    for start in range(0, T - chunk_size + 1, chunk_size):
        patt = tuple(sequence_6xT[:, start:start + chunk_size].astype(np.int8).flatten().tolist())
        patterns.append(patt)

    if not patterns:
        return 1.0

    most_common = max(patterns.count(p) for p in set(patterns))
    return most_common / len(patterns)


def is_valid(sequence_6xT, min_density=0.10, max_density=0.28):
    d = density(sequence_6xT)
    e = empty_step_ratio(sequence_6xT)
    r = repetitive_pattern_ratio(sequence_6xT)

    if d < min_density or d > max_density:
        return False
    if e > 0.45:
        return False
    if r > 0.50:
        return False

    return True


def main():
    data_dir = Path("data")
    out_dir = data_dir / "generated_samples" / "transformer"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DrumTransformer(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ).to(device)

    checkpoint = torch.load(data_dir / "transformer_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    saved = 0
    attempts = 0
    max_attempts = 100

    all_tokens = []
    all_sequences = []

    while saved < 10 and attempts < max_attempts:
        attempts += 1

        tokens, seq = generate_one_sample(
            model,
            device,
            total_steps=512,
            temperature=1.1,
            top_k=8,
        )

        if not is_valid(seq):
            continue

        saved += 1
        midi_path = out_dir / f"transformer_sample_{saved}.mid"
        npy_path = out_dir / f"transformer_sample_{saved}.npy"

        np.save(npy_path, seq)
        save_drum_midi(seq, midi_path, tempo=120, velocity=115, step_duration=0.125)

        all_tokens.append(tokens)
        all_sequences.append(seq)

        print(f"Saved {midi_path}")

    np.save(data_dir / "transformer_generated_tokens.npy", np.array(all_tokens, dtype=object))
    np.save(data_dir / "transformer_generated_sequences.npy", np.array(all_sequences, dtype=object))

    summary = {
        "num_saved_samples": saved,
        "num_attempts": attempts,
        "generation_steps": 512,
        "temperature": 1.1,
        "top_k": 8,
        "velocity": 115,
    }

    with open(data_dir / "transformer_generation_info.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()