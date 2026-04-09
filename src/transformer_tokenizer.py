import json
from pathlib import Path

import numpy as np


NUM_DRUMS = 6
VOCAB_SIZE = 2 ** NUM_DRUMS   # 64
BOS_TOKEN = VOCAB_SIZE        # 64
PAD_TOKEN = VOCAB_SIZE + 1    # 65


def step_to_token(step_bits):
    token = 0
    for i, bit in enumerate(step_bits):
        token |= (int(bit) & 1) << i
    return token


def token_to_step(token, num_drums=NUM_DRUMS):
    bits = []
    for i in range(num_drums):
        bits.append((token >> i) & 1)
    return np.array(bits, dtype=np.float32)


def sequence_to_tokens(sequence_6xT):
    steps = sequence_6xT.T
    tokens = np.array([step_to_token(step) for step in steps], dtype=np.int64)
    return tokens


def tokens_to_sequence(tokens, num_drums=NUM_DRUMS):
    steps = np.array([token_to_step(tok, num_drums) for tok in tokens], dtype=np.float32)
    return steps.T


def main():
    data_path = Path("data/groove_sequences.npy")
    out_path = Path("data/groove_step_tokens.npy")
    info_path = Path("data/groove_step_tokens_info.json")

    sequences = np.load(data_path)  # shape: (N, 6, 128)
    assert sequences.ndim == 3 and sequences.shape[1] == NUM_DRUMS

    token_sequences = np.array(
        [sequence_to_tokens(seq) for seq in sequences],
        dtype=np.int64
    )

    np.save(out_path, token_sequences)

    info = {
        "input_shape": list(sequences.shape),
        "output_shape": list(token_sequences.shape),
        "num_drums": NUM_DRUMS,
        "vocab_size_without_special_tokens": VOCAB_SIZE,
        "bos_token": BOS_TOKEN,
        "pad_token": PAD_TOKEN,
        "total_vocab_size": PAD_TOKEN + 1,
        "representation": "6-bit drum-step token",
        "channel_order": ["kick", "snare", "hihat", "tom", "crash", "ride"],
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print("Saved:", out_path)
    print("Token shape:", token_sequences.shape)


if __name__ == "__main__":
    main()