import numpy as np
import torch
import mido
from pathlib import Path

from vae import LSTMVAE

REVERSE_DRUM_MAPPING = {
    0: 36,  # Kick
    1: 38,  # Snare
    2: 42,  # Hi-Hat
    3: 47,  # Tom
    4: 49,  # Crash
    5: 51   # Ride
}


def pianoroll_to_midi(pianoroll, output_path, bpm=120, tpq=480):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))

    ticks_per_step = tpq // 4
    last_event_time = 0

    for step in range(pianoroll.shape[0]):
        for drum_idx in range(pianoroll.shape[1]):
            if pianoroll[step, drum_idx] > 0:
                note = REVERSE_DRUM_MAPPING[drum_idx]
                current_tick = step * ticks_per_step
                delta_time = current_tick - last_event_time

                track.append(mido.Message(
                    'note_on', note=note, velocity=90, time=delta_time, channel=9
                ))
                track.append(mido.Message(
                    'note_off', note=note, velocity=0, time=0, channel=9
                ))

                last_event_time = current_tick

    mid.save(output_path)


def generate_vae_samples(num_samples=15, threshold=0.5, z_scale=1.5):
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    model_path = project_root / 'data' / 'lstm_vae.pth'
    output_dir = project_root / 'data' / 'generated_samples' / 'vae'
    numpy_output_path = project_root / 'data' / 'vae_generated_samples.npy'

    if not model_path.exists():
        print(f"Error: model not found at {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LSTMVAE(input_dim=6, hidden_dim=64, latent_dim=32, seq_len=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_samples = []

    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, 32).to(device) * z_scale
            logits = model.decode(z)
            probs = torch.sigmoid(logits)
            sample_np = probs.squeeze(0).cpu().numpy()

            generated_samples.append(sample_np)

            binary_roll = (sample_np > threshold).astype(np.int32)

            filename = output_dir / f"vae_sample_{i+1}.mid"
            pianoroll_to_midi(binary_roll, filename)
            print(f"Saved {filename.name}")

    generated_samples = np.array(generated_samples)
    np.save(numpy_output_path, generated_samples)
    print(f"Saved NumPy samples to: {numpy_output_path}")
    print(f"Generated sample shape: {generated_samples.shape}")


if __name__ == "__main__":
    generate_vae_samples()