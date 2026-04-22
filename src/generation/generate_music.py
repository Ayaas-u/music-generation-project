import torch
import numpy as np
import mido
import os
from pathlib import Path
from lstm_autoencoder import LSTMAutoencoder

# Mapping back from our 6 categories to standard General MIDI drum notes
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
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
    
    ticks_per_step = tpq // 4 
    last_event_time = 0
    
    # Inside the loop where you append messages:
    for step in range(pianoroll.shape[0]):
        for drum_idx in range(pianoroll.shape[1]):
            if pianoroll[step, drum_idx] > 0.2:
                note = REVERSE_DRUM_MAPPING[drum_idx]
                current_tick = step * ticks_per_step
                delta_time = current_tick - last_event_time
                
                # ADD channel=9 HERE
                track.append(mido.Message('note_on', note=note, velocity=90, time=delta_time, channel=9))
                track.append(mido.Message('note_off', note=note, velocity=0, time=0, channel=9))
                
                last_event_time = current_tick

    mid.save(output_path)

def generate_samples(num_samples=15, save_numpy=True):
    # Set up absolute paths
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    model_path = project_root / 'data' / 'lstm_autoencoder.pth'
    output_dir = project_root / 'data' / 'generated_samples'
    numpy_output_path = project_root / 'data' / 'lstm_generated_samples.npy'

    if not model_path.exists():
        print(f"❌ Error: Could not find model at {model_path}")
        print("Please run: python src/lstm_autoencoder.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Load Model
    model = LSTMAutoencoder(input_dim=6, hidden_dim=64, latent_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {num_samples} samples into {output_dir}...")

    generated_samples = []

    with torch.no_grad():
        for i in range(num_samples):
            z_random = torch.randn(1, 32).to(device) * 3

            # Decode latent vector into sequence
            z_expanded = model.from_latent(z_random).unsqueeze(1).repeat(1, 128, 1)
            reconstruction, _ = model.decoder(z_expanded)
            generated_roll = torch.sigmoid(model.output_layer(reconstruction))

            sample_np = generated_roll.squeeze(0).cpu().numpy()

            # Save raw generated array for evaluation
            generated_samples.append(sample_np)

            filename = output_dir / f"sample_{i+1}.mid"
            pianoroll_to_midi(sample_np, filename)
            print(f"  ✓ Saved {filename.name}")

    if save_numpy:
        generated_samples = np.array(generated_samples)
        np.save(numpy_output_path, generated_samples)
        print(f"✅ Saved NumPy samples to: {numpy_output_path}")
        print(f"NumPy sample shape: {generated_samples.shape}")

if __name__ == "__main__":
    generate_samples()