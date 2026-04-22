import numpy as np
from pathlib import Path
import mido

# Standard MIDI drum mapping (General MIDI)
DRUM_MAPPING = {
    36: 0, 35: 0,              # Kick
    38: 1, 40: 1, 37: 1,       # Snare
    42: 2, 44: 2, 46: 2,       # Hi-Hat
    43: 3, 45: 3, 47: 3, 48: 3, 50: 3,  # Tom
    49: 4, 52: 4, 55: 4,       # Crash
    51: 5, 53: 5, 59: 5        # Ride
}

NUM_DRUMS = 6


def parse_midi_events(midi_file_path):
    try:
        midi = mido.MidiFile(midi_file_path)
        events = []

        for track in midi.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    events.append({
                        'type': 'note_on',
                        'note': msg.note,
                        'velocity': msg.velocity,
                        'time': current_time,
                        'channel': getattr(msg, 'channel', 9)
                    })

        return events, midi.ticks_per_beat
    except Exception as e:
        print(f"Error reading {midi_file_path}: {e}")
        return None, None


def events_to_pianoroll(events, ppq=480, steps_per_bar=16, bars=None):
    note_events = [e for e in events if e['type'] == 'note_on' and e.get('velocity', 0) > 0]

    if not note_events:
        return None, None

    max_time = max(e['time'] for e in note_events)
    ticks_per_step = (ppq * 4) / steps_per_bar
    total_steps = int(max_time / ticks_per_step) + 1

    if bars is not None:
        total_steps = min(total_steps, bars * steps_per_bar)

    piano_roll = np.zeros((NUM_DRUMS, total_steps), dtype=np.int32)

    for event in note_events:
        note = event['note']
        if note in DRUM_MAPPING:
            drum_idx = DRUM_MAPPING[note]
            step = int(event['time'] / ticks_per_step)
            if 0 <= step < total_steps:
                piano_roll[drum_idx, step] = 1

    return piano_roll, {
        'ppq': ppq,
        'steps_per_bar': steps_per_bar,
        'total_steps': total_steps
    }


def create_sequences(piano_roll, sequence_length=128, stride=64):
    num_drums, total_steps = piano_roll.shape
    sequences = []

    if total_steps < sequence_length:
        return sequences

    for start in range(0, total_steps - sequence_length + 1, stride):
        seq = piano_roll[:, start:start + sequence_length]
        if seq.shape[1] == sequence_length:
            sequences.append(seq)

    return sequences


if __name__ == "__main__":
    groove_dir = Path("data/groove")
    midi_files = sorted(groove_dir.rglob("*.mid"))

    if not midi_files:
        print("No MIDI files found in data/groove")
        exit(1)

    print(f"Found {len(midi_files)} MIDI files")

    all_sequences = []
    processed_files = 0
    skipped_files = 0

    for i, midi_file in enumerate(midi_files, 1):
        events, ppq = parse_midi_events(midi_file)

        if events is None:
            skipped_files += 1
            continue

        piano_roll, metadata = events_to_pianoroll(events, ppq=ppq, steps_per_bar=16)

        if piano_roll is None:
            skipped_files += 1
            continue

        sequences = create_sequences(piano_roll, sequence_length=128, stride=64)

        if len(sequences) == 0:
            skipped_files += 1
            continue

        all_sequences.extend(sequences)
        processed_files += 1

        if i % 50 == 0 or i == len(midi_files):
            print(f"[{i}/{len(midi_files)}] files scanned | usable files: {processed_files} | sequences: {len(all_sequences)}")

    if not all_sequences:
        print("No sequences created.")
        exit(1)

    all_sequences = np.array(all_sequences, dtype=np.int32)
    output_file = "data/groove_sequences.npy"
    np.save(output_file, all_sequences)

    print(f"\nSaved dataset to {output_file}")
    print(f"Final shape: {all_sequences.shape}")
    print(f"Processed files: {processed_files}")
    print(f"Skipped files: {skipped_files}")