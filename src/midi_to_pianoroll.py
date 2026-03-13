import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# Standard MIDI drum mapping (General MIDI)
DRUM_MAPPING = {
    # Kick drums
    36: 0,   # Acoustic Bass Drum
    35: 0,   # Bass Drum 1
    
    # Snare drums
    38: 1,   # Acoustic Snare
    40: 1,   # Electric Snare
    37: 1,   # Side Stick / Rimshot
    
    # Hi-Hat
    42: 2,   # Closed Hi-Hat
    44: 2,   # Pedal Hi-Hat
    46: 2,   # Open Hi-Hat
    
    # Tom drums
    43: 3,   # High Floor Tom
    45: 3,   # Low Tom
    47: 3,   # Low-Mid Tom
    48: 3,   # Hi-Mid Tom
    50: 3,   # High Tom
    
    # Cymbals
    49: 4,   # Crash Cymbal 1
    52: 4,   # Chinese Cymbal
    55: 4,   # Splash Cymbal
    51: 5,   # Ride Cymbal 1
    53: 5,   # Ride Bell
    59: 5,   # Ride Cymbal 2
}

NUM_DRUMS = 6  # Kick, Snare, HiHat, Tom, Crash, Ride

def get_ppq(midi_file_path):
    """Get pulses per quarter note (timing resolution) from MIDI file"""
    import mido
    try:
        midi = mido.MidiFile(midi_file_path)
        return midi.ticks_per_beat
    except:
        return 480  # Default value

def events_to_pianoroll(events, ppq=480, steps_per_bar=16, bars=None):
    """
    Convert parsed MIDI events to piano-roll representation
    
    Args:
        events: List of event dicts with 'type', 'note', 'time', etc.
        ppq: Pulses per quarter note (timing resolution)
        steps_per_bar: Number of steps per bar (e.g., 16)
        bars: Number of bars to include (None = auto-detect)
    
    Returns:
        piano_roll: numpy array of shape (num_drums, total_steps)
        metadata: dict with timing info
    """
    
    # Get note_on events only
    note_events = [e for e in events if e['type'] == 'note_on' and e.get('velocity', 0) > 0]
    
    if not note_events:
        return None, None
    
    # Find max duration
    max_time = max(e['time'] for e in note_events)
    
    # Calculate steps
    # ppq = ticks per quarter note
    # 4 quarter notes = 1 bar
    # steps_per_bar = desired resolution
    
    ticks_per_step = (ppq * 4) / steps_per_bar  # ticks per step
    total_steps = int(max_time / ticks_per_step) + 1
    
    # Limit to specific number of bars if requested
    if bars:
        max_steps = bars * steps_per_bar
        total_steps = min(total_steps, max_steps)
    
    # Create piano roll (drums × steps)
    piano_roll = np.zeros((NUM_DRUMS, total_steps), dtype=np.int32)
    
    # Fill in note events
    for event in note_events:
        note = event.get('note', 0)
        time = event['time']
        
        # Map MIDI note to drum category
        if note in DRUM_MAPPING:
            drum_idx = DRUM_MAPPING[note]
            step = int(time / ticks_per_step)
            
            if 0 <= step < total_steps:
                piano_roll[drum_idx, step] = 1
    
    metadata = {
        'ppq': ppq,
        'steps_per_bar': steps_per_bar,
        'total_steps': total_steps,
        'total_bars': total_steps / steps_per_bar,
        'num_drums': NUM_DRUMS,
        'drum_names': ['Kick', 'Snare', 'HiHat', 'Tom', 'Crash', 'Ride']
    }
    
    return piano_roll, metadata

def create_sequences(piano_roll, sequence_length=512):
    """
    Break piano-roll into fixed-length sequences
    
    Args:
        piano_roll: numpy array of shape (num_drums, total_steps)
        sequence_length: Length of each sequence (default 512 steps)
    
    Returns:
        sequences: List of numpy arrays, each shape (num_drums, sequence_length)
    """
    num_drums, total_steps = piano_roll.shape
    sequences = []
    
    for start in range(0, total_steps - sequence_length, sequence_length // 2):
        seq = piano_roll[:, start:start + sequence_length]
        
        # Only keep if full length
        if seq.shape[1] == sequence_length:
            sequences.append(seq)
    
    return sequences

# Test the conversion
if __name__ == "__main__":
    print("Testing Piano-Roll Conversion\n")
    
    # Load parsed sample data
    sample_file = Path('data/groove_parsed_sample.json')
    if not sample_file.exists():
        print("❌ Run parse_groove_dataset.py first")
        exit(1)
    
    with open(sample_file, 'r') as f:
        parsed_data = json.load(f)
    
    print(f"Loaded {len(parsed_data)} parsed MIDI files\n")
    
    # Convert first 5 files to piano-roll
    all_sequences = []
    
    for i, data in enumerate(parsed_data[:5]):
        midi_file = data['file']
        events = data['events']
        
        print(f"[{i+1}/5] Converting {Path(midi_file).stem}...")
        
        # Get PPQ from original MIDI
        ppq = get_ppq(midi_file)
        
        # Convert to piano-roll
        piano_roll, metadata = events_to_pianoroll(events, ppq=ppq, steps_per_bar=16)
        
        if piano_roll is not None:
            print(f"  ✓ Piano-roll shape: {piano_roll.shape}")
            print(f"    Total bars: {metadata['total_bars']:.1f}")
            print(f"    Drums: {', '.join(metadata['drum_names'])}")
            
            # Create sequences
            sequences = create_sequences(piano_roll, sequence_length=128)
            all_sequences.extend(sequences)
            print(f"  ✓ Created {len(sequences)} sequences")
        else:
            print(f"  ✗ Failed to convert")
    
    print(f"\n✓ Total sequences created: {len(all_sequences)}")
    
    if all_sequences:
        seq = all_sequences[0]
        print(f"\nSample sequence shape: {seq.shape}")
        print(f"  Drums (rows): {NUM_DRUMS}")
        print(f"  Timesteps (cols): {seq.shape[1]}")
        
        # Show visualization
        print("\nFirst 50 timesteps visualization:")
        drum_names = ['Kick ', 'Snare', 'HiHat', 'Tom  ', 'Crash', 'Ride ']
        for drum_idx, name in enumerate(drum_names):
            row = ''.join(['█' if seq[drum_idx, t] == 1 else '·' for t in range(50)])
            print(f"  {name}: {row}")
        
        # Save sequences
        output_file = 'data/groove_sequences.npy'
        np.save(output_file, np.array(all_sequences))
        print(f"\n✓ Saved {len(all_sequences)} sequences to {output_file}")