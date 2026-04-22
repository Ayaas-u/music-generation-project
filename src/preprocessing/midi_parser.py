import mido
from pathlib import Path
import json

def parse_midi_file(midi_path):
    """Parse a MIDI file and extract note events"""
    try:
        midi = mido.MidiFile(midi_path)
        events = []
        current_time = 0
        
        for track in midi.tracks:
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    events.append({
                        'type': 'note_on',
                        'note': msg.note,
                        'velocity': msg.velocity,
                        'time': current_time,
                        'channel': msg.channel
                    })
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    events.append({
                        'type': 'note_off',
                        'note': msg.note,
                        'time': current_time,
                        'channel': msg.channel
                    })
                elif msg.type == 'set_tempo':
                    events.append({
                        'type': 'set_tempo',
                        'tempo': msg.tempo,
                        'time': current_time
                    })
        
        return {
            'file': str(midi_path),
            'num_events': len(events),
            'duration': current_time,
            'events': events
        }
    except Exception as e:
        print(f"Error parsing {midi_path}: {e}")
        return None

# Find groove dataset
groove_dir = Path('data/groove')

if not groove_dir.exists():
    print("❌ Groove dataset not found")
    exit(1)

print(f"Parsing MIDI files from {groove_dir}...\n")

# Find all MIDI files
midi_files = sorted(list(groove_dir.rglob('*.mid')))
print(f"Found {len(midi_files)} MIDI files")
print(f"Parsing first 20 files as sample...\n")

# Parse first 20 as example
parsed_data = []
for i, midi_file in enumerate(midi_files[:20]):
    print(f"[{i+1}/20] {midi_file.parent.name}/{midi_file.stem}...", end=" ")
    data = parse_midi_file(str(midi_file))
    if data:
        parsed_data.append(data)
        print(f"✓ ({data['num_events']} events, duration: {data['duration']})")
    else:
        print("✗ Failed")

# Save parsed data
output_file = 'data/groove_parsed_sample.json'
with open(output_file, 'w') as f:
    json.dump(parsed_data, f, indent=2)

print(f"\n✓ Saved parsed data to {output_file}")

# Print statistics
if parsed_data:
    print("\nStatistics:")
    avg_events = sum(d['num_events'] for d in parsed_data) / len(parsed_data)
    avg_duration = sum(d['duration'] for d in parsed_data) / len(parsed_data)
    print(f"  Total files parsed: {len(parsed_data)}")
    print(f"  Avg events per file: {avg_events:.0f}")
    print(f"  Avg duration (ticks): {avg_duration:.0f}")
    print(f"  Min events: {min(d['num_events'] for d in parsed_data)}")
    print(f"  Max events: {max(d['num_events'] for d in parsed_data)}")