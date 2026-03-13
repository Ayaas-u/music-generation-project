import os
from pathlib import Path

# Check if groove data exists
groove_path = Path('data/groove')

if groove_path.exists():
    print("✓ data/groove directory found")
    
    # List contents
    print("\nContents:")
    for item in groove_path.iterdir():
        if item.is_dir():
            midi_count = len(list(item.rglob('*.mid')))
            print(f"  📁 {item.name} ({midi_count} MIDI files)")
        else:
            print(f"  📄 {item.name}")
    
    # Count total MIDI files
    total_midi = len(list(groove_path.rglob('*.mid')))
    print(f"\n✓ Total MIDI files: {total_midi}")
else:
    print("❌ data/groove not found")