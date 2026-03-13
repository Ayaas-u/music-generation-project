import os
import csv
from pathlib import Path

# Find the groove dataset directory
groove_dir = Path('data/groove')

if groove_dir.exists():
    print(f"✓ Found Groove dataset at: {groove_dir}\n")
    
    # Read metadata
    print("Dataset Metadata:")
    metadata_file = groove_dir / 'info.csv'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f"Total entries: {len(rows)}")
            print("\nFirst 5 entries (columns):")
            if rows:
                for col in list(rows[0].keys())[:8]:  # Show first 8 columns
                    print(f"  - {col}")
            
            print("\nFirst 5 entries (data):")
            for i, row in enumerate(rows[:5]):
                print(f"\n  Entry {i+1}:")
                print(f"    File: {row.get('split', 'N/A')}/{row.get('drummer', 'N/A')}.mid")
                print(f"    Genre: {row.get('primary_style', 'N/A')}")
                print(f"    Tempo: {row.get('bpm', 'N/A')}")
    
    # Count MIDI files
    midi_files = list(groove_dir.rglob('*.mid'))
    print(f"\n✓ Total MIDI files: {len(midi_files)}")
    
    # Show split distribution
    print("\nDataset splits:")
    splits = {}
    for midi_file in midi_files:
        
        split = midi_file.parent.parent.name  # Get split folder name
        splits[split] = splits.get(split, 0) + 1
    
    for split, count in sorted(splits.items()):
        print(f"  - {split}: {count} files")
else:
    print("❌ Groove dataset not found")