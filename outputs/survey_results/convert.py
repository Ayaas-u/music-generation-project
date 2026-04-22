import pandas as pd

# 1. Load your file
df = pd.read_csv('human_ratings.csv')

# 2. Identify the fixed column (Timestamp)
timestamp = df.iloc[:, 0]

all_frames = []

# 3. Loop through the tracks (there are 10 tracks, each taking 3 columns)
# Column index 1-3 is Track 1, 4-6 is Track 2, etc.
for i in range(10):
    start_col = 1 + (i * 3)
    # Grab the 3 columns for the current track
    track_data = df.iloc[:, start_col:start_col+3].copy()
    
    # Standardize column names for this slice
    track_data.columns = ['Gr', 'Co', 'Va']
    
    # Add the Participant/Timestamp and Track Label
    track_data.insert(0, 'Timestamp', timestamp)
    track_data.insert(1, 'Track', f'Track {i+1}')
    
    all_frames.append(track_data)

# 4. Stack them all on top of each other
final_df = pd.concat(all_frames, ignore_index=True)

# 5. Save the result
final_df.to_csv('final_long_format.csv', index=False)

print("Success! 'final_long_format.csv' has been created.")