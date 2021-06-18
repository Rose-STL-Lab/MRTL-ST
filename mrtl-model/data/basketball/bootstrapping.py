import pandas as pd
import numpy as np

num_append = 5

df = pd.read_pickle('./full_data_raw.pkl')
ones = df[df["shoot_label"] == 1]

frames = [df]

for i in range(num_append):
    frames.append(ones)

new_df = pd.concat(frames)

new_df = new_df.sample(frac=1).reset_index(drop=True)

print(f'New Shape: {new_df.shape}')
_, counts = np.unique(new_df['shoot_label'], return_counts=True)
print(f'New label percentages: {counts / new_df.shape[0]}')

# Creating a pickle file of our Game Logs dataframe
print('Writing preprocessed pickle file...')

new_df.to_pickle('full_data_bs.pkl')