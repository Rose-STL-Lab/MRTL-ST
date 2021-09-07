import pandas as pd
import numpy as np
import csv
import os

# Replace the read_pickle argument with the path of whatever file you want to add player names on
df = pd.read_pickle('full_data_bs.pkl')

# Just for logging purposes
print(f'New Shape: {df.shape}')
_, counts = np.unique(df['shoot_label'], return_counts=True)
print(f'New label percentages: {counts / df.shape[0]}')

# Add player name and playstyle columns to dataframe
df['bh_name'] = 'undefined'
df['bh_playstyle'] = 0
df['playstyle_name'] = 'undefined'

# Add player name and playstyle columns to dataframe header
cols = [
        'team_no', 'possession_no', 'teamID_A', 'teamID_B', 'timestamps',
        'quarter', 'a', 'bh_ID', 'bh_name', 'bh_playstyle', 'playstyle_name',
        'bh_x', 'bh_y', 'bh_dist_from_ball', 'bh_dist_from_basket',
        'bh_angle_from_basket', 'def1_ID', 'def2_ID', 'def3_ID', 'def4_ID',
        'def5_ID', 'def1_dist_from_bh', 'def2_dist_from_bh', 'def3_dist_from_bh',
        'def4_dist_from_bh', 'def5_dist_from_bh', 'def1_rel_angle_from_bh',
        'def2_rel_angle_from_bh', 'def3_rel_angle_from_bh',
        'def4_rel_angle_from_bh', 'def5_rel_angle_from_bh', 'def1_trunc_x',
        'def1_trunc_y', 'def2_trunc_x', 'def2_trunc_y', 'def3_trunc_x',
        'def3_trunc_y', 'def4_trunc_x', 'def4_trunc_y', 'def5_trunc_x',
        'def5_trunc_y', 'shoot_label'
        ]
df = df[cols]

# Get list of all the unique ball handler IDs in our dataframe
player_ids = df['bh_ID'].unique()

n = len(player_ids)
i = 0

# Iterate through each of the player_id in our list
for id in player_ids:
    i += 1
    
    # Logging purposes
    print(str(i) + '/' + str(n) + ' ids done')
    
    # Open the shot logs file
    shot_logs_csv_file = open(os.path.join('./', 'shots_fixed.csv'), "r")
    shot_logs = csv.reader(shot_logs_csv_file)
    
    # Open the playstyle file
    playstyle_csv_file = open(os.path.join('./', 'cluster_data.csv'), "r")
    playstyle = csv.reader(playstyle_csv_file)
    
    # make a copy of the shot_logs
    shot_logs1 = list(filter(lambda p: True, shot_logs))
    
    # Filter the shot_logs copy to only contain the current player_id being iterated on
    shot_log = list(filter(lambda p: str(id) == p[12], shot_logs1))

    # if player_id not found in the shot logs, ignore the current player_id. 
    if len(shot_log) == 0:
        continue

    # get name for the current player_id
    name = shot_log[0][13]
    
    # For every dataframe row with the current player_id, assign the bh_name column to the name retrieved above
    df.loc[df['bh_ID'] == id, 'bh_name'] = name

    # Close the shot logs file
    shot_logs_csv_file.close()
    
    # make a copy of the playstyles
    playstyle1 = list(filter(lambda p: True, playstyle))
    
    # Filter the playstyles copy to only contain the current player being iterated on
    playstyles = list(filter(lambda p: name == p[0], playstyle1))

    # if player not found in the playstyles, ignore the current player. 
    if len(playstyles) == 0:
        continue

    # get style for the current player
    style = int(playstyles[0][2])
    style_name = playstyles[0][1]
    
    # For every dataframe row with the current player, assign the bh_playstyle
    # and playstyle_name columns to the values retrieved above
    df.loc[df['bh_name'] == name, 'bh_playstyle'] = style
    df.loc[df['bh_name'] == name, 'playstyle_name'] = style_name

    # Close the shot logs file
    playstyle_csv_file.close()

# Remove the rows with no bh_name i.e. not found in shot_logs
df = df[df['bh_name'] != 'undefined']
df = df[df['playstyle_name'] != 'undefined']

# Reassign player IDs
df['a'], uniques = pd.factorize(df['bh_name'])

# Logging purposes
print(f'New Shape: {df.shape}')
_, counts = np.unique(df['shoot_label'], return_counts=True)
print(f'New label percentages: {counts / df.shape[0]}')

# Save the new dataset to a pickle file
df.to_pickle('full_data_with_names.pkl')

# Save player names and their a for reference
db = df[['a', 'bh_name']]

db.to_csv('player_names_and_ids.csv')