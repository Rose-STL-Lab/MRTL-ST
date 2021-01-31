import json as js
import pandas as pd
import numpy as np
import math
import csv
import os
from scipy.spatial.distance import euclidean


# Function to calculate euclidean distance between two players
# @param player_a_loc: location of the first player
# @param player_b_loc: location of the second player
# @return: the euclidean distance between the two players
def player_distance(player_a_loc, player_b_loc):
    return euclidean(player_a_loc, player_b_loc)


# Function to determine the ball handler for a specific timeframe
# Player closest to the ball is assigned as the ball handler
# @param plyrs: list of all players on-court
# @return ball_handler: Player object of the ball handler
# @return min_dist: distance between the ball and the ball_handler
# @return basket_dist: distance between the basket and the ball_handler
# @return basket_angle: angle between the basket and the ball_handler
def ball_handler_info(plyrs):
    ball_loc = [plyrs[0][2], plyrs[0][3], 0]
    min_dist = float('inf')
    ball_handler = []
    player_loc = []
    for plyr in plyrs:
        if plyr == plyrs[0]: continue
        player_loc = [plyr[2], plyr[3], 0]
        distance = player_distance(ball_loc, player_loc)
        if distance < min_dist:
            min_dist = distance
            ball_handler = plyr
    bh_loc = [ball_handler[2], ball_handler[3], 0]
    basket_loc = [5.35, -25, 0]
    basket_dist = player_distance(bh_loc, basket_loc)
    basket_angle = math.acos((ball_handler[2] - 5.35) / basket_dist)
    return [ball_handler, min_dist, basket_dist, basket_angle]


# Function to extract information of the defender's on-court
# @param ball_h: Player object of the ball handler
# @param plyrs: list of all players on-court
# @return: list containing all 5 defender's IDs, their distance from the ball handler,
#   their angle from the ball handler and their positions on the truncated coordinate plane
def get_def_info(ball_h, plyrs):
    bh_team_id = ball_h[0]
    bh_loc = [ball_h[2], ball_h[3], 0]
    basket_loc = [5.35, -25, 0]
    def_ids = []
    def_dist_from_bh = []
    def_angle_from_bh = []
    def_trunc_pos = []
    for defs in plyrs:
        if defs[0] != -1 and defs[0] != bh_team_id:  # Ensure that the player is a defender
            # Add to defender list
            def_ids.append(defs[1])
            # Calculate def's distance from ball handler
            plyr_loc = [defs[2], defs[3], 0]
            def_distance = player_distance(bh_loc, plyr_loc)
            def_dist_from_bh.append(def_distance)
            # Calculate def's relative angle using cosine rule
            a = player_distance(bh_loc, basket_loc)
            b = player_distance(bh_loc, plyr_loc)
            c = player_distance(plyr_loc, basket_loc)
            if b != 0:
                def_angle = math.acos((c ** 2 - a ** 2 - b ** 2) / (-2 * a * b))
                if ball_h[3] > defs[3]:
                    def_angle = 0 - def_angle
                def_trunc_x = def_distance * math.sin(def_angle)
                def_trunc_y = def_distance * math.cos(def_angle)
                if not (-6 < def_trunc_x < 6 and -2 < def_trunc_y < 10):
                    def_trunc_x = -100
                    def_trunc_y = -100
            else:
                def_angle = 0
                def_trunc_x = 0
                def_trunc_y = 0
            def_angle_from_bh.append(def_angle * 180/math.pi)
            # Def's truncated positions
            def_trunc_pos.append(def_trunc_x)
            def_trunc_pos.append(def_trunc_y)
    # Merge info to be returned
    return_list = def_ids  # Def IDs
    return_list.extend(def_dist_from_bh)  # Def_dist_from_bh
    return_list.extend(def_angle_from_bh)  # Def_rel_angle_from_bh
    return_list.extend(def_trunc_pos)  # Def_trunc
    return return_list


# Function to add the given game's log to the Game Logs list
# @param game_json_file: JSON file location of the current game to be added
# @param game_logs: The Game Logs list the current game is to be added to
# @param shot_logs: The Shot Logs of the whole season
def add_game_log(game_json_file, game_logs, shot_logs):
    # Opening the Game file
    # @var data: JSON File object containing the data for the game
    data_file = open(game_json_file, "r")
    data = js.loads(data_file.read())

    # Filtering the current game's Shot Log  from the whole season's Shot Log
    # @var shot_log: CSV File object containing the Shot Log for the game
    shot_log = list(filter(lambda p: data["gameid"] == p[5], shot_logs))

    # @var events: list containing all the possessions of the game
    events = data["events"]

    # @var home: list containing the home team's information
    home = events[0]["home"]
    # @var visitor: list containing the visitor team's information
    visitor = events[0]["visitor"]

    # @var moments_list: list containing all the timeframes of every possession of the game
    moments_list = []
    for event in events:
        moments_list.append(event["moments"])

    # This for-loop will loop through each of the game possessions
    # @var moments: list of all timeframes of the current possession
    for moments in moments_list:
        # For testing on individual possessions, replace the for-statement above with: moments = moments_list[159]

        # @var possession_no: Possession number of the current possession
        possession_no = moments_list.index(moments)
        # @var event_id: ID of the current possession
        event_id = events[possession_no]["eventId"]
        # @var event_shot_log: instance of the shot from the Shot Log that was attempted in the current possession
        event_shot_log = list(filter(lambda p: event_id == p[4], shot_log))
        # @var shoot_labelled: boolean flag of whether the shot has been labelled for the current possession
        shoot_labelled = False

        # This for-loop will loop through each timeframe of the current possession
        # @var moment: the current timeframe of the current possession
        for moment in moments:
            # Discard bad possessions
            if moment[3] is None:
                continue
            if len(moment[5]) < 11:
                continue

            # @var bh_info: list containing the current timeframe's ball handler's information
            bh_info = ball_handler_info(moment[5])
            # @var bh: Player object of the ball handler
            bh = bh_info[0]  # Ball handler

            # Testing purposes
            if not (bh[0] == 1610612745 and bh[2] < 48):
                continue

            # Populating the Game Log with the current timeframe
            game_log = [bh[0],  # Team no
                        possession_no,  # Possession no
                        home["teamid"],  # Team ID A (Home)
                        visitor["teamid"],  # Team ID B (Visitor)
                        moment[2],  # Timestamp (Game clock)
                        bh[1],  # Ball handler ID
                        bh[2],  # Ball handler X position
                        bh[3],  # Ball handler Y position
                        bh_info[1],  # Ball handler distance from ball
                        bh_info[2],  # Ball handler distance from basket
                        bh_info[3] * 180/math.pi]  # Ball handler angle from basket
            # Add defenders info to game log
            def_info = get_def_info(bh, moment[5])
            game_log.extend(def_info)  # Def_ID, Def_dist_from_bh, Def_rel_angle_from_bh, Def_trunc

            # Add shoot label to game log if exists and not already added
            if all([not shoot_labelled, len(event_shot_log) != 0 and float(event_shot_log[0][19]) == moment[2]]):
                game_log.append(1)
                index = len(game_logs) - 1
                sec_time = moment[3] + 1
                curr_log = game_logs[index]
                while all([index >= 0, curr_log[4] < sec_time, curr_log[5] == bh[1]]):
                    game_logs[index][36] = 1
                    index -= 1
                    curr_log = game_logs[index]
                shoot_labelled = True
            else:
                game_log.append(0)

            # Append current timeframe to the current possession's Game Log
            game_logs.append(game_log)

    # Closing the game JSON file
    data_file.close()


if __name__ == '__main__':
    # Opening the Shot Log file
    # @var shot_log: CSV File object containing the Shot Log for the season
    shot_logs_csv_file = open('shots_fixed.csv', "r")
    shot_logs = csv.reader(shot_logs_csv_file)

    # @var game_logs_headers: list containing the Column labels for the Game Log dataframe
    game_logs_headers = ['team_no', 'possession_no', 'teamID_A', 'teamID_B', 'timestamps', 'bh_ID', 'bh_x', 'bh_y',
                         'bh_dist_from_ball', 'bh_dist_from_basket', 'bh_angle_from_basket', 'def1_ID', 'def2_ID',
                         'def3_ID', 'def4_ID', 'def5_ID', 'def1_dist_from_bh', 'def2_dist_from_bh', 'def3_dist_from_bh',
                         'def4_dist_from_bh', 'def5_dist_from_bh', 'def1_rel_angle_from_bh', 'def2_rel_angle_from_bh',
                         'def3_rel_angle_from_bh', 'def4_rel_angle_from_bh', 'def5_rel_angle_from_bh', 'def1_trunc_x',
                         'def1_trunc_y', 'def2_trunc_x', 'def2_trunc_y', 'def3_trunc_x', 'def3_trunc_y', 'def4_trunc_x',
                         'def4_trunc_y', 'def5_trunc_x', 'def5_trunc_y', 'shoot_label']

    # @var game_logs: dataframe containing Game Logs of every possession of every game of the season
    game_logs = []

    # Adding a single game's log to the Game Logs list
    game_json_file = 'unzippedHOU/0021500415.json'
    add_game_log(game_json_file, game_logs, shot_logs)
    game_json_file = 'unzippedHOU/0021500426.json'
    add_game_log(game_json_file, game_logs, shot_logs)
    # game_json_file = 'data/basketball/unzippedHOU/0021500439.json'
    # add_game_log(game_json_file, game_logs, shot_logs)
    # game_json_file = 'data/basketball/unzippedHOU/0021500445.json'
    # add_game_log(game_json_file, game_logs, shot_logs)
    # game_json_file = 'data/basketball/unzippedHOU/0021500470.json'
    # add_game_log(game_json_file, game_logs, shot_logs)
    # game_json_file = 'data/basketball/unzippedHOU/0021500485.json'
    # add_game_log(game_json_file, game_logs, shot_logs)

    # Closing the Shot Logs CSV file
    shot_logs_csv_file.close()

    # @var df: our Game Logs dataframe
    df = pd.DataFrame(game_logs, columns=game_logs_headers)

    # Set dtypes
    int_cols = [
        'team_no', 'possession_no', 'teamID_A', 'teamID_B',
        'bh_ID', 'def1_ID', 'def2_ID', 'def3_ID', 'def4_ID', 'def5_ID',
        'shoot_label'
    ]
    float_cols = [
        'timestamps', 'bh_x', 'bh_y', 'bh_dist_from_ball', 'bh_dist_from_basket', 'bh_angle_from_basket',
        'def1_dist_from_bh', 'def2_dist_from_bh', 'def3_dist_from_bh', 'def4_dist_from_bh', 'def5_dist_from_bh',
        'def1_rel_angle_from_bh', 'def2_rel_angle_from_bh', 'def3_rel_angle_from_bh', 'def4_rel_angle_from_bh',
        'def5_rel_angle_from_bh', 'def1_trunc_x', 'def1_trunc_y', 'def2_trunc_x', 'def2_trunc_y', 'def3_trunc_x',
        'def3_trunc_y', 'def4_trunc_x', 'def4_trunc_y', 'def5_trunc_x', 'def5_trunc_y'
    ]
    df[int_cols] = df[int_cols].astype(np.int32)
    df[float_cols] = df[float_cols].astype(float)

    # Players must have at least n attempted shots (sum(shoot_label == 1) >= n) : n = 10
    # df = df[df.groupby('bh_ID').shoot_label.transform('sum') >= 20]

    # Reindex bh_ID into a
    df['a'] = df['bh_ID'].astype('category').cat.codes

    # Reorder columns
    cols = [
        'team_no', 'possession_no', 'teamID_A', 'teamID_B', 'timestamps', 'a',
        'bh_ID', 'bh_x', 'bh_y', 'bh_dist_from_ball', 'bh_dist_from_basket',
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

    df = df[(df.def1_trunc_x != -100) | (df.def2_trunc_x != -100) | (df.def3_trunc_x != -100) | (df.def4_trunc_x != -100) | (df.def5_trunc_x != -100)]

    # Not too sure what is going on here but I don't think it's too important for our purpose
    print(f'New Shape: {df.shape}')
    _, counts = np.unique(df['shoot_label'], return_counts=True)
    print(f'New label percentages: {counts / df.shape[0]}')

    # Creating a pickle file of our Game Logs dataframe
    print(f'Writing preprocessed pickle file...')
    df.to_pickle(os.path.join('output/', 'full_data.pkl'))

