# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# Calculates the distance between (x,y) and (xRef,yRef)
def calculate_distance(x, y, xRef, yRef):
    # Applies the Pythagorean Theorem to find the distance between the two points
    dist = np.sqrt(np.power(x - xRef, 2) + np.power(y - yRef, 2))
    return dist     # Returns the calculated distance

# Calculates the angle between (x,y) and (xRef,yRef). 0º for the right corner,
# 90º for the front of the basket, 180º for the left corner.
def calculate_angle(x, y, xRef, yRef):
    # Calculates the distance
    dist = calculate_distance(x, y, xRef, yRef)
    
    # Returns 0 if the distance is 0
    if dist == 0.0:
        return 0.0

    # yRef - y = dist * cos(angle)
    angle = np.arccos((yRef - y)/dist)
        
    return angle / np.pi * 180    # Returns the calculated angle in degrees

# Calculates the angle between (x,y) and (xRef,yRef). 0º for the right corner,
# 90º for the front of the basket, 180º for the left corner. Additional
# adjustment to calculate angle for defenders
def calculate_def_angle(x, y, bh_theta):
    # Calculates the distance
    dist = calculate_distance(x, y, 0, 0)
    
    # Returns 0 if the distance is 0
    if dist == 0.0:
        return 0.0

    # - y = dist * cos(angle)
    angle = np.arccos((0 - y)/dist)
    
    # Adjusts angle for values above 180º
    if x < 0:
        return 2 * np.pi - angle
        
    angle = (angle - np.pi - bh_theta) % (2 * np.pi)
    
    return angle / (2 * np.pi) * 360   # Returns adjusted angle in degrees

notrunc_data = pd.read_pickle("data/basketball/full_data_notrunc.pkl")

bh_pos = (data.loc[420, 'bh_x':'bh_y']).astype(np.float).to_numpy()
def_pos = (data.loc[420, 'def5_trunc_x':'def5_trunc_y']).astype(np.float).to_numpy()
def_abs_pos = [(def_pos[0] + bh_pos[0]), (def_pos[1] + bh_pos[1])]
x_basket = 5.35
y_basket = 25
n = 3
m = 5
a = 8
b = 6
max_length = 36
def_circle_r = 6
invalid_val = -100

notrunc_data['bh_angle_from_basket'] = notrunc_data.apply(lambda row :
                                                          calculate_angle(
                                                              row['bh_x'],
                                                              row['bh_y'],
                                                              x_basket,
                                                              y_basket),
                                                          axis = 1)
    
notrunc_data['bh_dist_from_basket'] = notrunc_data.apply(lambda row :
                                                          calculate_distance(
                                                              row['bh_x'],
                                                              row['bh_y'],
                                                              x_basket,
                                                              y_basket),
                                                          axis = 1)
    
notrunc_data.loc[(notrunc_data['bh_x'] < x_basket) &
                  (notrunc_data['bh_y'] <= y_basket),
                 'bh_angle_from_basket'] = 0.0
notrunc_data.loc[(notrunc_data['bh_x'] <= x_basket) &
                  (notrunc_data['bh_y'] > y_basket),
                 'bh_angle_from_basket'] = 179.999999
notrunc_data.loc[notrunc_data['bh_dist_from_basket'] >= max_length,
                 'bh_dist_from_basket'] = 35.999999

notrunc_data['def1_dist_from_bh'] = notrunc_data.apply(lambda row :
                                                       calculate_distance(
                                                           row['def1_trunc_x'],
                                                           row['def1_trunc_y'],
                                                           0, 0), axis = 1)
notrunc_data['def2_dist_from_bh'] = notrunc_data.apply(lambda row :
                                                       calculate_distance(
                                                           row['def2_trunc_x'],
                                                           row['def2_trunc_y'],
                                                           0, 0), axis = 1)
notrunc_data['def3_dist_from_bh'] = notrunc_data.apply(lambda row :
                                                       calculate_distance(
                                                           row['def3_trunc_x'],
                                                           row['def3_trunc_y'],
                                                           0, 0), axis = 1)
notrunc_data['def4_dist_from_bh'] = notrunc_data.apply(lambda row :
                                                       calculate_distance(
                                                           row['def4_trunc_x'],
                                                           row['def4_trunc_y'],
                                                           0, 0), axis = 1)
notrunc_data['def5_dist_from_bh'] = notrunc_data.apply(lambda row :
                                                       calculate_distance(
                                                           row['def5_trunc_x'],
                                                           row['def5_trunc_y'],
                                                           0, 0), axis = 1)

notrunc_data['def1_rel_angle_from_bh'] = notrunc_data.apply(lambda row :
                                                    calculate_def_angle(
                                                    row['def1_trunc_x'],
                                                    row['def1_trunc_y'],
                                                    row['bh_angle_from_basket']
                                                    ), axis = 1)
notrunc_data['def2_rel_angle_from_bh'] = notrunc_data.apply(lambda row :
                                                    calculate_def_angle(
                                                    row['def2_trunc_x'],
                                                    row['def2_trunc_y'],
                                                    row['bh_angle_from_basket']
                                                    ), axis = 1)
notrunc_data['def3_rel_angle_from_bh'] = notrunc_data.apply(lambda row :
                                                    calculate_def_angle(
                                                    row['def3_trunc_x'],
                                                    row['def3_trunc_y'],
                                                    row['bh_angle_from_basket']
                                                    ), axis = 1)
notrunc_data['def4_rel_angle_from_bh'] = notrunc_data.apply(lambda row :
                                                    calculate_def_angle(
                                                    row['def4_trunc_x'],
                                                    row['def4_trunc_y'],
                                                    row['bh_angle_from_basket']
                                                    ), axis = 1)
notrunc_data['def5_rel_angle_from_bh'] = notrunc_data.apply(lambda row :
                                                    calculate_def_angle(
                                                    row['def5_trunc_x'],
                                                    row['def5_trunc_y'],
                                                    row['bh_angle_from_basket']
                                                    ), axis = 1)
    
notrunc_data.loc[notrunc_data['def1_dist_from_bh'] >= def_circle_r ,
                 ['def1_dist_from_bh', 'def1_rel_angle_from_bh']] = invalid_val
notrunc_data.loc[notrunc_data['def2_dist_from_bh'] >= def_circle_r ,
                 ['def2_dist_from_bh', 'def2_rel_angle_from_bh']] = invalid_val
notrunc_data.loc[notrunc_data['def3_dist_from_bh'] >= def_circle_r ,
                 ['def3_dist_from_bh', 'def3_rel_angle_from_bh']] = invalid_val
notrunc_data.loc[notrunc_data['def4_dist_from_bh'] >= def_circle_r ,
                 ['def4_dist_from_bh', 'def4_rel_angle_from_bh']] = invalid_val
notrunc_data.loc[notrunc_data['def5_dist_from_bh'] >= def_circle_r ,
                 ['def5_dist_from_bh', 'def5_rel_angle_from_bh']] = invalid_val

notrunc_data.loc[(abs(notrunc_data['def1_trunc_x']) >= 6) |
                  (notrunc_data['def1_trunc_y'] < -2) |
                  (notrunc_data['def1_trunc_y'] >= 10),
                 ['def1_trunc_x', 'def1_trunc_y']] = invalid_val
notrunc_data.loc[(abs(notrunc_data['def2_trunc_x']) >= 6) |
                  (notrunc_data['def2_trunc_y'] < -2) |
                  (notrunc_data['def2_trunc_y'] >= 10),
                 ['def2_trunc_x', 'def2_trunc_y']] = invalid_val
notrunc_data.loc[(abs(notrunc_data['def3_trunc_x']) >= 6) |
                  (notrunc_data['def3_trunc_y'] < -2) |
                  (notrunc_data['def3_trunc_y'] >= 10),
                 ['def3_trunc_x', 'def3_trunc_y']] = invalid_val
notrunc_data.loc[(abs(notrunc_data['def4_trunc_x']) >= 6) |
                  (notrunc_data['def4_trunc_y'] < -2) |
                  (notrunc_data['def4_trunc_y'] >= 10),
                 ['def4_trunc_x', 'def4_trunc_y']] = invalid_val
notrunc_data.loc[(abs(notrunc_data['def5_trunc_x']) >= 6) |
                  (notrunc_data['def5_trunc_y'] < -2) |
                  (notrunc_data['def5_trunc_y'] >= 10),
                 ['def5_trunc_x', 'def5_trunc_y']] = invalid_val

notrunc_data.to_pickle("full_data_trunc.pkl")
