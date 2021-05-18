import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from matplotlib.patches import Circle, Rectangle, Arc

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

def draw_half_court_left(ax, color='black', lw=1):
    # Create various parts of an NBA basketball court
    # Court boundaries
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 50)
    boundaries = Rectangle((0, 0), 40, 50, linewidth=lw, color=color, fill=False)
    ax.add_patch(boundaries)

    # The paint
    outer_box_left = Rectangle((0, 17), 19, 16, linewidth=lw, color=color, fill=False)
    inner_box_left = Rectangle((0, 19), 19, 12, linewidth=lw, color=color, fill=False)
    ax.add_patch(outer_box_left)
    ax.add_patch(inner_box_left)

    # Free Throw Lines
    top_free_throw_left = Arc((19, 25), 12, 12, theta1=-90, theta2=90, linewidth=lw, color=color, fill=False)
    bottom_free_throw_left = Arc((19, 25), 12, 12, theta1=90, theta2=-90, linewidth=lw, color=color, fill=False,
                                 linestyle='dashed')
    ax.add_patch(top_free_throw_left)
    ax.add_patch(bottom_free_throw_left)

    # Three-point lines
    top_three_left = Rectangle((0, 47), 14, 0, linewidth=lw, color=color, fill=False)
    bottom_three_left = Rectangle((0, 3), 14, 0, linewidth=lw, color=color, fill=False)
    three_arc_left = Arc((5.2493, 25), 47.5, 47.5, theta1=-68.3, theta2=68.3, linewidth=lw, color=color, fill=False)
    ax.add_patch(top_three_left)
    ax.add_patch(bottom_three_left)
    ax.add_patch(three_arc_left)

    # Backboard and hoops
    hoop_left = Circle((5.2493, 25), 0.75, linewidth=lw, color=color, fill=False)
    backboard_left = Rectangle((4, 22), 0, 6, linewidth=lw, color=color, fill=False)
    ax.add_patch(hoop_left)
    ax.add_patch(backboard_left)

    # Restricted Area
    restricted_left = Arc((5.2493, 25), 8, 8, theta1=-90, theta2=90, linewidth=lw, color=color, fill=False)
    ax.add_patch(restricted_left)

    return ax

data = np.load("data/basketball/full_data.pkl", allow_pickle=True)
notrunc_data = pd.read_pickle("data/basketball/full_data_notrunc.pkl")
pd.set_option('max_columns', None)

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

print("Original DataFrame:\n", notrunc_data)

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
                  (abs(notrunc_data['def1_trunc_y']) >= 2),
                 ['def1_trunc_x', 'def1_trunc_y']] = invalid_val
notrunc_data.loc[(abs(notrunc_data['def2_trunc_x']) >= 6) |
                  (abs(notrunc_data['def2_trunc_y']) >= 2),
                 ['def2_trunc_x', 'def2_trunc_y']] = invalid_val
notrunc_data.loc[(abs(notrunc_data['def3_trunc_x']) >= 6) |
                  (abs(notrunc_data['def3_trunc_y']) >= 2),
                 ['def3_trunc_x', 'def3_trunc_y']] = invalid_val
notrunc_data.loc[(abs(notrunc_data['def4_trunc_x']) >= 6) |
                  (abs(notrunc_data['def4_trunc_y']) >= 2),
                 ['def4_trunc_x', 'def4_trunc_y']] = invalid_val
notrunc_data.loc[(abs(notrunc_data['def5_trunc_x']) >= 6) |
                  (abs(notrunc_data['def5_trunc_y']) >= 2),
                 ['def5_trunc_x', 'def5_trunc_y']] = invalid_val

print("New Dataframe:\n", notrunc_data)

notrunc_data.to_pickle("full_data_trunc.pkl")

# notrunc_data['bh_angle_from_basket'] = notrunc_data.apply(lambda row: calculate_angle())

# For every ballhandler, we must calculate its polar coordinates, then compare
# them to the polar coordinates of each defender, calculate the distance
# between them, then replace values for defenders that are too far.
#
# How to compare the distance between ballhandler and all defenders, using all
# their x and y coordinates, and eliminate defenders in one operation?
#
# Could maybe do defenders 1 by 1? Then how do I go through all ballhandlers?
#
# nontrunc_data.loc[['Color' == blue], 'Good'] = invalid_val
#
# Add column:
#
# dataframe['column name'] = function() (apply)
#
# if a defender was outside of range -> -100
#
# Since I'm creating this truncated file specifically for polar coordinates,
# wouldn't it be better to calculate polar coordinates and save them to the
# file so the model doesn't have to do the same operations every run?

print(bh_pos)
print(def_pos)
print(def_abs_pos)

r = calculate_distance(bh_pos[0], bh_pos[1], x_basket, y_basket)
theta = calculate_angle(bh_pos[0], bh_pos[1], x_basket, y_basket)
def_r = calculate_distance(def_pos[0], def_pos[1], 0, 0)
def_theta = calculate_angle(def_pos[0], def_pos[1], 0, 0)
    
def_theta = (def_theta - np.pi - theta) % (2 * np.pi)

def_abs_r = calculate_distance(def_abs_pos[0], def_abs_pos[1], x_basket, y_basket)
def_abs_theta = calculate_angle(def_abs_pos[0], def_abs_pos[1], x_basket, y_basket)

bh_pos_polar = (theta, r)
def_pos_polar = (def_theta, def_r)
def_abs_pos_polar = (def_abs_theta, def_abs_r)

print(bh_pos_polar)
print(def_pos_polar)
print(def_abs_pos_polar)

fig = plot.figure()
ax_court = fig.add_axes([0.1175, 1/7, 85/224, 5/7])
draw_half_court_left(ax_court)

ax_polar = fig.add_axes([0, 0, 1, 1], polar=True, frameon=False)
ax_polar.plot(theta, r, 'ro', c='r')
ax_polar.plot(def_abs_theta, def_abs_r, 'ro', c='b')
plot.thetagrids(range(0, 181, 180//n))
plot.rgrids(range(0, max_length+1, max_length//m))
ax_polar.set_thetamin(0)
ax_polar.set_thetamax(180)
ax_polar.set_rmin(0)
ax_polar.set_rmax(max_length)
ax_polar.set_theta_zero_location('S')
ax_polar.set_anchor('W')

plot.show()

bh_grid = np.zeros((m,n))
bh_i = int(r/max_length*m)
bh_j = int(theta/np.pi*n)

print((bh_i,bh_j))

bh_grid[bh_i][bh_j] = 1
plot.imshow(bh_grid, cmap='hot', interpolation='nearest')
plot.show()

fig = plot.figure()
ax_court = fig.add_axes([0.1175, 1/7, 85/224, 5/7])
draw_half_court_left(ax_court)

ax_polar = fig.add_axes([0, 0, 1, 1], polar=True, frameon=False)

rad = np.linspace(0, max_length, m)
azm = np.linspace(0, np.pi, 180)
r, th = np.meshgrid(rad, azm)
bh_heatmap = np.zeros((180, m))
for i in range(bh_j*180//n, (bh_j+1)*180//n):
    bh_heatmap[i][bh_i] = 1

ax_polar.pcolormesh(th, r, bh_heatmap, shading='gouraud', alpha=0.3)

ax_polar.plot(azm, r, color='k', ls='none')

plot.thetagrids(range(0, 181, 180//n))
plot.rgrids(range(0, max_length+1, max_length//m))
ax_polar.set_thetamin(0)
ax_polar.set_thetamax(180)
ax_polar.set_rmin(0)
ax_polar.set_rmax(max_length)
ax_polar.set_theta_zero_location('S')
ax_polar.set_anchor('W')

plot.grid()
plot.show()

def_grid = np.zeros((b,a))
def_i = int(def_r/def_circle_r*b)
def_j = int((def_theta + (np.pi/a) % (2 * np.pi))/(2 * np.pi)*a)

print((def_i, def_j))

def_grid[def_i][def_j] = 1
plot.imshow(def_grid, cmap='hot', interpolation='nearest')
plot.show()

fig = plot.figure()
ax_polar = fig.add_axes([0, 0, 1, 1], polar=True, frameon=False)

rad = np.linspace(0, def_circle_r, b)
azm = np.linspace(0, 2 * np.pi, 360)
r, th = np.meshgrid(rad, azm)
def_heatmap = np.zeros((360, b))
for i in range(def_j*360//a, (def_j+1)*360//a):
    def_heatmap[((i - 180//a - 1) % 360)][def_i] = 1

ax_polar.pcolormesh(th, r, def_heatmap, shading='gouraud', alpha=0.3)

ax_polar.plot(azm, r, color='k', ls='none')

plot.thetagrids(range(180//a, 360 - (180//a) + 1, 360//a))
plot.rgrids(range(0, def_circle_r + 1, def_circle_r//b))
ax_polar.set_thetamin(0)
ax_polar.set_thetamax(360)
ax_polar.set_rmin(0)
ax_polar.set_rmax(def_circle_r)
ax_polar.set_theta_zero_location('N')

plot.grid()
plot.show()