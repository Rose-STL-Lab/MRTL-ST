import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Arc
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.preprocessing import MinMaxScaler
import os
import utils

plt.rcParams.update({'font.size': 18})


def draw_half_court_down(ax, color='black', lw=1):
    # Create various parts of an NBA basketball court
    # Court boundaries
    ax.set_xlim(0, 50)
    ax.set_ylim(40, 0)
    boundaries = Rectangle((0, 0), 50, 40, linewidth=lw, color=color, fill=False)
    ax.add_patch(boundaries)

    # The paint
    outer_box_left = Rectangle((17, 40), 16, -19, linewidth=lw, color=color, fill=False)
    inner_box_left = Rectangle((19, 40), 12, -19, linewidth=lw, color=color, fill=False)
    ax.add_patch(outer_box_left)
    ax.add_patch(inner_box_left)

    # Free Throw Lines
    top_free_throw_left = Arc((25, 40 - 19), 12, 12, theta1=180, theta2=0, linewidth=lw, color=color, fill=False)
    bottom_free_throw_left = Arc((25, 40 - 19), 12, 12, theta1=0, theta2=180, linewidth=lw, color=color, fill=False,
                                 linestyle='dashed')
    ax.add_patch(top_free_throw_left)
    ax.add_patch(bottom_free_throw_left)

    # Three-point lines
    top_three_left = Rectangle((3, 40), 0, -14, linewidth=lw, color=color, fill=False)
    bottom_three_left = Rectangle((47, 40), 0, -14, linewidth=lw, color=color, fill=False)
    three_arc_left = Arc((25, 40 - 5.2493), 47.5, 47.5, theta1=201.7, theta2=-21.7, linewidth=lw, color=color,
                         fill=False)
    ax.add_patch(top_three_left)
    ax.add_patch(bottom_three_left)
    ax.add_patch(three_arc_left)

    # Backboard and hoops
    hoop_left = Circle((25, 40 - 5.2493), 0.75, linewidth=lw, color=color, fill=False)
    backboard_left = Rectangle((22, 40 - 4), 6, 0, linewidth=lw, color=color, fill=False)
    ax.add_patch(hoop_left)
    ax.add_patch(backboard_left)

    # Restricted Area
    restricted_left = Arc((25, 40 - 5.2493), 8, 8, theta1=180, theta2=0, linewidth=lw, color=color, fill=False)
    ax.add_patch(restricted_left)

    # Flip y-axis so origin is top left
    ax.invert_yaxis()

    return ax


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


def draw_full_court(ax, color='black', lw=1):
    # Create various parts of an NBA basketball court
    # Court boundaries
    ax.set_xlim(-5, 100)
    ax.set_ylim(-5, 60)
    boundaries = Rectangle((0, 0), 93.996, 50, linewidth=lw, color=color, fill=False)
    # boundaries = Rectangle((0, 0), 94, 50, linewidth=lw, color=color, fill=False)
    ax.add_patch(boundaries)

    # Center line
    ax.vlines(46.998, 0, 50)
    # ax.vlines(47, 0, 50)

    # Center court circles
    center_outer_circle = Circle((46.998, 25), radius=6, linewidth=lw, color=color, fill=False)
    center_inner_circle = Circle((46.998, 25), radius=2, linewidth=lw, color=color, fill=False)
    # center_outer_circle = Circle((47, 25), radius=6, linewidth=lw, color=color, fill=False)
    # center_inner_circle = Circle((47, 25), radius=2, linewidth=lw, color=color, fill=False)
    ax.add_patch(center_outer_circle)
    ax.add_patch(center_inner_circle)

    # The paint
    outer_box_left = Rectangle((0, 17), 19, 16, linewidth=lw, color=color, fill=False)
    inner_box_left = Rectangle((0, 19), 19, 12, linewidth=lw, color=color, fill=False)
    outer_box_right = Rectangle((74.996, 17), 19, 16, linewidth=lw, color=color, fill=False)
    inner_box_right = Rectangle((74.996, 19), 19, 12, linewidth=lw, color=color, fill=False)
    # outer_box_left = Rectangle((0, 17), 19, 16, linewidth=lw, color=color, fill=False)
    # inner_box_left = Rectangle((0, 19), 19, 12, linewidth=lw, color=color, fill=False)
    # outer_box_right = Rectangle((75, 17), 19, 16, linewidth=lw, color=color, fill=False)
    # inner_box_right = Rectangle((75, 19), 19, 12, linewidth=lw, color=color, fill=False)
    ax.add_patch(outer_box_left)
    ax.add_patch(outer_box_right)
    ax.add_patch(inner_box_left)
    ax.add_patch(inner_box_right)

    # Free Throw Lines
    top_free_throw_left = Arc((19, 25), 12, 12, theta1=-90, theta2=90, linewidth=lw, color=color, fill=False)
    top_free_throw_right = Arc((74.996, 25), 12, 12, theta1=90, theta2=-90, linewidth=lw, color=color, fill=False)
    bottom_free_throw_left = Arc((19, 25), 12, 12, theta1=90, theta2=-90, linewidth=lw, color=color, fill=False,
                                 linestyle='dashed')
    bottom_free_throw_right = Arc((74.996, 25), 12, 12, theta1=-90, theta2=90, linewidth=lw, color=color, fill=False,
                                  linestyle='dashed')
    # top_free_throw_left = Arc((19, 25), 12, 12, theta1=-90, theta2=90, linewidth=lw, color=color, fill=False)
    # top_free_throw_right = Arc((75, 25), 12, 12, theta1=90, theta2=-90, linewidth=lw, color=color, fill=False)
    # bottom_free_throw_left = Arc((19, 25), 12, 12, theta1=90, theta2=-90, linewidth=lw, color=color, fill=False,
    #                              linestyle='dashed')
    # bottom_free_throw_right = Arc((75, 25), 12, 12, theta1=-90, theta2=90, linewidth=lw, color=color, fill=False,
    #                               linestyle='dashed')
    ax.add_patch(top_free_throw_left)
    ax.add_patch(top_free_throw_right)
    ax.add_patch(bottom_free_throw_left)
    ax.add_patch(bottom_free_throw_right)

    # Three-point lines
    top_three_left = Rectangle((0, 47), 14, 0, linewidth=lw, color=color, fill=False)
    bottom_three_left = Rectangle((0, 3), 14, 0, linewidth=lw, color=color, fill=False)
    three_arc_left = Arc((5.2493, 25), 47.5, 47.5, theta1=-68.3, theta2=68.3, linewidth=lw, color=color, fill=False)
    # top_three_left = Rectangle((0, 47), 14, 0, linewidth=lw, color=color, fill=False)
    # bottom_three_left = Rectangle((0, 3), 14, 0, linewidth=lw, color=color, fill=False)
    # three_arc_left = Arc((5.25, 25), 47.5, 47.5, theta1=-68.3, theta2=68.3, linewidth=lw, color=color, fill=False)
    ax.add_patch(top_three_left)
    ax.add_patch(bottom_three_left)
    ax.add_patch(three_arc_left)
    top_three_right = Rectangle((79.996, 47), 14, 0, linewidth=lw, color=color, fill=False)
    bottom_three_right = Rectangle((79.996, 3), 14, 0, linewidth=lw, color=color, fill=False)
    three_arc_right = Arc((88.7467, 25), 47.5, 47.5, theta1=111.7, theta2=-111.7, linewidth=lw, color=color, fill=False)
    # top_three_right = Rectangle((80, 47), 14, 0, linewidth=lw, color=color, fill=False)
    # bottom_three_right = Rectangle((80, 3), 14, 0, linewidth=lw, color=color, fill=False)
    # three_arc_right = Arc((88.75, 25), 47.5, 47.5, theta1=111.7, theta2=-111.7, linewidth=lw, color=color, fill=False)
    ax.add_patch(top_three_right)
    ax.add_patch(bottom_three_right)
    ax.add_patch(three_arc_right)

    # Backboard and hoops
    hoop_left = Circle((5.2493, 25), 0.75, linewidth=lw, color=color, fill=False)
    backboard_left = Rectangle((4, 22), 0, 6, linewidth=lw, color=color, fill=False)
    hoop_right = Circle((88.7467, 25), 0.75, linewidth=lw, color=color, fill=False)
    backboard_right = Rectangle((89.996, 22), 0, 6, linewidth=lw, color=color, fill=False)
    # hoop_left = Circle((5.25, 25), 0.75, linewidth=lw, color=color, fill=False)
    # backboard_left = Rectangle((4, 22), 0, 6, linewidth=lw, color=color, fill=False)
    # hoop_right = Circle((88.75, 25), 0.75, linewidth=lw, color=color, fill=False)
    # backboard_right = Rectangle((90, 22), 0, 6, linewidth=lw, color=color, fill=False)
    ax.add_patch(hoop_left)
    ax.add_patch(backboard_left)
    ax.add_patch(hoop_right)
    ax.add_patch(backboard_right)

    # Restricted Area
    restricted_left = Arc((5.2493, 25), 8, 8, theta1=-90, theta2=90, linewidth=lw, color=color, fill=False)
    restricted_right = Arc((88.7467, 25), 8, 8, theta1=90, theta2=-90, linewidth=lw, color=color, fill=False)
    # restricted_left = Arc((5.25, 25), 8, 8, theta1=-90, theta2=90, linewidth=lw, color=color, fill=False)
    # restricted_right = Arc((88.75, 25), 8, 8, theta1=90, theta2=-90, linewidth=lw, color=color, fill=False)
    ax.add_patch(restricted_left)
    ax.add_patch(restricted_right)

    return ax


def loss_time(train_times, train_loss, val_times, val_loss, low_index=None, fp_fig=None):
    fig = plt.figure(figsize=(8, 8))

    t_t = copy.deepcopy(train_times)
    v_t = copy.deepcopy(val_times)

    # Cumsum for times
    for res in range(1, len(v_t)):
        last = v_t[res - 1][-1][-1]
        for epoch in range(len(v_t[res])):
            t_t[res][epoch] = [prev + last for prev in t_t[res][epoch]]
            v_t[res][epoch] = [prev + last for prev in v_t[res][epoch]]

    lines = [row[-1][-1] for row in v_t]
    plt.plot(np.concatenate(t_t).ravel(), np.concatenate(train_loss).ravel(), label='Train')
    plt.plot(np.concatenate(v_t).ravel(), np.concatenate(val_loss).ravel(), label='Val')
    for i, x in enumerate(lines[:-1]):
        if low_index is not None and i == (low_index - 1):
            color = 'r'
        else:
            color = 'k'
        plt.axvline(x, color=color, linestyle=':')
    plt.legend(loc='upper right')
    plt.xlabel('Time[s]')
    plt.ylabel('Loss')
    plt.xlim(xmin=0)
    if fp_fig is not None:
        plt.savefig(fp_fig)
    return fig


def F1_time(times, F1, low_index=None, fp_fig=None):
    fig = plt.figure(figsize=(8, 8))

    t = copy.deepcopy(times)

    # Cumsum for times
    for res in range(1, len(t)):
        last = t[res - 1][-1][-1]
        for epoch in range(len(t[res])):
            t[res][epoch] = [prev + last for prev in t[res][epoch]]

    lines = [row[-1][-1] for row in t]
    plt.plot(np.concatenate(t).ravel(), np.concatenate(F1).ravel())
    for i, x in enumerate(lines[:-1]):
        if low_index is not None and i == (low_index - 1):
            color = 'r'
        else:
            color = 'k'
        plt.axvline(x, color=color, linestyle=':')
    plt.xlabel('Time[s]')
    plt.ylabel('Val F1')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    if fp_fig is not None:
        plt.savefig(fp_fig)
    return fig


def latent_factor_heatmap(X, draw_court, normalize=True, cmap='RdBu_r', fp_fig=None):
    if draw_court:
        X = utils.finegrain(X, [40, 50], 0)

    shape = X.shape

    if normalize:
        if draw_court:
            scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X.view(-1, shape[-1]).cpu().numpy()).reshape(*shape)
    else:
        X = X.cpu().numpy()

    fig = plt.figure(figsize=(12, 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(shape[-1] // 5, 5),
                     axes_pad=0.0, share_all=True)
    for k, ax in enumerate(grid):
        # ax.set_title('K={0}'.format(k + 1))
        if draw_court:
            draw_half_court_left(ax)
            data = X[..., k - 1].transpose()
            im = ax.imshow(data, cmap=cmap, origin='lower')
        else:
            ax.add_patch(Circle((X.shape[0] / 2 - 0.5, X.shape[1] / 6 - 0.5), 0.08 * X.shape[0], ec='k', fc='g'))
            data = X[..., k - 1].transpose()
            im = ax.imshow(data, cmap=cmap, origin='lower')


    # This affects all axes as share_all = True.
    grid.axes_llc.set_xticks([])
    grid.axes_llc.set_yticks([])

    if fp_fig is not None:
        plt.savefig(fp_fig)
    return fig

def latent_factor_polar_heatmap(X, fig_dir, b, c, draw_court, low=True, normalize=True, cmap='RdBu_r'):
    b_str = utils.size_to_str(b)
    c_str = utils.size_to_str(c)
    shape = X.shape

    if normalize:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X.view(-1, shape[-1]).cpu().numpy()).reshape(*shape)
    else:
        X = X.cpu().numpy()

    for k in range(shape[-1]):
        # ax.set_title('K={0}'.format(k + 1))
        fig = plt.figure()
        if draw_court:
            ax_court = fig.add_axes([0.14, 1/7, 0.42, 5/7])
            draw_half_court_left(ax_court)
            ax_court.set_xticks([])
            ax_court.set_yticks([])
            
            data = X[..., k - 1].transpose()

            ax_polar = fig.add_axes([0, 0, 1, 1], polar=True, frameon=False)
            rad = np.linspace(0, 36, 100)
            azm = np.linspace(0, np.pi, 180)
            r, th = np.meshgrid(rad, azm)
            heatmap = np.zeros((180, 100))
            for i in range(heatmap.shape[0]):
                for j in range(heatmap[0].shape[0]):
                    heatmap[i][j] = data[i*b[1]//180][j*b[0]//100]
    
            ax_polar.pcolormesh(th, r, heatmap, shading='flat', cmap=cmap, alpha=0.3)
            ax_polar.plot(azm, r, ls='none')

            plt.thetagrids(range(0, 181, 180//b[1]))
            plt.rgrids(range(0, 37, 36//b[0]))
            ax_polar.set_thetamin(0)
            ax_polar.set_thetamax(180)
            ax_polar.set_rmin(0)
            ax_polar.set_rmax(36)
            ax_polar.set_theta_zero_location('S')
            ax_polar.set_anchor('W')
            ax_polar.set_xticks([])
            ax_polar.set_yticks([])
            plt.grid()
            
            k_str = str(k+1)
            
            if low:
                fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_B_heatmap({2}).png".format(b_str, c_str, k_str))
            else:
                fp_fig = os.path.join(fig_dir,
                          "full_{0},{1}_B_heatmap({2}).png".format(b_str, c_str, k_str))
                
            if fp_fig is not None:
                plt.savefig(fp_fig)
            
        else:
            data = X[..., k - 1].transpose()

            ax_polar = fig.add_axes([0, 0, 1, 1], polar=True)
            rad = np.linspace(0, 6, 100)
            azm = np.linspace(0, 2 * np.pi, 360)
            r, th = np.meshgrid(rad, azm)
            heatmap = np.zeros((360, 100))
            for i in range(heatmap.shape[0]):
                for j in range(heatmap[0].shape[0]):
                    heatmap[((i - 15) % 360)][j] = data[i*c[1]//360][j*c[0]//100]
    
            ax_polar.pcolormesh(th, r, heatmap, shading='flat', cmap=cmap)
            ax_polar.plot(azm, r, ls='none')

            plt.thetagrids(range(180//c[1], 360 - (180//c[1]) + 1, 360//c[1]))
            plt.rgrids(range(0, 7, 6//c[0]))
            ax_polar.set_thetamin(0)
            ax_polar.set_thetamax(360)
            ax_polar.set_rmin(0)
            ax_polar.set_rmax(6)
            ax_polar.set_theta_zero_location('N')
            ax_polar.set_xticks([])
            ax_polar.set_yticks([])
            plt.grid()
            
            k_str = str(k+1)
            
            if low:
                fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_C_heatmap({2}).png".format(b_str, c_str, k_str))
            else:
                fp_fig = os.path.join(fig_dir,
                          "full_{0},{1}_C_heatmap({2}).png".format(b_str, c_str, k_str))
                
            if fp_fig is not None:
                plt.savefig(fp_fig)
                
        plt.close()