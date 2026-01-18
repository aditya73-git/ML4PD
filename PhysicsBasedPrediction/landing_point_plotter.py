from sys import prefix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_landing_points_for_throw(data, impact_log, throw_id, output_folder=None, show=True):
    
    dataset = data.copy()

    throw_data = dataset[dataset['throw_id'] == throw_id].sort_values('t').copy()
    
    ground_truth_impact_point = throw_data[(throw_data['t'] > 0.0)]

    pred_impact_points = throw_data.copy()
    real_impact_points = impact_log.copy()
    impact_id_offset = 20
    impact_id = throw_id + impact_id_offset
    real_impact_points = real_impact_points[real_impact_points['ID'] == impact_id]

    real_impact_point_x = real_impact_points.iloc[0]['X_cm'] / 100.0
    real_impact_point_y = real_impact_points.iloc[0]['Y_cm'] / 100.0

    ground_truth_impact_point_x = ground_truth_impact_point.iloc[0]['x_land_grand_truth']
    ground_truth_impact_point_y = ground_truth_impact_point.iloc[0]['y_land_grand_truth']

    # Color gradient based on time: early = light, late = dark
    # Normalize t to [0,1]
    t_time = pred_impact_points['t'].values
    t_min, t_max = np.min(t_time), np.max(t_time)
    t_norm = (t_time - t_min) / (t_max - t_min)


    plt.figure(figsize=(7, 7))

    # Predicted points with blue gradient
    plt.scatter(
        pred_impact_points['x_land_pred'].values,
        pred_impact_points['y_land_pred'].values,
        c=t_norm,
        cmap='Blues',
        s=35,
        edgecolors='black',
        label='Predicted landing points'
    )

    # Ground truth point as a red dot
    plt.scatter(
        [ground_truth_impact_point_x],
        [ground_truth_impact_point_y],
        c='red',
        s=80,
        marker='o',
        edgecolors='black',
        linewidths=0.6,
        label='Ground truth landing point'
    )

    # Measured impact point (green)
    plt.scatter(
        [real_impact_point_x],
        [real_impact_point_y],
        c='green',
        s=90,
        marker='o',
        edgecolors='black',
        linewidths=0.6,
        label='Measured impact (log)'
    )

    all_impact_points_x = np.concatenate([pred_impact_points['x_land_pred'].values, np.array([ground_truth_impact_point_x])])
    all_impact_points_y = np.concatenate([pred_impact_points['y_land_pred'].values, np.array([ground_truth_impact_point_y])])

    max_abs = max(np.max(np.abs(all_impact_points_x)), np.max(np.abs(all_impact_points_y)))
    lim = max_abs * 1.1 if max_abs > 0 else 1.0

    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Landing Point Predictions for Throw {throw_id}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, f"landing_points_throw_{throw_id}.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_landing_error_over_time_for_throw(data, impact_log, throw_id, output_folder=None, show=True):

    dataset = data.copy()

    throw_data = dataset[dataset['throw_id'] == throw_id].sort_values('t').copy()
    
    ground_truth_impact_point = throw_data[(throw_data['t'] > 0.0)]

    pred_impact_points = throw_data.copy()
    real_impact_points = impact_log.copy()
    impact_id_offset = 20
    impact_id = throw_id + impact_id_offset
    real_impact_points = real_impact_points[real_impact_points['ID'] == impact_id]

    real_impact_point_x = real_impact_points.iloc[0]['X_cm'] / 100.0
    real_impact_point_y = real_impact_points.iloc[0]['Y_cm'] / 100.0

    #ground_truth_impact_point_x = ground_truth_impact_point.iloc[0]['x_land_grand_truth']
    #ground_truth_impact_point_y = ground_truth_impact_point.iloc[0]['y_land_grand_truth']

    # Color gradient based on time: early = light, late = dark
    # Normalize t to [0,1]
    t_time = pred_impact_points['t'].values
    t_min, t_max = np.min(t_time), np.max(t_time)
    t_norm = (t_time - t_min) / (t_max - t_min)

    error_x = pred_impact_points['x_land_pred'].values - real_impact_point_x
    error_y = pred_impact_points['y_land_pred'].values - real_impact_point_y
    error = np.sqrt(error_x**2 + error_y**2)


    t_land_theoretical = ground_truth_impact_point.iloc[0]['t'] + ground_truth_impact_point.iloc[0]['t_to_land_grand_truth']

    plt.scatter(
        t_time,
        error,
        c=t_norm,
        cmap='Blues',
        s=35,
        edgecolors='black',
        label='Prediction error'
    )

    plt.axvline(t_land_theoretical, color='red', linewidth=2, label='Theoretical landing time')

    plt.xlabel("t (s)")
    plt.ylabel("Landing point error (m)")
    plt.title(f"Landing Prediction Error Over Time (Throw {throw_id})")
    plt.legend()

    plt.grid(True, alpha=0.3)

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, f"landing_error_throw_{throw_id}.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()