from sys import prefix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
import os
import utilities as utils

def plot_landing_points_for_throw(data, impact_log, throw_id, output_folder=None, show=True):
    
    dataset = data.copy()

    throw_data = dataset[dataset['throw_id'] == throw_id].sort_values('t').copy()
    
    ground_truth_impact_point = throw_data[(throw_data['t'] > 0.0)]

    pred_impact_points = throw_data.copy()
    real_impact_points = impact_log.copy()
    impact_id_offset = 20
    impact_id = throw_id + impact_id_offset
    real_impact_points_for_ID = real_impact_points[real_impact_points['ID'] == impact_id]

    real_impact_point_x = real_impact_points_for_ID.iloc[0]['X_cm'] / 100.0
    real_impact_point_y = real_impact_points_for_ID.iloc[0]['Y_cm'] / 100.0

    ground_truth_impact_point_x = ground_truth_impact_point.iloc[0]['x_land_grand_truth']
    ground_truth_impact_point_y = ground_truth_impact_point.iloc[0]['y_land_grand_truth']

    # Color gradient based on time: early = light, late = dark
    # Normalize t to [0,1]
    t_time = pred_impact_points['t'].values
    t_min, t_max = np.min(t_time), np.max(t_time)
    t_norm = (t_time - t_min) / (t_max - t_min)


    plt.figure(figsize=(7, 7))

    # Predicted points with blue gradient
    sc = plt.scatter( pred_impact_points['x_land_pred'].values, pred_impact_points['y_land_pred'].values, c=t_norm, cmap='Blues', s=35, edgecolors='black', label='Predicted landing points')
    cbar = plt.colorbar(sc, fraction=0.035, pad=0.04)
    cbar.set_label("Frame progression\n(light = first, dark = last)")

    # Ground truth point as a red dot
    plt.scatter([ground_truth_impact_point_x], [ground_truth_impact_point_y], c='red', s=80, marker='o', edgecolors='black', linewidths=0.6, label='Ground truth landing point')

    # Measured impact point (green)
    plt.scatter([real_impact_point_x], [real_impact_point_y], c='green', s=90, marker='o', edgecolors='black', linewidths=0.6, label='Measured impact (log)')

    all_impact_points_x = np.concatenate([dataset.loc[dataset['vel_z_measured'] != 0.0,'x_land_pred'].values, np.array(real_impact_points['X_cm'] / 100.0)])
    all_impact_points_y = np.concatenate([dataset.loc[dataset['vel_z_measured'] != 0.0,'y_land_pred'].values, np.array(real_impact_points['Y_cm'] / 100.0)])

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

    #if show:
    #    plt.show()
    #else:
    #    plt.close()

def plot_landing_points_for_throw_with_RMS(data, impact_log, throw_id, output_folder=None, show=True):
    
    dataset = data.copy()

    throw_data = dataset[dataset['throw_id'] == throw_id].sort_values('t').copy()
    
    ground_truth_impact_point = throw_data[(throw_data['t'] > 0.0)]

    pred_impact_points = throw_data.copy()
    real_impact_points = impact_log.copy()
    impact_id_offset = 20
    impact_id = throw_id + impact_id_offset
    real_impact_points_for_ID = real_impact_points[real_impact_points['ID'] == impact_id]

    real_impact_point_x = real_impact_points_for_ID.iloc[0]['X_cm'] / 100.0
    real_impact_point_y = real_impact_points_for_ID.iloc[0]['Y_cm'] / 100.0

    ground_truth_impact_point_x = ground_truth_impact_point.iloc[0]['x_land_grand_truth']
    ground_truth_impact_point_y = ground_truth_impact_point.iloc[0]['y_land_grand_truth']

    # Color gradient based on time: early = light, late = dark
    # Normalize t to [0,1]
    t_time = pred_impact_points['t'].values
    t_min, t_max = np.min(t_time), np.max(t_time)
    t_norm = (t_time - t_min) / (t_max - t_min)

    # --- 6c: uncertainty from v_err ---
    # Compute sigma for this throw (you can switch to global by passing throw_id=None)
    sigma_vx, sigma_vy, sigma_vz, sigma_vxy = utils.calc_velocity_sigmas_from_v_err(dataset, throw_id=throw_id)

    # We need time-to-land per frame for circle radius. Prefer an existing column if you already computed it.
    t_to_land = pred_impact_points['t_to_land_pred'].values


    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    # Predicted points with blue gradient
    plt.scatter( pred_impact_points['x_land_pred'].values, pred_impact_points['y_land_pred'].values, c=t_norm, cmap='Blues', s=35, edgecolors='black', label='Predicted landing points')

    # Uncertainty circles per frame (radius grows with remaining time)
    # r(t) = t_to_land(t) * sigma_vxy
    # To avoid a solid blob, we keep alpha low and optionally skip some frames.

    if np.isfinite(sigma_vx) and np.isfinite(sigma_vy):
        step = 1  # set to 2 or 3 if plot is too cluttered

        for i in range(0, len(pred_impact_points), step):

            # skip the first frame (t==0)
            if pred_impact_points.iloc[i]['t'] <= 0.0:
                continue

            x_c = pred_impact_points.iloc[i]['x_land_pred']
            y_c = pred_impact_points.iloc[i]['y_land_pred']

            if not np.isfinite(x_c) or not np.isfinite(y_c) or not np.isfinite(t_to_land[i]):
                continue

            # propagate velocity uncertainty to landing uncertainty
            sigma_x = t_to_land[i] * sigma_vx
            sigma_y = t_to_land[i] * sigma_vy

            width = 2.0 *  sigma_x
            height = 2.0 * sigma_y

            # If you want the ellipses to stay visible, clamp the colormap
            t_adjusted = 0.3 + 0.7 * t_norm[i]
            color = plt.cm.Blues(t_adjusted)

            e = Ellipse(
                (x_c, y_c),
                width=width,
                height=height,
                angle=0.0,          # axis-aligned ellipse
                fill=False,
                edgecolor=color,
                linewidth=1.5,
                alpha=0.45
            )

            e.set_zorder(1)
            ax.add_patch(e)

    # Ground truth point as a red dot
    plt.scatter([ground_truth_impact_point_x], [ground_truth_impact_point_y], c='red', s=80, marker='o', edgecolors='black', linewidths=0.6, label='Ground truth landing point')

    # Measured impact point (green)
    plt.scatter([real_impact_point_x], [real_impact_point_y], c='green', s=90, marker='o', edgecolors='black', linewidths=0.6, label='Measured impact (log)')

    all_impact_points_x = pred_impact_points.loc[pred_impact_points['vel_z_measured'] != 0.0,'x_land_pred'].values
    all_impact_points_y = pred_impact_points.loc[pred_impact_points['vel_z_measured'] != 0.0,'y_land_pred'].values

    max_abs = max(np.max(np.abs(all_impact_points_x)), np.max(np.abs(all_impact_points_y)))
    lim = max_abs * 1.5 if max_abs > 0 else 1.0

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

    #if show:
    #    plt.show()
    #else:
    #    plt.close()

def plot_landing_error_over_time_for_throw(data, impact_log, throw_id, output_folder=None, show=True):

    dataset = data.copy()
    real_impact_points = impact_log.copy()

    throw_data = dataset[dataset['throw_id'] == throw_id].sort_values('t').copy()

    plt.figure(figsize=(7, 7))
    
    ground_truth_impact_point = throw_data[(throw_data['t'] > 0.0)]

    pred_impact_points = throw_data.copy()
    impact_id_offset = 20
    impact_id = throw_id + impact_id_offset
    real_impact_points = real_impact_points[real_impact_points['ID'] == impact_id]

    real_impact_point_x = real_impact_points.iloc[0]['X_cm'] / 100.0
    real_impact_point_y = real_impact_points.iloc[0]['Y_cm'] / 100.0

    #ground_truth_impact_point_x = ground_truth_impact_point.iloc[0]['x_land_grand_truth']
    #ground_truth_impact_point_y = ground_truth_impact_point.iloc[0]['y_land_grand_truth']

    # Color gradient based on time: early = light, late = dark
    # Normalize t to [0,1]
    t_time = pred_impact_points['t_to_land_pred'].values
    t_min, t_max = np.min(t_time), np.max(t_time)
    t_norm = 1.0 - (t_time - t_min) / (t_max - t_min)

    error_x = pred_impact_points['x_land_pred'].values - real_impact_point_x
    error_y = pred_impact_points['y_land_pred'].values - real_impact_point_y
    error = np.sqrt(error_x**2 + error_y**2)



    sc = plt.scatter(t_time, error, c=t_norm, cmap='Blues', s=35, edgecolors='black', label='Prediction error')
    cbar = plt.colorbar(sc, fraction=0.035, pad=0.04)
    cbar.set_label("Frame progression\n(light = first, dark = last)")

    plt.xlabel("Time to land t_to_land (s)")
    plt.ylabel("Landing point error (m)")
    plt.title(f"Landing Prediction Error Over Time (Throw {throw_id})")
    plt.legend()

    plt.grid(True, alpha=0.3)

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, f"landing_error_throw_{throw_id}.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')

    #if show:
    #    plt.show()
    #else:
    #    plt.close()

def plot_impact_log_data(data, impact_log, output_folder=None, show=True):

    dataset = data.copy()

    real_impact_points = impact_log.copy()
    impact_id = 21 #We need all the impacts
    real_impact_points = real_impact_points[real_impact_points['ID'] >= impact_id]

    real_impact_point_x = real_impact_points['X_cm'] / 100.0
    real_impact_point_y = real_impact_points['Y_cm'] / 100.0

    plt.figure(figsize=(7, 7))

    # Measured impact point (green)
    plt.scatter([real_impact_point_x], [real_impact_point_y], c='green', s=90, marker='o', edgecolors='black', linewidths=0.6, label='Measured impact (log)')

    all_impact_points_x = np.concatenate([dataset.loc[dataset['vel_z_measured'] != 0.0,'x_land_pred'].values, np.array(real_impact_point_x)])
    all_impact_points_y = np.concatenate([dataset.loc[dataset['vel_z_measured'] != 0.0,'y_land_pred'].values, np.array(real_impact_point_y)])
    
    max_abs = max(np.max(np.abs(all_impact_points_x)), np.max(np.abs(all_impact_points_y)))
    lim = max_abs * 1.1

    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"All recorded landing points")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()

    plt.grid(True, alpha=0.3)

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, f"i_points_all.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')

    #if show:
    #    plt.show()
    #else:
    #    plt.close()

def plot_landing_error_all_throws(data, impact_log, output_folder=None, show=True):
    """
    Plots landing point prediction error over time for ALL throws in one figure.

    - Error = distance between predicted and ground truth landing points
    - Each throw is plotted independently
    - Same blue gradient logic (early = light, late = dark)
    """

    dataset = data[data['vel_z_measured'] != 0.0].copy()
    real_impact_points = impact_log.copy()

    all_t = []
    all_error = []

    plt.figure(figsize=(9, 5))

    for throw_id in sorted(dataset['throw_id'].unique()):
        throw_data = dataset[dataset['throw_id'] == throw_id].sort_values('t').copy()
        pred_impact_points = throw_data.copy()

        impact_id_offset = 20
        impact_id = throw_id + impact_id_offset
        real_impact_point_for_throw = real_impact_points[real_impact_points['ID'] == impact_id]

        real_impact_point_x = real_impact_point_for_throw.iloc[0]['X_cm'] / 100.0
        real_impact_point_y = real_impact_point_for_throw.iloc[0]['Y_cm'] / 100.0

        error_x = pred_impact_points['x_land_pred'].values - real_impact_point_x
        error_y = pred_impact_points['y_land_pred'].values - real_impact_point_y
        error = np.sqrt(error_x**2 + error_y**2)

        t_time = pred_impact_points['t_to_land_pred'].values

        # Collect for global statistics
        all_t.extend(t_time)
        all_error.extend(error)

        #Color Gradient per throw.
        t_min, t_max = np.min(t_time), np.max(t_time)
        t_norm = 1 - (t_time - t_min) / (t_max - t_min)

        plt.scatter(t_time, error, c=t_norm, cmap='Blues', s=25, edgecolors='black', alpha=0.7)

    plt.xlabel("Time to land (s)")
    plt.ylabel("Landing point error (m)")
    plt.title("Landing Prediction Error Over Time (All Throws)")
    plt.ylim(bottom=-0.02, top=0.5)
    plt.axhline(0.5, linestyle='--', linewidth=1) #Error values above 0.5 m are clipped for readability
    plt.grid(True, alpha=0.3)

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, "landing_error_scatter_all_throws.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')

    plt.close()

    # ---- Binning & statistics (this is the key addition) ----

    all_t = np.array(all_t)
    all_error = np.array(all_error)

    n_bins = 8
    bins = np.linspace(all_t.min(), all_t.max(), n_bins + 1)

    binned_errors = []
    bin_labels = []

    for i in range(n_bins):
        mask = (all_t >= bins[i]) & (all_t < bins[i + 1])
        binned_errors.append(all_error[mask])
        bin_labels.append(f"{bins[i]:.2f}–{bins[i+1]:.2f}")

    plt.figure(figsize=(10, 5))

    plt.boxplot(
        binned_errors,
        labels=bin_labels,
        showfliers=True
    )

    means = [np.mean(err) if len(err) > 0 else np.nan for err in binned_errors]
    for i, mean in enumerate(means):
        if np.isfinite(mean):
            plt.text(
                i + 1,          # boxplot positions start at 1
                -0.014,
                f"{mean:.3f}",  # format to 3 decimals
                ha='center',
                va='bottom',
                fontsize=8,
                color='red'
            )

    plt.scatter(
        range(1, len(means) + 1),
        means,
        color='red',
        s=25,
        zorder=3,
        label='Mean error'
    )

    plt.legend()

    plt.xlabel("Time to land (s)")
    plt.ylabel("Landing point error (m)")
    plt.title("Landing Prediction Error Over Time (All Throws)")
    plt.ylim(bottom=-0.02, top=0.5)
    plt.axhline(0.5, linestyle='--', linewidth=1) #Error values above 0.5 m are clipped for readability
    plt.grid(True, alpha=0.3)

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, "landing_error_all_throws.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')

    #if show:
    #    plt.show()
    #else:
    #    plt.close()

def plot_landing_error_all_throws_with_corr(data, impact_log, output_folder=None, show=True):
    """
    
    """

    dataset = data[data['vel_z_measured'] != 0.0].copy()
    real_impact_points = impact_log.copy()

    all_t = []
    all_error = []

    plt.figure(figsize=(9, 5))

    for throw_id in sorted(dataset['throw_id'].unique()):
        throw_data = dataset[dataset['throw_id'] == throw_id].sort_values('t').copy()
        pred_impact_points = throw_data.copy()

        impact_id_offset = 20
        impact_id = throw_id + impact_id_offset
        real_impact_point_for_throw = real_impact_points[real_impact_points['ID'] == impact_id]

        real_impact_point_x = real_impact_point_for_throw.iloc[0]['X_cm'] / 100.0
        real_impact_point_y = real_impact_point_for_throw.iloc[0]['Y_cm'] / 100.0

        error_x = pred_impact_points['x_land_corr'].values - real_impact_point_x
        error_y = pred_impact_points['y_land_corr'].values - real_impact_point_y
        error = np.sqrt(error_x**2 + error_y**2)

        t_time = pred_impact_points['t_to_land_pred'].values

        # Collect for global statistics
        all_t.extend(t_time)
        all_error.extend(error)

        #Color Gradient per throw.
        t_min, t_max = np.min(t_time), np.max(t_time)
        t_norm = 1 - (t_time - t_min) / (t_max - t_min)

        plt.scatter(t_time, error, c=t_norm, cmap='Blues', s=25, edgecolors='black', alpha=0.7)

    plt.xlabel("Time to land (s)")
    plt.ylabel("Landing point error (m)")
    plt.title("Corrigated Landing Prediction Error Over Time (All Throws)")
    plt.ylim(bottom=-0.02, top=0.5)
    plt.axhline(0.5, linestyle='--', linewidth=1) #Error values above 0.5 m are clipped for readability
    plt.grid(True, alpha=0.3)

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, "landing_error_scatter_all_throws_corr.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')

    plt.close()

    # ---- Binning & statistics (this is the key addition) ----
    all_t = np.array(all_t)
    all_error = np.array(all_error)

    n_bins = 8
    bins = np.linspace(all_t.min(), all_t.max(), n_bins + 1)

    binned_errors = []
    bin_labels = []

    for i in range(n_bins):
        mask = (all_t >= bins[i]) & (all_t < bins[i + 1])
        binned_errors.append(all_error[mask])
        bin_labels.append(f"{bins[i]:.2f}–{bins[i+1]:.2f}")

    plt.figure(figsize=(10, 5))

    plt.boxplot(
        binned_errors,
        labels=bin_labels,
        showfliers=True
    )

    means = [np.mean(err) if len(err) > 0 else np.nan for err in binned_errors]
    for i, mean in enumerate(means):
        if np.isfinite(mean):
            plt.text(
                i + 1,          # boxplot positions start at 1
                -0.014,
                f"{mean:.3f}",  # format to 3 decimals
                ha='center',
                va='bottom',
                fontsize=8,
                color='red'
            )

    plt.scatter(
        range(1, len(means) + 1),
        means,
        color='red',
        s=25,
        zorder=3,
        label='Mean error'
    )

    plt.legend()

    plt.xlabel("Time to land (s)")
    plt.ylabel("Landing point error (m)")
    plt.title("Corrigated Landing Prediction Error Over Time (All Throws)")
    plt.ylim(bottom=-0.02, top=0.5)
    plt.axhline(0.5, linestyle='--', linewidth=1) #Error values above 0.5 m are clipped for readability
    plt.grid(True, alpha=0.3)

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, "landing_error_all_throws_corr.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')

def plot_systematic_offset(data, impact_log, output_folder = None, show = True):

    dataset = data.copy()
    real_impact_points = impact_log.copy()
    offsets = []

    plt.figure(figsize=(9, 5))

    for throw_id, group in dataset.groupby('throw_id'):
        # Ground truth landing point (same for all rows except t==0)
        ground_truth_impact_point = group[group['x_land_grand_truth'].notna() & group['y_land_grand_truth'].notna()]

        ground_truth_impact_point_x = ground_truth_impact_point.iloc[0]['x_land_grand_truth']
        ground_truth_impact_point_y = ground_truth_impact_point.iloc[0]['y_land_grand_truth']

        # Real impact point
        impact_row = impact_log[impact_log['ID'] == throw_id]
        if len(impact_row) == 0:
            continue

        real_impact_point_x = impact_row.iloc[0]['X_cm'] / 100.0
        real_impact_point_y = impact_row.iloc[0]['Y_cm'] / 100.0

        offset = np.sqrt((real_impact_point_x - ground_truth_impact_point_x)**2 + (real_impact_point_y - ground_truth_impact_point_y)**2)

        offsets.append({'ID': throw_id, 'offset (m)': offset})
        
        plt.bar(throw_id, offset, color='steelblue')

    plt.xlabel("Throw ID")
    plt.ylabel("Systematic offset (m)")
    plt.title("Systematic Landing-Point Offset per Throw")
    plt.grid(axis='y', alpha=0.3)

    offset_df = pd.DataFrame(offsets)

    output_folder_csv = "PhysicsBasedPrediction"

    os.makedirs(output_folder_csv, exist_ok=True)

    csv_path = os.path.join(output_folder_csv, "systematic_offset_per_throw.csv")
    offset_df.to_csv(csv_path, index=False)

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, f"landing_points_all.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')

    #if show:
    #    plt.show()
    #else:
    #    plt.close()
