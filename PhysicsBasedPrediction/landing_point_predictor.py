### Physics Based Prediction

# Initial Setup
from sys import prefix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import landing_point_plotter as plotter

# Input and output path and folders
#LOCATION_DATA_CSV = "Location Tracking/beer_pong_trajectories.csv" #Speed tracking csv includes this dataset as well
SPEED_DATA_CSV = "Speed Tracking/beer_pong_velocity_output.csv"
OUTPUT_FOLDER = "PhysicsBasedPrediction"
OUTPUT_FOLDER_PER_THROW_IMPACT_FIG = "PhysicsBasedPrediction/impact_points"
OUTPUT_FOLDER_PER_THROW_ERROR_FIG = "PhysicsBasedPrediction/error"
OUTPUT_FOLDER_PER_THROW_ERROR_CORR_FIG = "PhysicsBasedPrediction/corr_rms"
OUTPUT_FOLDER_PER_THROW_ERROR_RMS_FIG = "PhysicsBasedPrediction/error_rms"
OUTPUT_DATA_CSV = "PhysicsBasedPrediction/landing_point_prediction.csv"
IMPACT_DATA_CSV = "Labelling/impact_log.csv"

# Enable interactive plotting
#plt.ion()

# Load dataset 

data = pd.read_csv(SPEED_DATA_CSV)
impact_log = pd.read_csv(IMPACT_DATA_CSV)

# 6a: Predict landing point

def calc_landing_point_from(data, g=9.81):
    """
    Takes current velocity and current location and calculates landing point
    using simple quadratic equation (gravity only, no drag):

    z(t) = z0 + vz0*t - 1/2*g*t^2
    solve for z (0) to get t_land
    then solve for x_land = x0 + vx0*t_land and the same for y
 
    Inputs (from csv):
        x_raw, y_raw, z_raw
        vel_x_measured, vel_y_measured, vel_z_measured

    Output:
        Adds per-frame landing point prediction:
            x_land_pred, y_land_pred  (z_land_pred is assumed to be always 0)
        Also adds:
            t_land_pred  (time until impact)
    """

    dataset = data.copy()

    # Current state (per frame)
    x0 = dataset['x_raw'].values
    y0 = dataset['y_raw'].values
    z0 = dataset['z_raw'].values

    vx = dataset['vel_x_measured'].values
    vy = dataset['vel_y_measured'].values
    vz = dataset['vel_z_measured'].values

    # z(t) = z0 + vz0*t - 1/2*g*t^2 = 0
    # a*t^2 + b*t + c = 0 with: a = -0.5*g, b = vz, c = z0
    a = -0.5 * g
    b = vz
    c = z0

    disc = b**2 - 4*a*c

    #solve equation
    t_to_land = (vz + np.sqrt(disc)) / g

    # Landing point in x,y at that time (z will be 0 by construction)
    x_land = x0 + vx * t_to_land
    y_land = y0 + vy * t_to_land

    dataset['x_land_pred'] = x_land
    dataset['y_land_pred'] = y_land
    dataset['t_to_land_pred'] = t_to_land

    return dataset

df = calc_landing_point_from(data)

def calc_landing_point_from_ground_truth(data):

    dataset = data.copy()

    dataset['t_to_land_grand_truth'] = np.nan
    dataset['x_land_grand_truth'] = np.nan
    dataset['y_land_grand_truth'] = np.nan

    for throw_id, throw_group in dataset.groupby('throw_id'):
        throw_group_sorted = throw_group.sort_values('t')

        # second line = first row where t > 0
        throw_group_valid_ground_truth_values_only = throw_group_sorted[throw_group_sorted['t'] > 0.0]

        row = throw_group_valid_ground_truth_values_only.iloc[0]

        coeff_x_1 = row['coeff_x_1']
        coeff_x_0 = row['coeff_x_0']
        coeff_y_1 = row['coeff_y_1']
        coeff_y_0 = row['coeff_y_0']
        coeff_z_2 = row['coeff_z_2']
        coeff_z_1 = row['coeff_z_1']
        coeff_z_0 = row['coeff_z_0']


        disc = coeff_z_1**2 - 4.0 * coeff_z_2 * coeff_z_0
        t_land_ground_truth = (-coeff_z_1 - np.sqrt(disc)) / (2.0 * coeff_z_2)
        # Landing coordinates from fitted x(t), y(t)
        x_land = coeff_x_1 * t_land_ground_truth + coeff_x_0
        y_land = coeff_y_1 * t_land_ground_truth + coeff_y_0

        # Fill into all rows of this throw except the first frame (t==0.0)
        fill_mask = (dataset['throw_id'] == throw_id) & (dataset['t'] > 0.0)

        dataset.loc[fill_mask, 't_to_land_grand_truth'] = t_land_ground_truth - dataset.loc[fill_mask, 't']
        dataset.loc[fill_mask, 'x_land_grand_truth'] = x_land
        dataset.loc[fill_mask, 'y_land_grand_truth'] = y_land

    return dataset

df = calc_landing_point_from_ground_truth(df)
print("Physics based prediction (Task 6a):")
print(df[['throw_id', 't', 'x_land_pred', 'y_land_pred', 't_to_land_pred','t_to_land_grand_truth', 'x_land_grand_truth', 'y_land_grand_truth']].head(30))

# 6b: Calculate error
def calc_landing_error(data, impact_log):

    dataset = data.copy()

    # Initialize error columns
    dataset['x_pred_err'] = np.nan
    dataset['y_pred_err'] = np.nan
    dataset['pred_err'] = np.nan

    for throw_id, group in dataset.groupby('throw_id'):
        # Get real impact point for this throw
        impact_row = impact_log.loc[impact_log['ID'] == throw_id]

        x_real = impact_row.iloc[0]['X_cm'] / 100.0
        y_real = impact_row.iloc[0]['Y_cm'] / 100.0

        #Mask for current throw and predicted landing points
        mask = ((dataset['throw_id'] == throw_id) & dataset['x_land_pred'].notna() & dataset['y_land_pred'].notna())

        dataset.loc[mask, 'x_pred_err'] = dataset.loc[mask, 'x_land_pred'] - x_real
        dataset.loc[mask, 'y_pred_err'] = dataset.loc[mask, 'y_land_pred'] - y_real
        dataset.loc[mask, 'pred_err'] = np.sqrt(dataset.loc[mask, 'x_pred_err']**2 + dataset.loc[mask, 'y_pred_err']**2)

    return dataset

df = calc_landing_error(df, impact_log)

# 6c: Account for Error

def calc_landing_point_corr_from(data, g=9.81, min_points=3, vel_weight=0.5):
    """
    Calculates corrected landing points using current + all previous frames (per throw).

    Idea:
      - Fit position vs time using all frames up to current frame:
          x(t): linear, y(t): linear, z(t): quadratic
      - Fit measured velocity vs time using all frames up to current frame:
          vx(t), vy(t), vz(t): linear
      - Estimate the "current" state at time t_i from the fits, then predict landing:
          z(t_land) = 0 (using fitted z(t), choose root > t_i)
          x_land = x(t_land), y_land = y(t_land)

    Inputs (from csv):
        throw_id, t
        x_raw, y_raw, z_raw
        vel_x_measured, vel_y_measured, vel_z_measured

    Output:
        Adds per-frame corrected landing point prediction:
            x_land_corr, y_land_corr
    """

    dataset = data.copy()

    # Start with fallback = original predictions (so early frames are not NaN)
    dataset['x_land_corr'] = dataset['x_land_pred']
    dataset['y_land_corr'] = dataset['y_land_pred']

    for throw_id, group in dataset.groupby('throw_id'):
        throw = group.sort_values('t').copy()

        t_all = throw['t'].values
        x_all = throw['x_raw'].values
        y_all = throw['y_raw'].values
        z_all = throw['z_raw'].values

        vx_all = throw['vel_x_measured'].values
        vy_all = throw['vel_y_measured'].values
        vz_all = throw['vel_z_measured'].values

        x_corr = np.full_like(t_all, np.nan, dtype=float)
        y_corr = np.full_like(t_all, np.nan, dtype=float)

        for i in range(len(throw)):
            # Need enough history to fit stably
            if i + 1 < min_points:
                continue

            t_hist = t_all[:i+1]

            # Position fits
            coeff_x = np.polyfit(t_hist, x_all[:i+1], 1)  # x = coeff_x1*t + coeff_x0
            coeff_y = np.polyfit(t_hist, y_all[:i+1], 1)  # y = coeff_y1*t + coeff_y0
            coeff_z = np.polyfit(t_hist, z_all[:i+1], 2)  # z = coeff_z2*t^2 + coeff_z1*t + coeff_z0

            # Velocity fits (smooth measured velocities)
            coeff_vx = np.polyfit(t_hist, vx_all[:i+1], 1)  # vx = a*t + b
            coeff_vy = np.polyfit(t_hist, vy_all[:i+1], 1)
            coeff_vz = np.polyfit(t_hist, vz_all[:i+1], 1)

            t_i = float(t_all[i])

            # Smoothed current position from position-fit at t_i
            x_i = coeff_x[0] * t_i + coeff_x[1]
            y_i = coeff_y[0] * t_i + coeff_y[1]
            z_i = coeff_z[0] * (t_i**2) + coeff_z[1] * t_i + coeff_z[2]

            # Velocity from derivative of position-fit
            vx_pos = coeff_x[0]
            vy_pos = coeff_y[0]
            vz_pos = 2.0 * coeff_z[0] * t_i + coeff_z[1]

            # Velocity from velocity-fit at t_i
            vx_vel = coeff_vx[1]
            vy_vel = coeff_vy[1]
            vz_vel = coeff_vz[0] * t_i + coeff_vz[1]

            # Blend them (vel_weight=0.5 means equal weight)
            vx_i = (1.0 - vel_weight) * vx_pos + vel_weight * vx_vel
            vy_i = (1.0 - vel_weight) * vy_pos + vel_weight * vy_vel
            vz_i = (1.0 - vel_weight) * vz_pos + vel_weight * vz_vel

            # Find landing time from fitted z(t) = 0
            # Solve: z_i + vz_i*t - 0.5*g*t^2 = 0
            a = -0.5 * g
            b = vz_i
            c = z_i

            disc = b**2 - 4.0 * a * c
            
            t_land_rel = (b + np.sqrt(disc)) / g
            
            # Landing point from current state
            x_land = x_i + vx_i * t_land_rel
            y_land = y_i + vy_i * t_land_rel

            x_corr[i] = x_land
            y_corr[i] = y_land

        mask_idx = throw.index
        corrected_mask = np.isfinite(x_corr) & np.isfinite(y_corr)

        dataset.loc[mask_idx[corrected_mask], 'x_land_corr'] = x_corr[corrected_mask]
        dataset.loc[mask_idx[corrected_mask], 'y_land_corr'] = y_corr[corrected_mask]

    return dataset

df = calc_landing_point_corr_from(df)


#Export Data
df.to_csv(OUTPUT_DATA_CSV, index=False)
print(f"\nFile '{OUTPUT_DATA_CSV}' exported successfully.")

#Plotting

plotter.plot_impact_log_data(df, impact_log, output_folder=OUTPUT_FOLDER, show=True )
#
#for throw_id in range(1, 31):
#    plotter.plot_landing_points_for_throw(df, impact_log, throw_id, output_folder=OUTPUT_FOLDER_PER_THROW_IMPACT_FIG, show=True )
#    plotter.plot_landing_error_over_time_for_throw(df, impact_log, throw_id,  output_folder=OUTPUT_FOLDER_PER_THROW_ERROR_FIG, show=True )
#    #plotter.plot_landing_points_for_throw_with_RMS(df, impact_log, throw_id, output_folder=OUTPUT_FOLDER_PER_THROW_ERROR_RMS_FIG, show=True )


plotter.plot_landing_error_all_throws(df, impact_log,output_folder=OUTPUT_FOLDER, show=True )
plotter.plot_landing_error_all_throws_with_corr(df, impact_log,output_folder=OUTPUT_FOLDER, show=True )
plotter.plot_systematic_offset(df, impact_log, output_folder=OUTPUT_FOLDER, show=True )

plt.show
