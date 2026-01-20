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

print("Physics based prediction (Task 6a):")
print(df[['throw_id', 't', 'x_land_pred', 'y_land_pred', 't_to_land_pred','t_to_land_grand_truth', 'x_land_grand_truth', 'y_land_grand_truth']].head(30))

# Figure tests for throw 1
plotter.plot_landing_points_for_throw(df, impact_log, throw_id=1, output_folder=None, show=True )
#plotter.plot_landing_points_for_throw(df, impact_log, throw_id=2, output_folder=None, show=True )
#plotter.plot_landing_points_for_throw(df, impact_log, throw_id=3, output_folder=None, show=True )
#plotter.plot_landing_points_for_throw(df, impact_log, throw_id=4, output_folder=None, show=True )

plotter.plot_landing_points_for_throw_with_RMS(df, impact_log, throw_id=4, output_folder=None, show=True )

plotter.plot_landing_error_over_time_for_throw(df, impact_log, throw_id=1,  output_folder=None, show=True )
plotter.plot_impact_log_data(df, impact_log, output_folder=None, show=True )

plotter.plot_landing_error_all_throws(df, impact_log,output_folder=None, show=True )
plotter.plot_systematic_offset(df, impact_log, output_folder=None, show=True )

plt.show()

df.to_csv(OUTPUT_DATA_CSV, index=False)
print(f"\nFile '{OUTPUT_DATA_CSV}' exported successfully.")