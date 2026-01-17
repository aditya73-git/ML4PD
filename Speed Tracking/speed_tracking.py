#%% Initial Setup
from sys import prefix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define file path for dataset
INPUT_CSV = "Location Tracking/beer_pong_trajectories.csv"
OUTPUT_FOLDER = "Speed Tracking"
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "beer_pong_velocity_output.csv")

# Load dataset
df = pd.read_csv(INPUT_CSV)


#%% 5a) Determine velocities based in fitted coefficients

def get_ground_truth_velocity(df):
    """
    Calculates the analytical velocity (Ground Truth) for Task 5a.
    Logic: Applies the power rule d/dt (c_n * t^n) = n * c_n * t^(n-1)
    for all polynomial coefficients provided in the dataset.
    """

    # Define axes
    axes = ['x', 'y', 'z']

    for axis in axes:
        # Define the prefix
        prefix = f'coeff_{axis}_'

        # Identify coefficient columns for current axis
        coeff_cols = [col for col in df.columns if col.startswith(prefix)]

        # Initialise velocity with zeros
        vel_ground_truth = np.zeros(len(df))
        t = df['t'].to_numpy()

        for col in coeff_cols:
            # Extract power from column name
            power = int(col.replace(prefix, ''))

            # Apply power rule: d/dt (c_n * t^n) = n * c_n * t^(n-1)
            if power > 0:
                c_n = df[col].to_numpy()
            
            vel_component = power * c_n * t**(power - 1)

            # Accumulate velocity components
            vel_ground_truth += vel_component

            # Add velocity column to DataFrame
        df[f'vel_{axis}_ground_truth'] = vel_ground_truth

    return df

    # Apply the calculation to our dataset
df = get_ground_truth_velocity(df)

# Show the results for verification
print("Analytical Ground Truth Velocities calculated (Task 5a):")
print(df[['throw_id', 't', 'vel_x_ground_truth', 'vel_y_ground_truth', 'vel_z_ground_truth']].head())

# %% 5b) Determine current x, y, z velocities based on raw tracking data

def get_current_velocity(df):
    """
    Calculates the measured velocity (Task 5b) using finite difference method.
    Logic: v(t) = (x(t+dt) - x(t)) / dt
    """

    # Define axes
    axes = ['x', 'y', 'z']

    # Preallocate velocity columns in df
    for axis in axes:
        df[f'vel_{axis}_measured'] = 0.0

    # Group by throw_id to handle each throw separately
    for throw_id, group in df.groupby('throw_id'):
        t = group['t'].values

        for axis in axes:
            # Interpolate raw position data for filling missing values
            pos = group[f'{axis}_raw'].interpolate(method='linear').bfill().ffill().to_numpy()
            
            # Calculate finite difference velocities
            delta_pos = np.diff(pos)
            delta_t = np.diff(t)
            delta_t[delta_t == 0] = 1e-6  # Prevent division by zero

            v_measured = np.zeros(len(group))
            v_measured[1:] = delta_pos / delta_t
            v_measured[0] = v_measured[1]  # Initial frame velocity

            # Update df
            df.loc[group.index, f'vel_{axis}_measured'] = v_measured

    return df

# Apply the calculation to our dataset
df = get_current_velocity(df)

# Show the results for verification
print("Measured Velocities calculated (Task 5b):")
print(df[['throw_id', 't', 'vel_x_measured', 'vel_y_measured', 'vel_z_measured']].head())

# Plot measured velocities for throw_id = i and superimpose with ground truth
throw_id = 26
data_throw = df[df['throw_id'] == throw_id]
t = data_throw['t'].values
vel_x_gt = data_throw['vel_x_ground_truth'].values
vel_y_gt = data_throw['vel_y_ground_truth'].values
vel_z_gt = data_throw['vel_z_ground_truth'].values
vel_x_meas = data_throw['vel_x_measured'].values
vel_y_meas = data_throw['vel_y_measured'].values
vel_z_meas = data_throw['vel_z_measured'].values
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, vel_x_meas, 'b-', label='Measured Velocity X')
plt.plot(t, vel_x_gt, 'r--', label='Ground Truth Velocity X')
plt.xlabel('Time (s)')
plt.ylabel('Velocity X (m/s)')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(t, vel_y_meas, 'b-', label='Measured Velocity Y')
plt.plot(t, vel_y_gt, 'r--', label='Ground Truth Velocity Y')
plt.xlabel('Time (s)')
plt.ylabel('Velocity Y (m/s)')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(t, vel_z_meas, 'b-', label='Measured Velocity Z')
plt.plot(t, vel_z_gt, 'r--', label='Ground Truth Velocity Z')
plt.xlabel('Time (s)')
plt.ylabel('Velocity Z (m/s)')
plt.legend()
plt.tight_layout()
plt.show()

# %% 5c) Error Quantification & Final Export
# Calculate Absolute Error for each axis at every frame
for axis in ['x', 'y', 'z']:
    df[f'v_err_{axis}'] = np.abs(df[f'vel_{axis}_ground_truth'] - df[f'vel_{axis}_measured'])

# Print the Global Mean Absolute Error (MAE) for the report
print("\n--- Global Performance Summary (MAE) ---")
print(df[['v_err_x', 'v_err_y', 'v_err_z']].mean())

# Export the final dataset
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nFile '{OUTPUT_CSV}' exported successfully.")