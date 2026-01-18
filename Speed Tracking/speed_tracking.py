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
plt.figure(1, figsize=(12, 8))
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

# %% 5c) Error Quantification
# Calculate Absolute Error for each axis at every frame
for axis in ['x', 'y', 'z']:
    df[f'v_err_{axis}'] = np.abs(df[f'vel_{axis}_ground_truth'] - df[f'vel_{axis}_measured'])

# Compute MAE, RMSE, Max Error for all throws and axes
print("\n" + "="*80)
print(f"{'VELOCITY ERROR ANALYSIS (Finite Differences)':^80}")
print("="*80)

baseline_stats = []
axes = ['x', 'y', 'z']
for axis in axes:
    errors = df[f'v_err_{axis}']
    mae = errors.mean()
    rmse = np.sqrt((errors**2).mean())
    max_err = errors.max()
    baseline_stats.append({'Axis': axis.upper(), 'MAE': mae, 'RMSE': rmse, 'Max Error': max_err})
    print(f"Axis {axis.upper()}: MAE = {mae:.4f} m/s, RMSE = {rmse:.4f} m/s, Max Error = {max_err:.4f} m/s")
print("="*80)

# Print the Absolute Error over normalized time for all throws
plt.figure(2, figsize=(15, 10))
plt.suptitle('Absolute Velocity Error over Normalized Time', fontsize=16) 
for i, axis in enumerate(axes):
    plt.subplot(3, 1, i+1)
    for tid, group in df.groupby('throw_id'):
        # Normalize time from 0 to 1
        t_norm = (group['t'] - group['t'].min()) / (group['t'].max() - group['t'].min())

        plt.scatter(t_norm, group[f'v_err_{axis}'], color='gray', alpha=0.5, s=10)
    # Plot average trend
    plt.title(f"Axis {axis.upper()}")
    plt.ylabel('Absolute Velocity Error (m/s)')
    plt.grid(True, alpha=0.3)
plt.xlabel('Normalized Time')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])



# %% 5d) Option 1: Weighted Moving Average

def get_velocity_wma(df, weights):
    """
    Calculates velocity using Weighted Moving Average on position data
    followed by finite differences.
    """
    axes = ['x', 'y', 'z']
    k = len(weights)
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # Normalize

    for axis in axes:
        df[f'vel_{axis}_wma'] = 0.0

    for throw_id, group in df.groupby('throw_id'):
        t = group['t'].values

        for axis in axes:
            pos = group[f'{axis}_raw'].interpolate().bfill().ffill().to_numpy()
            pos_wma = np.zeros_like(pos)

            # Causal weighted moving average
            for i in range(len(pos)):
                start = max(0, i - k + 1)
                w = weights[-(i - start + 1):]
                w = w / np.sum(w)
                pos_wma[i] = np.sum(pos[start:i+1] * w[::-1])

            # Finite differences
            v = np.zeros_like(pos)
            dt = np.diff(t)
            dt[dt == 0] = 1e-6
            v[1:] = np.diff(pos_wma) / dt
            v[0] = v[1]

            df.loc[group.index, f'vel_{axis}_wma'] = v

    return df

weights = [0.5, 0.3, 0.2]  # recent frames weighted more
df = get_velocity_wma(df, weights)

# Plot WMA velocities for throw_id = i and superimpose with ground truth
plt.figure(3, figsize=(12, 8))
plt.suptitle(f"WMA Velocity Estimation - Throw {throw_id}", fontsize=14)
for i, axis in enumerate(axes):
    plt.subplot(3, 1, i+1)
    d_ex = df[df['throw_id'] == throw_id]
    plt.plot(d_ex['t'], d_ex[f'vel_{axis}_ground_truth'], 'r--', label='Ground Truth', linewidth=2)
    plt.plot(d_ex['t'], d_ex[f'vel_{axis}_wma'], 'b-', label='WMA Estimated', linewidth=1.5)
    plt.ylabel(f'Vel {axis.upper()} (m/s)')
    plt.legend(loc='upper right')
plt.tight_layout()

# Error Quantification for WMA
for axis in axes:
    df[f'v_err_{axis}_wma'] = np.abs(df[f'vel_{axis}_ground_truth'] - df[f'vel_{axis}_wma'])

# Compute MAE, RMSE, Max Error for all throws and axes
print("\n" + "="*80)
print(f"{'VELOCITY ERROR ANALYSIS (WMA)':^80}")
print("="*80)
wma_stats = []
for axis in axes:
    errors = df[f'v_err_{axis}_wma']
    wma_stats.append({
        'Axis': axis.upper(), 
        'MAE': errors.mean(),
        'RMSE': np.sqrt((errors**2).mean()),
        'Max Error': errors.max()
    })
    print(f"Axis {axis.upper()}: MAE = {errors.mean():.4f} m/s, RMSE = {np.sqrt((errors**2).mean()):.4f} m/s, Max Error = {errors.max():.4f} m/s")
print("="*80)

# Print the Absolute Error over normalized time for all throws
plt.figure(4, figsize=(15, 10))
plt.suptitle('Absolute Velocity Error (WMA) over Normalized Time', fontsize=16) 
for i, axis in enumerate(axes):
    plt.subplot(3, 1, i+1)
    for tid, group in df.groupby('throw_id'):
        # Normalize time from 0 to 1
        t_norm = (group['t'] - group['t'].min()) / (group['t'].max() - group['t'].min())

        plt.scatter(t_norm, group[f'v_err_{axis}_wma'], color='RoyalBlue', alpha=0.5, s=10)
    # Plot average trend
    plt.title(f"Axis {axis.upper()}")
    plt.ylabel('Absolute Velocity Error (m/s)')
    plt.grid(True, alpha=0.3)
plt.xlabel('Normalized Time')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])



# %% 5d) Option 2: Causal Sliding Window Polynomial Fit

def get_velocity_sliding_poly(df, window_size=7):
    """
    Calculates velocity by fitting a 2nd order polynomial to a 
    sliding window of the LAST N samples.
    Velocity is the analytical derivative of the fit at the current time.
    """
    axes = ['x', 'y', 'z']
    for axis in axes:
        df[f'vel_{axis}_slpoly'] = 0.0

    for tid, group in df.groupby('throw_id'):
        t_vals = group['t'].values
        idx = group.index
        
        for axis in axes:
            pos_raw = group[f'{axis}_raw'].interpolate().bfill().ffill().values
            v_est = np.zeros_like(pos_raw)

            # Control for initial frames with insufficient data (minimum 3 points for parabola)
            for i in range(len(pos_raw)):
                if i < 2: 
                    v_est[i] = 0 if i == 0 else (pos_raw[i] - pos_raw[i-1]) / (t_vals[i] - t_vals[i-1])
                    continue
                
                # Define the sliding window (up to i)
                start_win = max(0, i - window_size + 1)
                t_win = t_vals[start_win:i+1]
                p_win = pos_raw[start_win:i+1]
                
                # Fit local parabola: p(t) = a*t^2 + b*t + c
                # This is the "Least Squares" solution from Bishop Chapter 3.1
                coeffs = np.polyfit(t_win, p_win, 2)
                
                # Compute velocity as derivative: v(t) = 2*a*t + b
                v_est[i] = 2 * coeffs[0] * t_vals[i] + coeffs[1]
                
            df.loc[idx, f'vel_{axis}_slpoly'] = v_est
    return df

# Apply the sliding window polyfit
df = get_velocity_sliding_poly(df, window_size=7)

# Plot velocities by sliding polyfit for throw_id = i and superimpose with ground truth
plt.figure(5, figsize=(12, 8))
plt.suptitle(f"Sliding Polyfit Velocity Estimation - Throw {throw_id}", fontsize=14)
for i, axis in enumerate(axes):
    plt.subplot(3, 1, i+1)
    d_ex = df[df['throw_id'] == throw_id]
    plt.plot(d_ex['t'], d_ex[f'vel_{axis}_ground_truth'], 'r--', label='Ground Truth', linewidth=2)
    plt.plot(d_ex['t'], d_ex[f'vel_{axis}_slpoly'], 'b-', label='Sliding Polyfit Estimated', linewidth=1.5)
    plt.ylabel(f'Vel {axis.upper()} (m/s)')
    plt.legend(loc='upper right')
plt.tight_layout()

# Error Quantification for Sliding Polyfit
for axis in axes:
    df[f'v_err_{axis}_slpoly'] = np.abs(df[f'vel_{axis}_ground_truth'] - df[f'vel_{axis}_slpoly'])

# Compute MAE, RMSE, Max Error for all throws and axes
print("\n" + "="*80)
print(f"{'VELOCITY ERROR ANALYSIS (Sliding Polyfit)':^80}")
print("="*80)
slpoly_stats = []
for axis in axes:
    errors = df[f'v_err_{axis}_slpoly']
    slpoly_stats.append({
        'Axis': axis.upper(), 
        'MAE': errors.mean(),
        'RMSE': np.sqrt((errors**2).mean()),
        'Max Error': errors.max()
    })
    print(f"Axis {axis.upper()}: MAE = {errors.mean():.4f} m/s, RMSE = {np.sqrt((errors**2).mean()):.4f} m/s, Max Error = {errors.max():.4f} m/s")
print("="*80)

# Print the Absolute Error over normalized time for all throws
plt.figure(6, figsize=(15, 10))
plt.suptitle('Absolute Velocity Error (Sliding Polyfit) over Normalized Time', fontsize=16)
for i, axis in enumerate(axes):
    plt.subplot(3, 1, i+1)
    for tid, group in df.groupby('throw_id'):
        # Normalize time from 0 to 1
        t_norm = (group['t'] - group['t'].min()) / (group['t'].max() - group['t'].min())

        plt.scatter(t_norm, group[f'v_err_{axis}_slpoly'], color='ForestGreen', alpha=0.5, s=10)
    plt.title(f"Axis {axis.upper()}")
    plt.ylabel('Absolute Velocity Error (m/s)')
    plt.grid(True, alpha=0.3)
plt.xlabel('Normalized Time')
plt.tight_layout()


# %% Final comparison of all methods
# Plot all methods for throw_id = i
plt.figure(7, figsize=(12, 8))
plt.suptitle(f"Velocity Estimation Comparison - Throw {throw_id}", fontsize=14)
for i, axis in enumerate(axes):
    plt.subplot(3, 1, i+1)
    d_ex = df[df['throw_id'] == throw_id]
    plt.plot(d_ex['t'], d_ex[f'vel_{axis}_ground_truth'], 'r--', label='Ground Truth', linewidth=2)
    plt.plot(d_ex['t'], d_ex[f'vel_{axis}_measured'], 'b-', label='Finite Differences', linewidth=1.5)
    plt.plot(d_ex['t'], d_ex[f'vel_{axis}_wma'], 'g-', label='WMA Estimated', linewidth=1.5)
    plt.plot(d_ex['t'], d_ex[f'vel_{axis}_slpoly'], 'm-', label='Sliding Polyfit Estimated', linewidth=1.5)
    plt.ylabel(f'Vel {axis.upper()} (m/s)')
    plt.legend(loc='upper right')
plt.tight_layout()

# Plot histograms of performance metrics for all methods
methods = ['measured', 'wma', 'slpoly']
metrics_to_plot = ['MAE', 'RMSE', 'Max Error']
axes_names = ['X', 'Y', 'Z']

plt.figure(8, figsize=(18, 12))
plt.suptitle('Performance Metrics Comparison Across Methods', fontsize=16)

for m_idx, metric in enumerate(metrics_to_plot):
    plt.subplot(3, 1, m_idx+1)

    # Prepare data for plotting
    x = np.arange(len(axes_names))
    width = 0.2

    # Extract values from saved stats lists
    val_base = [baseline_stats[i][metric] for i in range(3)]
    val_wma = [wma_stats[i][metric] for i in range(3)]
    val_slpoly = [slpoly_stats[i][metric] for i in range(3)]

    plt.bar(x - width, val_base, width, label='Finite Differences', color='gray', alpha=0.7)
    plt.bar(x, val_wma, width, label='WMA', color='RoyalBlue', alpha=0.7)
    plt.bar(x + width, val_slpoly, width, label='Sliding Polyfit', color='ForestGreen', alpha=0.7)

    plt.title(f'{metric} Comparison')
    plt.xticks(x, axes_names)
    plt.ylabel(f'{metric} (m/s)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Print percentage improvements
print("\n" + "="*80)
print(f"{'PERCENTAGE IMPROVEMENT OVER BASELINE':^80}")
print("="*80)
for i, axis in enumerate(axes):
    base_mae = baseline_stats[i]['MAE']
    wma_mae = wma_stats[i]['MAE']
    slpoly_mae = slpoly_stats[i]['MAE']

    imp_wma = ((base_mae - wma_mae) / base_mae) * 100
    imp_slpoly = ((base_mae - slpoly_mae) / base_mae) * 100

    print(f"Axis {axis.upper()}: WMA Improvement = {imp_wma:.2f}%, Sliding Polyfit Improvement = {imp_slpoly:.2f}%")

# %% Export Final Dataset (Optimized with Sliding Polyfit)

# Define columns to keep from original tracking
export_cols = ['throw_id', 't', 'x_raw', 'y_raw', 'z_raw', 'coeff_x_1', 'coeff_x_0', 'coeff_y_1', 'coeff_y_0',
               'coeff_z_2', 'coeff_z_1', 'coeff_z_0']

# Map Sliding Polyfit to standard names for compatibility
rename_map = {}
for ax in ['x', 'y', 'z']:
    export_cols += [f'vel_{ax}_ground_truth', f'vel_{ax}_slpoly', f'v_err_{ax}_slpoly']
    rename_map[f'vel_{ax}_slpoly'] = f'vel_{ax}_measured'
    rename_map[f'v_err_{ax}_slpoly'] = f'v_err_{ax}'
df_final = df[export_cols].rename(columns=rename_map)

# Export to CSV
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
df_final.to_csv(OUTPUT_CSV, index=False)

print(f"Dataset exported to: {OUTPUT_CSV}")
print(f"Final columns: {df_final.columns.tolist()}")

plt.show()
