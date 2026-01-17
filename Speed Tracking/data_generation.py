import numpy as np
import pandas as pd

def generate_position_dataset(n_throws=30, filename="beer_pong_velocity_input.csv"):
    fps = 60 
    g = 9.81
    rows = []

    for throw_id in range(1, n_throws + 1):
        # Randomized duration for each throw
        duration = np.random.uniform(0.8, 1.2)
        t = np.linspace(0, duration, int(fps * duration))
        
        # 1. Physics Trajectory (Ground Truth)
        # X and Y have constant velocity, Z has constant acceleration (gravity)
        vx_true = np.random.uniform(2.1, 2.6)
        vy_true = np.random.uniform(-0.1, 0.1) # Lateral deviation
        vz0_true = np.random.uniform(3.0, 3.5)
        
        x_true = vx_true * t
        y_true = vy_true * t
        z_true = vz0_true * t - 0.5 * g * t**2 + 0.2

        # 2. Raw Data with Noise (Simulating Task 4b)
        noise_level = 0.015
        x_raw = x_true + np.random.normal(0, noise_level, len(t))
        y_raw = y_true + np.random.normal(0, noise_level, len(t))
        z_raw = z_true + np.random.normal(0, noise_level, len(t))

        # --- REAL ERRORS INJECTION ---
        
        # A. Isolated Outliers (Simulating tracking errors)
        outlier_idx = np.random.choice(range(len(t)), 2, replace=False)
        for idx in outlier_idx:
            x_raw[idx] += np.random.uniform(0.3, 0.5) * np.random.choice([-1, 1])
            y_raw[idx] += np.random.uniform(0.3, 0.5) * np.random.choice([-1, 1])
            z_raw[idx] += np.random.uniform(0.3, 0.5) * np.random.choice([-1, 1])

        # B. Isolated Missing Frames (Simulating detection failure)
        nan_indices = np.random.choice(range(len(t)), 3, replace=False)
        x_raw[nan_indices] = y_raw[nan_indices] = z_raw[nan_indices] = np.nan

        # 3. Fitting (Simulating Task 4e) 
        # We use np.polyfit to get coefficients from the noisy raw data
        mask = ~np.isnan(x_raw)
        
        # Polynomial fit: degree 1 for X/Y, degree 2 for Z
        c_x = np.polyfit(t[mask], x_raw[mask], 1) # [slope, intercept]
        c_y = np.polyfit(t[mask], y_raw[mask], 1) # [slope, intercept]
        c_z = np.polyfit(t[mask], z_raw[mask], 2) # [a, b, c] -> at^2 + bt + c

        # Organize data into rows
        for i in range(len(t)):
            rows.append({
                'throw_id': throw_id,
                't': t[i],
                'x_raw': x_raw[i],
                'y_raw': y_raw[i],
                'z_raw': z_raw[i],
                # Generic polynomial coefficient naming: coeff_[axis]_[power]
                'coeff_x_1': c_x[0], 'coeff_x_0': c_x[1],
                'coeff_y_1': c_y[0], 'coeff_y_0': c_y[1],
                'coeff_z_2': c_z[0], 'coeff_z_1': c_z[1], 'coeff_z_0': c_z[2]
            })

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Dataset generated with 3D movement and generic coefficients: {filename}")

if __name__ == "__main__":
    generate_position_dataset()