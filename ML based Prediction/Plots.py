import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

def plot_throw_predictions(y_x, y_y, px_gpr, py_gpr, px_gb, py_gb, std_x=None, std_y=None):
    """
    Plots throw-level predictions and uncertainties (Corrected for LOOCV).

    Parameters:
    -----------
    y_x, y_y : array-like
        Actual target positions (X, Y) in meters.
    px_gpr, py_gpr : array-like
        GP predicted positions (X, Y) in meters.
    px_gb, py_gb : array-like
        Gradient Boosting predicted positions (X, Y) in meters.
    std_x, std_y : array-like (optional)
        Standard deviations from the GP prediction for each throw.
        *Crucial for LOOCV visualization.*
    """
    
    plt.figure(figsize=(10, 10))
    
    # 1. Actual throws
    plt.scatter(y_x*100, y_y*100, color='black', label='Actual Impact', s=60, zorder=3)
    
    # 2. GP Predictions
    plt.scatter(px_gpr*100, py_gpr*100, color='red', marker='x', label='GP Predicted', s=60, zorder=3)
    
    # 3. GB Predictions
    plt.scatter(px_gb*100, py_gb*100, color='green', marker='^', label='GB Predicted', s=40, alpha=0.6, zorder=2)
    
    # 4. Error lines (GP)
    for i in range(len(y_x)):
        plt.plot([y_x[i]*100, px_gpr[i]*100], [y_y[i]*100, py_gpr[i]*100], 
                 color='gray', alpha=0.3, linestyle='--', zorder=1)

    # 5. Uncertainty Ellipses (95% Confidence)
    if std_x is not None and std_y is not None:
        ax = plt.gca()
        for i in range(len(px_gpr)):
            # Width/Height = 1.96 * std * 2 (diameter) * 100 (cm conversion)
            width = 1.96 * std_x[i] * 2 * 100
            height = 1.96 * std_y[i] * 2 * 100
            
            ellipse = Ellipse(xy=(px_gpr[i]*100, py_gpr[i]*100), 
                              width=width, height=height, 
                              edgecolor='red', fc='None', alpha=0.2, linewidth=1.5)
            ax.add_patch(ellipse)

    plt.xlabel("Target X [cm]")
    plt.ylabel("Target Y [cm]")
    plt.title("Landing Prediction: GP (Red) vs GB (Green)\nEllipses = 95% Confidence Regions")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.show()

def plot_throw_trajectory(throw_id, vel_df, gpr_pipeline_x=None, gpr_pipeline_y=None):
    """
    Plots X(t) and Y(t) for a single throw using a Trajectory-Specific GP Fit.
    
    *Note: Ignores the passed gpr_pipelines because they are for landing prediction,
    not trajectory fitting.*
    """
    
    # 1. Select raw frames for this throw
    df_throw = vel_df[vel_df['throw_id'] == throw_id].sort_values("t")
    
    if len(df_throw) == 0:
        print(f"Error: Throw ID {throw_id} not found.")
        return

    t = df_throw['t'].values.reshape(-1, 1)
    x = df_throw['x_raw'].values
    y = df_throw['y_raw'].values
    
    # 2. Fit Temporary GPs for Visualization (The "Trajectory Model")
    # Using Matern kernel for physics-based smoothing
    kernel = 1.0 * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-3)
    
    gp_x = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
    gp_y = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
    
    gp_x.fit(t, x)
    gp_y.fit(t, y)
    
    # 3. Predict smooth curve for plotting
    t_pred = np.linspace(t.min(), t.max(), 100).reshape(-1, 1)
    px, std_x = gp_x.predict(t_pred, return_std=True)
    py, std_y = gp_y.predict(t_pred, return_std=True)
    
    # 4. Plot
    plt.figure(figsize=(14, 6))
    
    # Plot X vs Time
    plt.subplot(1, 2, 1)
    plt.scatter(t, x, color='red', label='Measured Data', zorder=3)
    plt.plot(t_pred, px, color='blue', label='GP Fit (Mean)', linewidth=2, zorder=2)
    plt.fill_between(t_pred.ravel(), px - 1.96*std_x, px + 1.96*std_x, color='blue', alpha=0.2, label='95% Uncertainty')
    plt.title(f"Throw {int(throw_id)}: Lateral Trajectory (X)", fontsize=14)
    plt.xlabel("Time [s]")
    plt.ylabel("X Position [m]")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Plot Y vs Time
    plt.subplot(1, 2, 2)
    plt.scatter(t, y, color='red', label='Measured Data', zorder=3)
    plt.plot(t_pred, py, color='green', label='GP Fit (Mean)', linewidth=2, zorder=2)
    plt.fill_between(t_pred.ravel(), py - 1.96*std_y, py + 1.96*std_y, color='green', alpha=0.2, label='95% Uncertainty')
    plt.title(f"Throw {int(throw_id)}: Distance Trajectory (Y)", fontsize=14)
    plt.xlabel("Time [s]")
    plt.ylabel("Y Position [m]")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Module updated for LOOCV and Trajectory Visualization.")