import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation

def plot_logo_validation_results(true_x, true_y, gpr_pred_x, gpr_pred_y, gb_pred_x, gb_pred_y, gpr_std_x=None, gpr_std_y=None):
    """
    Plots the results from Leave-One-Group-Out Validation.
    
    Args:
        true_x (array-like): Ground truth X positions (in meters).
        true_y (array-like): Ground truth Y positions (in meters).
        gpr_pred_x (array-like): Gaussian Process predicted X positions (in meters).
        gpr_pred_y (array-like): Gaussian Process predicted Y positions (in meters).
        gb_pred_x (array-like): Gradient Boosting predicted X positions (in meters).
        gb_pred_y (array-like): Gradient Boosting predicted Y positions (in meters).
        gpr_std_x (array-like, optional): Standard deviation of GPR prediction for X (in meters).
        gpr_std_y (array-like, optional): Standard deviation of GPR prediction for Y (in meters).
    """
    
    plt.figure(figsize=(10, 10))
    
    # Convert meters to cm for cleaner plotting numbers
    tx_cm = np.array(true_x) * 100
    ty_cm = np.array(true_y) * 100
    gpr_px_cm = np.array(gpr_pred_x) * 100
    gpr_py_cm = np.array(gpr_pred_y) * 100
    gb_px_cm = np.array(gb_pred_x) * 100
    gb_py_cm = np.array(gb_pred_y) * 100

    # 1. Plot Actual Landing Spots
    plt.scatter(tx_cm, ty_cm, color='black', label='Actual Landing', s=80, marker='o', zorder=3, edgecolors='white')
    
    # 2. Plot Gaussian Process Predictions
    plt.scatter(gpr_px_cm, gpr_py_cm, color='red', marker='x', label='GP Prediction', s=80, zorder=3, linewidths=2)
    
    # 3. Plot Gradient Boosting Predictions
    plt.scatter(gb_px_cm, gb_py_cm, color='blue', marker='^', label='GB Prediction', s=60, alpha=0.7, zorder=2)
    
    # 4. Draw Lines Connecting Predictions to Truth (Error Lines)
    # We only draw lines for the GPR to avoid clutter
    for i in range(len(true_x)):
        plt.plot([tx_cm[i], gpr_px_cm[i]], [ty_cm[i], gpr_py_cm[i]], 
                 color='red', alpha=0.2, linestyle='-', zorder=1)

    # 5. Draw Uncertainty Ellipses (if standard deviations provided)
    if gpr_std_x is not None and gpr_std_y is not None:
        ax = plt.gca()
        # std is also in meters, so convert to cm (*100)
        std_x_cm = np.array(gpr_std_x) * 100
        std_y_cm = np.array(gpr_std_y) * 100
        
        for i in range(len(gpr_px_cm)):
            # 95% Confidence Interval is +/- 1.96 * sigma
            # Ellipse width/height is the full span (diameter), so 2 * 1.96 * sigma
            width = 1.96 * std_x_cm[i] * 2
            height = 1.96 * std_y_cm[i] * 2
            
            ellipse = Ellipse(xy=(gpr_px_cm[i], gpr_py_cm[i]), 
                              width=width, height=height, 
                              edgecolor='red', fc='red', alpha=0.05, linewidth=0)
            ax.add_patch(ellipse)
            # Add a thin border
            ellipse_border = Ellipse(xy=(gpr_px_cm[i], gpr_py_cm[i]), 
                              width=width, height=height, 
                              edgecolor='red', fc='None', alpha=0.3, linewidth=1, linestyle=':')
            ax.add_patch(ellipse_border)

    # Labels and Grid
    plt.xlabel("Landing X (cm)", fontsize=12)
    plt.ylabel("Landing Y (cm)", fontsize=12)
    plt.title("Leave-One-Group-Out Validation Results", fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.axis('equal') # Crucial to see true distance errors
    
    # Save and Show
    plt.tight_layout()
    plt.savefig("logo_validation_results.png")
    plt.show()


def plot_realtime_axis_convergence(
    timeline,
    pred_x_gpr, std_x_gpr, pred_x_gb,
    pred_y_gpr, std_y_gpr, pred_y_gb,
    true_x_cm, true_y_cm,
    throw_id
):
    """
    Plots the convergence of predictions for both X (Lateral)
    and Y (Distance) axes over the duration of a throw.
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # ---------- X Axis (Lateral) ----------
    ax1.plot(timeline, pred_x_gpr, 'b-', linewidth=2, label='Gaussian Process')
    ax1.fill_between(
        timeline,
        np.array(pred_x_gpr) - 1.96 * np.array(std_x_gpr),
        np.array(pred_x_gpr) + 1.96 * np.array(std_x_gpr),
        alpha=0.1,
        label='95% Confidence'
    )
    ax1.plot(timeline, pred_x_gb, 'r--', linewidth=2, label='Gradient Boosting')

    ax1.axhline(
        true_x_cm,
        linestyle='-',
        linewidth=1,
        label=f'True Impact ({true_x_cm:.1f} cm)'
    )

    ax1.set_ylabel("Lateral Position (cm)", fontsize=12)
    ax1.set_title(
        f"X-Axis Prediction (Final: {pred_x_gpr[-1]:.1f} cm) | Throw ID: {throw_id}",
        fontsize=14
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---------- Y Axis (Distance) ----------
    ax2.plot(timeline, pred_y_gpr, 'b-', linewidth=2, label='Gaussian Process')
    ax2.fill_between(
        timeline,
        np.array(pred_y_gpr) - 1.96 * np.array(std_y_gpr),
        np.array(pred_y_gpr) + 1.96 * np.array(std_y_gpr),
        alpha=0.1,
        label='95% Confidence'
    )
    ax2.plot(timeline, pred_y_gb, 'r--', linewidth=2, label='Gradient Boosting')

    ax2.axhline(
        true_y_cm,
        linestyle='-',
        linewidth=1,
        label=f'True Impact ({true_y_cm:.1f} cm)'
    )

    ax2.set_ylabel("Distance Position (cm)", fontsize=12)
    ax2.set_xlabel("Time in Air (s)", fontsize=12)
    ax2.set_title(
        f"Y-Axis Prediction (Final: {pred_y_gpr[-1]:.1f} cm)",
        fontsize=14
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("absolute_convergence_plot.png", dpi=300)
    plt.show()

def animate_gp_covariance_evolution(
    K_matrix,
    axis_label="X",
    save_path="gp_covariance_evolution.gif",
    interval=600
):
    """
    Animates the evolution of the Gaussian Process covariance matrix
    across Leave-One-Group-Out iterations.

    Args:
        K_matrix (list of np.ndarray): List of covariance matrices (NxN),
                                       one per LOGO iteration.
        axis_label (str): 'X' or 'Y' for labeling.
        save_path (str): Output path for the animation (.gif or .mp4).
        interval (int): Time between frames in milliseconds.
    """

    num_frames = len(K_matrix)
    N = K_matrix[0].shape[0]

    # Global color scale for visual consistency
    vmin = min(np.min(K) for K in K_matrix)
    vmax = max(np.max(K) for K in K_matrix)

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        K_matrix[0],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        origin="lower"
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Covariance", fontsize=11)

    title = ax.set_title(
        f"GP Covariance Matrix Evolution ({axis_label}-Axis)\nIteration 1 / {num_frames}",
        fontsize=13
    )

    ax.set_xlabel("Training Throw Index")
    ax.set_ylabel("Training Throw Index")

    def update(frame):
        K = K_matrix[frame]

        # Enforce symmetry (numerical safety)
        K = 0.5 * (K + K.T)

        im.set_data(K)
        title.set_text(
            f"GP Covariance Matrix Evolution ({axis_label}-Axis)\n"
            f"Iteration {frame + 1} / {num_frames}"
        )
        return [im]

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=interval,
        blit=False
    )

    plt.tight_layout()
    anim.save(save_path, dpi=120)
    plt.show()

def plot_gp_correlation_matrix(K, title="GP Correlation Matrix"):
    """
    Plots the correlation matrix derived from a GP covariance matrix.

    Args:
        K (np.ndarray): Covariance matrix (NxN)
        title (str): Plot title
    """

    # Convert covariance to correlation
    d = np.sqrt(np.diag(K))
    Corr = K / np.outer(d, d)

    # Numerical safety
    Corr = np.clip(Corr, -1.0, 1.0)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        Corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        origin="lower"
    )

    plt.colorbar(im, label="Correlation")
    plt.xlabel("Training Throw Index")
    plt.ylabel("Training Throw Index")
    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_gp_posterior_grid(
    gpr_model,           # your trained Pipeline
    X_train,             # training DataFrame used for fitting
    feature_list=None,   # list of features to plot; if None, take first 6
    n_points=100,        # points along each feature
    show_std=True,       # show 95% confidence interval
    n_cols=3,            # number of columns in grid
    title_prefix="GPR Posterior"
):
    """
    Plots a grid of posterior mean predictions for multiple features,
    holding other features constant at their mean.
    
    Args:
        gpr_model: trained Pipeline with GPR as last step
        X_train: pd.DataFrame used for training
        feature_list: list of feature names to vary; if None, use first 6
        n_points: number of points along x-axis
        show_std: whether to plot the 95% confidence interval
        n_cols: columns in subplot grid
        title_prefix: prefix for subplot titles
    """
    
    if feature_list is None:
        feature_list = X_train.columns[:6].tolist()  # first 6 features by default
    
    n_features = len(feature_list)
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()
    
    for idx, feature_name in enumerate(feature_list):
        if feature_name not in X_train.columns:
            raise ValueError(f"Feature '{feature_name}' not in X_train!")
        
        # Build test matrix
        X_test_full = np.tile(X_train.mean().values, (n_points, 1))
        col_idx = X_train.columns.get_loc(feature_name)
        X_test_full[:, col_idx] = np.linspace(
            X_train[feature_name].min(), 
            X_train[feature_name].max(), 
            n_points
        )
        X_test_full_df = pd.DataFrame(X_test_full, columns=X_train.columns)
        
        # Predict
        y_mean, y_std = gpr_model.predict(X_test_full_df, return_std=True)
        
        # Plot
        ax = axes[idx]
        ax.plot(X_test_full[:, col_idx], y_mean, 'b-', lw=2, label='Posterior Mean')
        if show_std:
            ax.fill_between(
                X_test_full[:, col_idx],
                y_mean - 1.96*y_std,
                y_mean + 1.96*y_std,
                color='blue', alpha=0.2, label='95% CI'
            )
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Prediction")
        ax.set_title(f"{title_prefix} vs {feature_name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Remove empty subplots if any
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

    
# ---------------------------------------------------------
# Usage Example (Add this to your main script after Section 5):
# ---------------------------------------------------------
# plot_logo_validation_results(
#     true_x_all, true_y_all, 
#     gpr_preds_x, gpr_preds_y, 
#     gb_preds_x, gb_preds_y
# )

if __name__ == "__main__":
    print("Module updated for LOOCV and Trajectory Visualization.")