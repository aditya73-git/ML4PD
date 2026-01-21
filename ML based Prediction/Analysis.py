import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.base import clone

# 1. LOAD & ALIGN DATA
vel = pd.read_csv("beer_pong_velocity_output.csv")
impact = pd.read_csv("impact_log.csv")

# Align IDs: Impact ID 21 corresponds to Throw ID 1
impact["throw_id"] = impact["ID"] - 20
vel["vel_mag"] = np.sqrt(vel["vel_x_measured"]**2 + vel["vel_y_measured"]**2 + vel["vel_z_measured"]**2)

# 2. FEATURE EXTRACTION
def get_features(df_partial):
    res = {}
    res["duration"] = df_partial["t"].max() - df_partial["t"].min()
    # Position
    for ax in ["x", "y", "z"]:
        col = f"{ax}_raw"
        res[f"{col}_first"] = df_partial[col].iloc[0]
        res[f"{col}_last"] = df_partial[col].iloc[-1]
        res[f"{col}_std"] = df_partial[col].std() if len(df_partial) > 1 else 0.0
    # Velocity
    for ax in ["x", "y", "z"]:
        col = f"vel_{ax}_measured"
        res[f"{col}_mean"] = df_partial[col].mean()
        res[f"{col}_last"] = df_partial[col].iloc[-1]
        res[f"{col}_std"] = df_partial[col].std() if len(df_partial) > 1 else 0.0
    # Magnitude
    res["vel_mag_mean"] = df_partial["vel_mag"].mean()
    res["vel_mag_max"] = df_partial["vel_mag"].max()
    res["vel_mag_last"] = df_partial["vel_mag"].iloc[-1]
    return pd.Series(res)

data_rows = []
targets_x = []
targets_y = []

# Merge common IDs
common_ids = sorted(list(set(vel["throw_id"]).intersection(set(impact["throw_id"]))))

for tid in common_ids:
    full_throw = vel[vel["throw_id"] == tid].sort_values("t")
    tgt = impact[impact["throw_id"] == tid]
    
    if len(full_throw) > 0 and len(tgt) > 0:
        data_rows.append(get_features(full_throw))
        targets_x.append(tgt["X_cm"].values[0] / 100.0)
        targets_y.append(tgt["Y_cm"].values[0] / 100.0)

X = pd.DataFrame(data_rows)
feature_names = X.columns
y_x = np.array(targets_x)
y_y = np.array(targets_y)

# 3. DEFINE MODEL (Your Exact Model)
# Note: Scalar length_scale=1.0 means Isotropic (no ARD possible in kernel)
kernel = 1.0 * Matern(length_scale=1.0, nu=2.5) + \
         WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-9, 1e-1))

gpr_base = Pipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(f_regression, k=15)),
    ("gpr", GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=42))
])

# Clone for X and Y
gpr_x_model = clone(gpr_base)
gpr_y_model = clone(gpr_base)

# 4. FIT MODELS
gpr_x_model.fit(X, y_x)
gpr_y_model.fit(X, y_y)

# 5. ANALYSIS FUNCTION (Robust to Isotropic Kernels)
def analyze_model(model, X_input, axis_name, feature_names):
    # Extract Steps
    gpr_step = model.named_steps["gpr"]
    select_step = model.named_steps["select"]
    scaler_step = model.named_steps["scaler"]
    
    # --- PLOT 1: HEATMAP (Valid for all kernels) ---
    # Transform X to see what GPR sees
    X_transformed = select_step.transform(scaler_step.transform(X_input))
    
    K_matrix = gpr_step.kernel_(X_transformed)
    plt.figure(figsize=(10, 8))
    sns.heatmap(K_matrix, cmap="viridis", square=True, xticklabels=False, yticklabels=False)
    plt.title(f"{axis_name}-Axis Model: Similarity Matrix", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"heatmap_{axis_name}.png")
    plt.show()
    
    # --- PLOT 2: FEATURE IMPORTANCE ---
    # Since kernel is Isotropic (scalar length_scale), we cannot use ARD.
    # We use SelectKBest scores instead, which decided which features to keep.
    
    # Get scores for ALL features (SelectKBest calculates them for everything)
    scores = select_step.scores_
    
    # Filter to only the selected ones (k=15)
    mask = select_step.get_support()
    selected_scores = scores[mask]
    selected_names = feature_names[mask]
    
    # Sort
    indices = np.argsort(selected_scores)[::-1]
    sorted_names = selected_names[indices]
    sorted_scores = selected_scores[indices]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_scores)), sorted_scores, color='#34495e', edgecolor='black')
    plt.xticks(range(len(sorted_scores)), sorted_names, rotation=45, ha='right', fontsize=9)
    plt.title(f"{axis_name}-Axis Model: Feature Importance (SelectKBest F-Scores)", fontsize=16)
    plt.ylabel("F-Score (Relevance)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    # Add text explaining why this isn't ARD
    plt.figtext(0.5, 0.01, "Note: Model uses Isotropic Kernel (Scalar Length Scale). Importance based on Feature Selection step.", 
                ha="center", fontsize=9, style='italic')
    plt.savefig(f"feature_importance_{axis_name}.png")
    plt.show()

# Run Analysis
analyze_model(gpr_x_model, X, "X", feature_names)
analyze_model(gpr_y_model, X, "Y", feature_names)