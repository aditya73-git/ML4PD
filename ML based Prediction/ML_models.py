import pandas as pd
import numpy as np
from sklearn.base import clone

# Import custom plot modules
from Plots import plot_throw_predictions as plot1
from Plots import plot_throw_trajectory as plot2

# ML Libraries
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
vel = pd.read_csv("beer_pong_velocity_output.csv")
impact = pd.read_csv("impact_log.csv")

# Align throws (Impact IDs 21-50 map to Velocity IDs 1-30)
impact["throw_id"] = impact["ID"] - 20
impact["target_x_m"] = impact["X_cm"] / 100.0
impact["target_y_m"] = impact["Y_cm"] / 100.0

# --------------------------------------------------
# 2. Advanced Feature Engineering
# --------------------------------------------------
vel["vel_mag"] = np.sqrt(vel["vel_x_measured"]**2 + vel["vel_y_measured"]**2 + vel["vel_z_measured"]**2)

agg_rules = {
    "t": [lambda x: x.max() - x.min()], # Duration
    "coeff_x_1": "first", "coeff_x_0": "first",
    "coeff_y_1": "first", "coeff_y_0": "first",
    "coeff_z_2": "first", "coeff_z_1": "first", "coeff_z_0": "first",
    "x_raw": ["first", "last", "std"],
    "y_raw": ["first", "last", "std"],
    "z_raw": ["first", "last", "std"],
    "vel_x_measured": ["mean", "last", "std"],
    "vel_y_measured": ["mean", "last", "std"],
    "vel_z_measured": ["mean", "last", "std"],
    "vel_mag": ["mean", "max", "last"]
}

agg = vel.groupby("throw_id").agg(agg_rules)
agg.columns = [f"{c[0]}_{c[1]}" if isinstance(c, tuple) else c for c in agg.columns]
agg = agg.rename(columns={"t_<lambda_0>": "duration"})
agg = agg.reset_index()

# Merge
df = agg.merge(impact[["throw_id", "target_x_m", "target_y_m"]], on="throw_id", how="inner")

X = df.drop(columns=["throw_id", "target_x_m", "target_y_m"])
y_x = df["target_x_m"]
y_y = df["target_y_m"]

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# --------------------------------------------------
# 3. Model Definition
# --------------------------------------------------
feature_selector = SelectKBest(score_func=f_regression, k=10)

kernel = 1.0 * Matern(length_scale=1.0, nu=2.5) + \
         WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-10, 1e-1))

# Define Base Pipelines
gpr_pipeline_base = Pipeline([
    ("scaler", StandardScaler()),
    ("select", feature_selector),
    ("gpr", GaussianProcessRegressor(
        kernel=kernel, 
        n_restarts_optimizer=5, 
        normalize_y=True, 
        random_state=42
    ))
])

gb_pipeline_base = Pipeline([
    ("select", feature_selector),
    ("gb", GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.01, max_depth=2, subsample=0.9, random_state=42
    ))
])

# Clone for X and Y specific usage
gpr_pipeline_x = clone(gpr_pipeline_base)
gpr_pipeline_y = clone(gpr_pipeline_base)
gb_pipeline_x = clone(gb_pipeline_base)
gb_pipeline_y = clone(gb_pipeline_base)

# --------------------------------------------------
# 4. Rigorous Evaluation (Leave-One-Out Cross-Validation)
# --------------------------------------------------
loo = LeaveOneOut()

# Containers for metrics
errors_gpr = []
errors_gb = []

# Containers for Plotting (Initialize these!)
px_gpr_all, py_gpr_all = [], []
px_gb_all, py_gb_all = [], []
std_x_all, std_y_all = [], []

print("\nRunning Leave-One-Out Cross-Validation (30 folds)...")

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    
    # Get targets for this split
    yx_train, yx_test = y_x.iloc[train_idx], y_x.iloc[test_idx]
    yy_train, yy_test = y_y.iloc[train_idx], y_y.iloc[test_idx]
    
    # --- Gaussian Process (Fit & Predict) ---
    gpr_pipeline_x.fit(X_train, yx_train)
    px, std_x = gpr_pipeline_x.predict(X_test, return_std=True)
    
    gpr_pipeline_y.fit(X_train, yy_train)
    py, std_y = gpr_pipeline_y.predict(X_test, return_std=True)
    
    # Store Predictions
    px_gpr_all.append(px[0])
    py_gpr_all.append(py[0])
    std_x_all.append(std_x[0])
    std_y_all.append(std_y[0])
    
    # --- Gradient Boosting (Fit & Predict) ---
    gb_pipeline_x.fit(X_train, yx_train)
    px_gb = gb_pipeline_x.predict(X_test)
    
    gb_pipeline_y.fit(X_train, yy_train)
    py_gb = gb_pipeline_y.predict(X_test)
    
    px_gb_all.append(px_gb[0])
    py_gb_all.append(py_gb[0])
    
    # --- Calculate Errors ---
    sq_err_gpr = ((px[0] - yx_test.values[0])*100)**2 + ((py[0] - yy_test.values[0])*100)**2
    errors_gpr.append(sq_err_gpr)

    sq_err_gb = ((px_gb[0] - yx_test.values[0])*100)**2 + ((py_gb[0] - yy_test.values[0])*100)**2
    errors_gb.append(sq_err_gb)

# Calculate RMSE
rmse_gpr = np.sqrt(np.mean(errors_gpr))
rmse_gb  = np.sqrt(np.mean(errors_gb))

print("\nFINAL ROBUST RESULTS (LOOCV)")
print("--------------------------------")
print(f"Gaussian Process RMSE : {rmse_gpr:.2f} cm")
print(f"Gradient Boosting RMSE: {rmse_gb:.2f} cm")

# --------------------------------------------------
# 5. Plotting
# --------------------------------------------------

# Plot 1: Overall Accuracy (Predicted vs True)
plot1(
    y_x.values, y_y.values,
    np.array(px_gpr_all), np.array(py_gpr_all),
    np.array(px_gb_all), np.array(py_gb_all),
    std_x=np.array(std_x_all),  # Pass the collected uncertainty
    std_y=np.array(std_y_all)
)

# Plot 2: Trajectory Visualization (for the very last test throw)
last_throw_id = df.iloc[test_idx[0]]["throw_id"]
print(f"\nVisualizing Trajectory for Throw ID: {int(last_throw_id)}")
plot2(last_throw_id, vel)