import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Custom plotting functions for visualizing results and covariance evolution
from Plots import plot_logo_validation_results, plot_realtime_axis_convergence, plot_feature_hist
from Plots import plot_gp_correlation_matrix, animate_gp_covariance_evolution, plot_gp_posterior_grid

# Scikit-Learn tools for building the ML pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut

# --------------------------------------------------
# 1. SETUP & DATA LOADING
# --------------------------------------------------
# Load raw velocity data (trajectory frames) and impact logs (ground truth landing spots)
vel = pd.read_csv(r"ML4PD/Speed Tracking/beer_pong_velocity_output.csv")
impact = pd.read_csv(r"ML4PD/Labelling/impact_log.csv")

# DATA ALIGNMENT:
# The recording hardware started counting at different indices.
# Velocity IDs 1-30 correspond to Impact IDs 21-50. We sync them here.
impact["throw_id"] = impact["ID"] - 20

# Convert target coordinates from cm to meters for numerical stability in the model
impact["target_x_m"] = impact["X_cm"] / 100.0
impact["target_y_m"] = impact["Y_cm"] / 100.0

# PHYSICS CALCULATION:
# Compute total velocity magnitude (speed) combining x, y, z vectors.
# This represents the total kinetic energy of the throw.
vel["vel_mag"] = np.sqrt(vel["vel_x_measured"]**2 + vel["vel_y_measured"]**2 + vel["vel_z_measured"]**2)

# --------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------
def get_features(df_partial):
    """
    Summarizes a sequence of trajectory frames into a single feature vector.
    
    Inputs:
        df_partial: DataFrame containing frames 0 to t (current time)
    Returns:
        pd.Series of kinematic features (Velocity, Position, Duration, Stability)
    """
    res = {}
    
    # 1. TEMPORAL PHYSICS: Time of flight is a key driver for lateral drift (Y-axis error).
    res["duration"] = df_partial["t"].max() - df_partial["t"].min()
    
    # 2. POSITION STATS: Where is the ball now? Where did it start?
    # 'last' = current state, 'first' = release point.
    for ax in ["x", "y", "z"]:
        col = f"{ax}_raw"
        res[f"{col}_first"] = df_partial[col].iloc[0]
        res[f"{col}_last"] = df_partial[col].iloc[-1]
        
        # 'std' captures wobble/instability in the flight path.
        res[f"{col}_std"] = df_partial[col].std() if len(df_partial) > 1 else 0.0

    # 3. VELOCITY STATS: How fast is it moving?
    for ax in ["x", "y", "z"]:
        col = f"vel_{ax}_measured"
        res[f"{col}_mean"] = df_partial[col].mean() # Smooths out sensor noise
        res[f"{col}_last"] = df_partial[col].iloc[-1] # Instantaneous velocity
        res[f"{col}_std"] = df_partial[col].std() if len(df_partial) > 1 else 0.0
        
    # 4. ENERGY STATS: Total energy determines depth (X-axis distance).
    res["vel_mag_mean"] = df_partial["vel_mag"].mean()
    res["vel_mag_max"] = df_partial["vel_mag"].max()
    res["vel_mag_last"] = df_partial["vel_mag"].iloc[-1]
    
    return pd.Series(res)

# --------------------------------------------------
# 3. DATASET PREPARATION
# --------------------------------------------------
data_rows = []
targets_x = []
targets_y = []
groups = []

# Loop through every unique throw to create "Whole Trajectory" training examples.
# Note: For the main training loop, we use the *full* flight path to train the final outcome.
for tid in vel["throw_id"].unique():
    full_throw = vel[vel["throw_id"] == tid].sort_values("t")
    tgt = impact[impact["throw_id"] == tid]
    
    if len(tgt) == 0: continue
    
    feats = get_features(full_throw)
    data_rows.append(feats)
    
    # We train two separate models: one for Depth (X) and one for Lateral (Y)
    targets_x.append(tgt["target_x_m"].values[0])
    targets_y.append(tgt["target_y_m"].values[0])
    groups.append(tid)

X = pd.DataFrame(data_rows)
y_x = np.array(targets_x)
y_y = np.array(targets_y)
groups = np.array(groups)

print(f"Data Prepared: {X.shape[0]} throws.")

# --------------------------------------------------
# 4. MODEL DEFINITIONS
# --------------------------------------------------
# KERNEL DESIGN:
# 1. Matern: General purpose kernel that handles non-linear physical relationships well.
# 2. WhiteKernel: Explicitly models noise. This tells the GPR: "Data isn't perfect, expect variance."
#    Lowered bound to 1e-9 to prevent convergence warnings on clean data.
kernel = 1.0 * Matern(length_scale=1.0, nu=2.5) + \
         WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-9, 1e-1))

# PIPELINE ARCHITECTURE:
# 1. Standardize: Scale inputs so features with large units (like Velocity) don't dominate.
# 2. SelectKBest: AUTOMATIC PHYSICS SELECTION.
#    - X-model will pick Energy/Speed features.
#    - Y-model will pick Duration/Drift features.
# 3. GPR: The actual predictor.
gpr_base = Pipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(f_regression, k=15)),
    ("gpr", GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=42))
])

# BASELINE MODEL:
# Gradient Boosting is used to compare accuracy against GPR.
# GPR is preferred for "Uncertainty," but GB is usually a strong benchmark for "Accuracy."
gb_base = Pipeline([
    ("select", SelectKBest(f_regression, k=15)),
    ("gb", GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42))
])

# Clone base pipelines for X and Y axes (they need to be trained independently)
gpr_x_model, gpr_y_model = clone(gpr_base), clone(gpr_base)
gb_x_model, gb_y_model = clone(gb_base), clone(gb_base)

def extract_gp_covariance(pipeline, X_train):
   # """Helper to extract the internal covariance matrix for visualization."""
    scaler = pipeline.named_steps["scaler"]
    selector = pipeline.named_steps["select"]
    gpr = pipeline.named_steps["gpr"]

    X_t = scaler.transform(X_train)
    X_t = selector.transform(X_t)

    K = gpr.kernel_(X_t)
    return 0.5 * (K + K.T)  # enforce symmetry

# --------------------------------------------------
# 5. LEAVE-ONE-GROUP-OUT CROSS VALIDATION (RMSE)
# --------------------------------------------------
# STRATEGY:
# We must test on a *completely new throw* (Group), not just random frames.
# This prevents "Data Leakage" where the model memorizes a specific throw's path.
logo = LeaveOneGroupOut()

test_throw_ids = []
gpr_preds_x, gpr_preds_y = [], []
gb_preds_x, gb_preds_y = [], []
true_x_all, true_y_all = [], []
std_x_gpr, std_y_gpr = [], []
K_x_matrix, K_y_matrix = [], []

from collections import Counter
x_feat_counter = Counter()
y_feat_counter = Counter()

print("\nRunning Leave-One-Group-Out Validation...")

for train_idx, test_idx in logo.split(X, y_x, groups=groups):
    # Split data: Hold out one specific throw ID for testing
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    yx_train, yx_test = y_x[train_idx], y_x[test_idx]
    yy_train, yy_test = y_y[train_idx], y_y[test_idx]
    
    # Append the throw ID of the test group
    test_throw_ids.append(groups[test_idx[0]])
    
    # Train GPR models
    gpr_x_model.fit(X_train, yx_train)
    gpr_y_model.fit(X_train, yy_train)
    
    # 1. Get Selected Features for X
    mask_x = gpr_x_model.named_steps["select"].get_support()
    feats_x = X.columns[mask_x]
    x_feat_counter.update(feats_x) # Update counts
    
    # 2. Get Selected Features for Y
    mask_y = gpr_y_model.named_steps["select"].get_support()
    feats_y = X.columns[mask_y]
    y_feat_counter.update(feats_y) # Update counts
    
    # (Optional) Save Covariance Matrices for later animation
    Kx = extract_gp_covariance(gpr_x_model, X_train)
    Ky = extract_gp_covariance(gpr_y_model, X_train)
    K_x_matrix.append(Kx)
    K_y_matrix.append(Ky)
    
    # Train Baseline (Gradient Boosting)
    gb_x_model.fit(X_train, yx_train)
    gb_y_model.fit(X_train, yy_train)
    
    # Predict the landing point for the unseen throw
    px_g, std_x = gpr_x_model.predict(X_test, return_std = True)
    py_g, std_y = gpr_y_model.predict(X_test, return_std = True)
    
    px_b = gb_x_model.predict(X_test)
    py_b = gb_y_model.predict(X_test)
    
    # Store predictions to calculate global error later
    gpr_preds_x.append(px_g[0])
    gpr_preds_y.append(py_g[0])
    std_x_gpr.append(std_x[0])
    std_y_gpr.append(std_y[0])
    gb_preds_x.append(px_b[0])
    gb_preds_y.append(py_b[0])
    true_x_all.append(yx_test[0])
    true_y_all.append(yy_test[0])


# --- Metrics Calculation ---
gpr_preds_x = np.array(gpr_preds_x)
gpr_preds_y = np.array(gpr_preds_y)
gb_preds_x = np.array(gb_preds_x)
gb_preds_y = np.array(gb_preds_y)
true_x_all = np.array(true_x_all)
true_y_all = np.array(true_y_all)

# Calculate Euclidean Error (Hypotenuse of X-error and Y-error) in cm
err_gpr = np.sqrt((gpr_preds_x - true_x_all)**2 + (gpr_preds_y - true_y_all)**2) * 100
err_gb = np.sqrt((gb_preds_x - true_x_all)**2 + (gb_preds_y - true_y_all)**2) * 100

# --- NEW: SAVE TO CSV ---
results_df = pd.DataFrame({
    "throw_id": test_throw_ids,
    "true_x": true_x_all*100,
    "true_y": true_y_all*100,
    "gpr_pred_x": gpr_preds_x*100,
    "gpr_pred_y": gpr_preds_y*100,
    "gb_pred_x": gb_preds_x*100,
    "gb_pred_y": gb_preds_y*100
})

results_df.to_csv("model_predictions_comparison.csv", index=False)
print("Results saved to 'model_predictions_comparison.csv'")

plot_feature_hist(x_feat_counter, "X_Axis")
plot_feature_hist(y_feat_counter, "Y_Axis")

rmse_gpr = np.sqrt(np.mean(err_gpr**2))
rmse_gb = np.sqrt(np.mean(err_gb**2))

print("------------------------------------------------")
print(f"TOTAL RMSE (Gaussian Process):  {rmse_gpr:.2f} cm")
print(f"TOTAL RMSE (Gradient Boosting): {rmse_gb:.2f} cm")
print("------------------------------------------------")

plot_logo_validation_results(
    true_x_all, true_y_all, 
    gpr_preds_x, gpr_preds_y, 
    gb_preds_x, gb_preds_y,
    std_x_gpr, std_y_gpr
)

# i = 5
# plot_gp_correlation_matrix(
#     K_x_matrix[i],
#     title=f"GP Correlation Matrix (X-Axis) – Iteration {i+1}"
# )
# plot_gp_posterior_grid(
#     gpr_model=gpr_x_model,
#     X_train=X_train,
#     feature_list=["x_raw_first", "x_raw_last", "vel_x_measured_mean", "y_raw_last", "duration", "vel_mag_mean"],
#     n_points=100
# )


# print("Animating covarinece matirces")
# print("------------------------------------------------")
# animate_gp_covariance_evolution(
#     K_x_matrix,
#     axis_label="X",
#     save_path="gp_covariance_x_evolution.gif"
# )
# K = K_x_matrix[0]
# print("min:", K.min(), "max:", K.max(), "std:", K.std())
# animate_gp_covariance_evolution(
#     K_y_matrix,
#     axis_label="Y",
#     save_path="gp_covariance_y_evolution.gif"
# )


# --------------------------------------------------
# 6. VISUALIZATION (Real-Time Simulation)
# --------------------------------------------------
# We simulate a "Live" scenario using Throw ID 16 or 5.
# The model sees Frame 1... then Frame 2... then Frame 3... updating its prediction.
VISUALIZE_ID = 5
print(f"\nGenerating Absolute Position Plot for Throw {VISUALIZE_ID}...")

# 1. Retrain on everything EXCEPT Throw 16 (Strict blind test)
X_train_vis = X[groups != VISUALIZE_ID]
yx_train_vis = y_x[groups != VISUALIZE_ID]
yy_train_vis = y_y[groups != VISUALIZE_ID]

gpr_x_model.fit(X_train_vis, yx_train_vis)
gpr_y_model.fit(X_train_vis, yy_train_vis)
gb_x_model.fit(X_train_vis, yx_train_vis)
gb_y_model.fit(X_train_vis, yy_train_vis)

# 2. Get the "Ground Truth" for Throw 16
throw_df = vel[vel["throw_id"] == VISUALIZE_ID].sort_values("t")
impact_row = impact[impact["throw_id"] == VISUALIZE_ID]
# Extract the values (using .values[0] to get the scalar number)
true_x_cm = impact_row["X_cm"].values[0]
true_y_cm = impact_row["Y_cm"].values[0]

timeline = []
pred_x_gpr, pred_std_x_gpr, pred_x_gb = [], [], []
pred_y_gpr, pred_std_y_gpr, pred_y_gb = [], [], []
# 3. REAL-TIME LOOP: Feed data incrementally
for i in range(5, len(throw_df)):
    sub = throw_df.iloc[:i] # "What the camera has seen so far"
    t_curr = sub["t"].values[-1]
    ft = pd.DataFrame([get_features(sub)]) # Extract features from partial path
    
    # Predict with Uncertainty (return_std=True)
    # This is the "Cone of Probability" narrowing down over time.
    pxg, sx = gpr_x_model.predict(ft, return_std=True)
    pyg, sy = gpr_y_model.predict(ft, return_std=True)
    
    # Baseline Prediction (Point estimate only)
    pxb = gb_x_model.predict(ft)
    pyb = gb_y_model.predict(ft)
    
    timeline.append(t_curr)
    
    # Convert meters to cm for visualization
    pred_x_gpr.append(pxg[0] * 100)
    pred_std_x_gpr.append(sx[0] * 100)
    pred_x_gb.append(pxb[0] * 100)
    
    pred_y_gpr.append(pyg[0] * 100)
    pred_std_y_gpr.append(sy[0] * 100)
    pred_y_gb.append(pyb[0] * 100)
    

# 4. Plot the "Cone of Convergence"
plot_realtime_axis_convergence(timeline,
    pred_x_gpr, pred_std_x_gpr, pred_x_gb,
    pred_y_gpr, pred_std_y_gpr, pred_y_gb,
    true_x_cm, true_y_cm, VISUALIZE_ID)

    
# import joblib

# print("\n------------------------------------------------")
# print("TRAINING FINAL PRODUCTION MODELS...")
# print("------------------------------------------------")

# # 1. Define the Final Models (clones of your base pipelines)
# final_gpr_x = clone(gpr_base)
# final_gpr_y = clone(gpr_base)
# final_gb_x  = clone(gb_base)
# final_gb_y  = clone(gb_base)

# # 2. Fit on the ENTIRE dataset (X contains all 30 throws)
# final_gpr_x.fit(X, y_x)
# final_gpr_y.fit(X, y_y)
# final_gb_x.fit(X, y_x)
# final_gb_y.fit(X, y_y)

# # 3. Save to disk (.pkl files)
# # These files will appear in your project folder
# joblib.dump(final_gpr_x, 'beer_pong_gpr_x.pkl')
# joblib.dump(final_gpr_y, 'beer_pong_gpr_y.pkl')
# joblib.dump(final_gb_x,  'beer_pong_gb_x.pkl')
# joblib.dump(final_gb_y,  'beer_pong_gb_y.pkl')

# print("Success! Models saved to .pkl files.")