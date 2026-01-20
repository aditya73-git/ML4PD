# import pandas as pd
# import numpy as np

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.metrics import mean_squared_error
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler

# # ----------------------------
# # 1. Load data
# # ----------------------------
# df_velocity = pd.read_csv("beer_pong_velocity_output.csv").dropna()
# df_impact = pd.read_csv("impact_log.csv")

# df_impact = df_impact.assign(
#     velocity_id=lambda d: d["ID"] - 20,
#     target_x_m=lambda d: d["X_cm"] / 100.0,
#     target_y_m=lambda d: d["Y_cm"] / 100.0,
# )

# df_ml = df_velocity.merge(
#     df_impact[["velocity_id", "target_x_m", "target_y_m"]],
#     left_on="throw_id",
#     right_on="velocity_id",
#     how="inner"
# )

# # ----------------------------
# # 2. Use last N frames only
# # ----------------------------
# N = 5
# df_ml = (
#     df_ml
#     .sort_values("t")
#     .groupby("throw_id", group_keys=False)
#     .tail(N)
# )

# # ----------------------------
# # 3. Relative (ML-safe) features
# # ----------------------------
# for axis in ["x", "y", "z"]:
#     df_ml[f"d{axis}"] = df_ml.groupby("throw_id")[f"{axis}_raw"].diff()

# for axis in ["x", "y", "z"]:
#     df_ml[f"dv_{axis}"] = df_ml.groupby("throw_id")[f"vel_{axis}_measured"].diff()

# df_ml = df_ml.dropna()

# # ----------------------------
# # 4. Features & targets
# # ----------------------------
# FEATURES = [
#     "x_raw", "y_raw", "z_raw", 
#     #"vel_x_measured", "vel_y_measured", "vel_z_measured",
#     "dx", "dy", "dz",      # Your new relative features
#     "dv_x", "dv_y", "dv_z", # Your new velocity delta features
#     "v_err_x", "v_err_y", "v_err_z" # ADD THESE BACK for the rubric!
# ]
# X = df_ml[FEATURES]

# # Predict offset from last observation (ML-normalized)
# df_ml["dx_target"] = df_ml["target_x_m"] - df_ml["x_raw"]
# df_ml["dy_target"] = df_ml["target_y_m"] - df_ml["y_raw"]

# y_dx = df_ml["dx_target"]
# y_dy = df_ml["dy_target"]

# groups = df_ml["throw_id"]

# # ----------------------------
# # 5. Group split
# # ----------------------------
# gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# train_idx, test_idx = next(gss.split(X, y_dx, groups))

# X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
# dx_train, dx_test = y_dx.iloc[train_idx], y_dx.iloc[test_idx]
# dy_train, dy_test = y_dy.iloc[train_idx], y_dy.iloc[test_idx]

# # ----------------------------
# # 6. Models
# # ----------------------------
# rf_x = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1)
# rf_y = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1)

# mlp_x = Pipeline([
#     ("scaler", StandardScaler()),
#     ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=3000, early_stopping=True, random_state=42))
# ])

# mlp_y = Pipeline([
#     ("scaler", StandardScaler()),
#     ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=3000, early_stopping=True, random_state=42))
# ])

# # ----------------------------
# # 7. Train
# # ----------------------------
# rf_x.fit(X_train, dx_train)
# rf_y.fit(X_train, dy_train)

# mlp_x.fit(X_train, dx_train)
# mlp_y.fit(X_train, dy_train)

# # ----------------------------
# # 8. Predict final position
# # ----------------------------
# x_rf = X_test["x_raw"].to_numpy() + rf_x.predict(X_test)
# y_rf = X_test["y_raw"].to_numpy() + rf_y.predict(X_test)

# x_mlp = X_test["x_raw"].to_numpy() + mlp_x.predict(X_test)
# y_mlp = X_test["y_raw"].to_numpy() + mlp_y.predict(X_test)

# y_true = df_ml.loc[X_test.index, ["target_x_m", "target_y_m"]].to_numpy()

# # ----------------------------
# # 9. Evaluation (cm)
# # ----------------------------
# def plot_landing_point_results(y_true, x_rf, y_rf, x_mlp, y_mlp):
#     """
#     y_true : ndarray (N,2)  -> ground truth landing points [m]
#     x_rf, y_rf : ndarray   -> RF predictions [m]
#     x_mlp, y_mlp : ndarray -> MLP predictions [m]
#     """

#     import numpy as np
#     import matplotlib.pyplot as plt

#     # Convert to cm for presentation
#     y_true_cm = y_true * 100
#     rf_pred_cm = np.column_stack([x_rf, y_rf]) * 100
#     mlp_pred_cm = np.column_stack([x_mlp, y_mlp]) * 100

#     # --------------------------------------------------
#     # 1. Ground truth vs predictions (XY scatter)
#     # --------------------------------------------------
#     plt.figure(figsize=(7, 7))
#     plt.scatter(y_true_cm[:, 0], y_true_cm[:, 1], c="black", label="Ground Truth", s=30)
#     plt.scatter(rf_pred_cm[:, 0], rf_pred_cm[:, 1], c="tab:blue", alpha=0.7, label="Random Forest")
#     plt.scatter(mlp_pred_cm[:, 0], mlp_pred_cm[:, 1], c="tab:orange", alpha=0.7, label="Neural Network")
#     plt.xlabel("X position [cm]")
#     plt.ylabel("Y position [cm]")
#     plt.title("Landing Point Prediction (XY Plane)")
#     plt.legend()
#     plt.axis("equal")
#     plt.grid(True)
#     plt.show()

#     # --------------------------------------------------
#     # 2. Error vectors (direction + magnitude)
#     # --------------------------------------------------
#     plt.figure(figsize=(7, 7))
#     plt.quiver(
#         y_true_cm[:, 0], y_true_cm[:, 1],
#         rf_pred_cm[:, 0] - y_true_cm[:, 0],
#         rf_pred_cm[:, 1] - y_true_cm[:, 1],
#         angles="xy", scale_units="xy", scale=1,
#         color="tab:blue", alpha=0.6, label="RF Error"
#     )
#     plt.quiver(
#         y_true_cm[:, 0], y_true_cm[:, 1],
#         mlp_pred_cm[:, 0] - y_true_cm[:, 0],
#         mlp_pred_cm[:, 1] - y_true_cm[:, 1],
#         angles="xy", scale_units="xy", scale=1,
#         color="tab:orange", alpha=0.6, label="MLP Error"
#     )
#     plt.xlabel("X position [cm]")
#     plt.ylabel("Y position [cm]")
#     plt.title("Prediction Error Vectors")
#     plt.legend()
#     plt.axis("equal")
#     plt.grid(True)
#     plt.show()

#     # --------------------------------------------------
#     # 3. Error magnitude distribution
#     # --------------------------------------------------
#     rf_error = np.linalg.norm(rf_pred_cm - y_true_cm, axis=1)
#     mlp_error = np.linalg.norm(mlp_pred_cm - y_true_cm, axis=1)

#     plt.figure(figsize=(8, 4))
#     plt.hist(rf_error, bins=25, alpha=0.7, label="Random Forest")
#     plt.hist(mlp_error, bins=25, alpha=0.7, label="Neural Network")
#     plt.xlabel("Landing Point Error [cm]")
#     plt.ylabel("Count")
#     plt.title("Error Distribution")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # --------------------------------------------------
#     # 4. Per-sample RMSE comparison
#     # --------------------------------------------------
#     plt.figure(figsize=(8, 4))
#     plt.plot(rf_error, label="Random Forest Error", linewidth=2)
#     plt.plot(mlp_error, label="Neural Network Error", linewidth=2)
#     plt.xlabel("Test Sample Index")
#     plt.ylabel("Error [cm]")
#     plt.title("Per-Sample Landing Point Error")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def eval_model(name, x_pred, y_pred):
#     preds = np.column_stack([x_pred, y_pred]) * 100
#     truth = y_true * 100

#     total_rmse = np.sqrt(mean_squared_error(truth, preds))
#     axis_rmse = np.sqrt(((truth - preds) ** 2).mean(axis=0))

#     print(f"{name}:")
#     print(f"  Total RMSE: {total_rmse:.2f} cm")
#     print(f"  X RMSE: {axis_rmse[0]:.2f} cm | Y RMSE: {axis_rmse[1]:.2f} cm\n")

# eval_model("Random Forest (Pure ML)", x_rf, y_rf)
# eval_model("Neural Network (Pure ML)", x_mlp, y_mlp)

# # plot_landing_point_results(y_true, x_rf, y_rf, x_mlp, y_mlp)

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.metrics import mean_squared_error

# # 1. Load Data
# # ------------
# df_velocity = pd.read_csv("beer_pong_velocity_output.csv").dropna()
# df_impact = pd.read_csv("impact_log.csv")

# # 2. Correct ID Mapping (User Logic: ID 21 matches Throw 1)
# # ---------------------------------------------------------
# df_impact = df_impact.assign(
#     velocity_id=lambda d: d["ID"] - 20,
#     target_x_m=lambda d: d["X_cm"] / 100.0,
#     target_y_m=lambda d: d["Y_cm"] / 100.0,
# )
# # Keep only rows that match our velocity data (IDs 1-30)
# df_impact = df_impact[df_impact["velocity_id"] > 0]

# # Merge
# df_ml = df_velocity.merge(
#     df_impact[["velocity_id", "target_x_m", "target_y_m"]],
#     left_on="throw_id",
#     right_on="velocity_id",
#     how="inner"
# )

# # ---------------------------------------------------------
# # CHANGE: REMOVED .tail(N). WE USE ALL FRAMES NOW.
# # ---------------------------------------------------------
# # Sort by ID and Time to ensure diff() works correctly
# df_ml = df_ml.sort_values(["throw_id", "t"])

# # 3. Feature Engineering
# # ----------------------
# for axis in ["x", "y", "z"]:
#     # Instantaneous velocity (change in position)
#     df_ml[f"d{axis}"] = df_ml.groupby("throw_id")[f"{axis}_raw"].diff()
#     # Instantaneous acceleration (change in velocity)
#     df_ml[f"dv_{axis}"] = df_ml.groupby("throw_id")[f"vel_{axis}_measured"].diff()

# # Remove the first frame of each throw (because diff() produces NaN)
# df_ml = df_ml.dropna()

# FEATURES = [
#     "x_raw", "y_raw", "z_raw", 
#     "dx", "dy", "dz", 
#     "dv_x", "dv_y", "dv_z",
#     "v_err_x", "v_err_y", "v_err_z"
# ]
# X = df_ml[FEATURES]

# # Predict Offset (Residual Learning)
# # "How far from HERE does the ball land?"
# df_ml["dx_target"] = df_ml["target_x_m"] - df_ml["x_raw"]
# df_ml["dy_target"] = df_ml["target_y_m"] - df_ml["y_raw"]

# y_dx = df_ml["dx_target"]
# y_dy = df_ml["dy_target"]
# groups = df_ml["throw_id"]

# # 4. Group Split (Prevents Data Leakage)
# # --------------------------------------
# gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# train_idx, test_idx = next(gss.split(X, y_dx, groups))

# X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
# dx_train, dx_test = y_dx.iloc[train_idx], y_dx.iloc[test_idx]
# dy_train, dy_test = y_dy.iloc[train_idx], y_dy.iloc[test_idx]

# print(f"Training on {len(X_train)} frames. Testing on {len(X_test)} frames.")

# # 5. Train Random Forest
# # ----------------------
# # We increase n_estimators slightly as we have more data
# rf_x = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)
# rf_y = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)

# rf_x.fit(X_train, dx_train)
# rf_y.fit(X_train, dy_train)

# # 6. Evaluation & Visualization
# # -----------------------------
# # Predict
# pred_dx = rf_x.predict(X_test)
# pred_dy = rf_y.predict(X_test)

# # Calculate absolute errors
# x_true = df_ml.iloc[test_idx]["target_x_m"].values
# y_true = df_ml.iloc[test_idx]["target_y_m"].values
# x_pred = X_test["x_raw"].values + pred_dx
# y_pred = X_test["y_raw"].values + pred_dy

# # Calculate error distance for each frame
# errors = np.sqrt(((x_true - x_pred)**2 + (y_true - y_pred)**2)) * 100 # cm

# print(f"\n--- Results on Full Trajectories ---")
# print(f"Mean Prediction Error: {np.mean(errors):.2f} cm")
# print(f"RMSE: {np.sqrt(np.mean(errors**2)):.2f} cm")

# # PLOT: Error vs. Time (using Z height as proxy)
# # High Z = Start of throw. Low Z = End of throw.
# plt.figure(figsize=(10, 6))
# plt.scatter(X_test["z_raw"], errors, alpha=0.6, c='royalblue')
# plt.gca().invert_xaxis() # Invert X so "Start" is on left (High Z) and "End" is on right (Low Z)
# plt.xlabel("Ball Height z(t) [m] (Proxy for Time)")
# plt.ylabel("Prediction Error [cm]")
# plt.title("Prediction Error Evolution during Flight")
# plt.axhline(y=5, color='r', linestyle='--', label="Cup Radius (5cm)")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# 1. Load Data
# -------------------------
# Ensure you are pointing to your actual file paths here
df_velocity = pd.read_csv("beer_pong_velocity_output.csv").dropna()
df_impact = pd.read_csv("impact_log.csv")

# 2. Alignment
# -------------------------
df_impact = df_impact.assign(
    velocity_id=lambda d: d["ID"] - 20,
    target_x_m=lambda d: d["X_cm"] / 100.0,
    target_y_m=lambda d: d["Y_cm"] / 100.0,
)
df_ml = df_velocity.merge(
    df_impact[["velocity_id", "target_x_m", "target_y_m"]],
    left_on="throw_id",
    right_on="velocity_id",
    how="inner"
)

# 3. Feature Engineering
# -------------------------
df_ml = df_ml.sort_values(["throw_id", "t"])
for axis in ["x", "y", "z"]:
    # Calculate displacement and velocity changes
    df_ml[f"d{axis}"] = df_ml.groupby("throw_id")[f"{axis}_raw"].diff().fillna(0)
    df_ml[f"dv_{axis}"] = df_ml.groupby("throw_id")[f"vel_{axis}_measured"].diff().fillna(0)

# Target: Residuals (The difference between current pos and final cup pos)
df_ml["dx_target"] = df_ml["target_x_m"] - df_ml["x_raw"]
df_ml["dy_target"] = df_ml["target_y_m"] - df_ml["y_raw"]

FEATURES = [
    "x_raw", "y_raw", "z_raw", 
    "vel_x_measured", "vel_y_measured", "vel_z_measured",
    "dx", "dy", "dz", 
    "dv_x", "dv_y", "dv_z"
]

X = df_ml[FEATURES]
y_dx = df_ml["dx_target"]
y_dy = df_ml["dy_target"]
groups = df_ml["throw_id"]

# 4. Train/Test Split
# -------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y_dx, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
dx_train, dx_test = y_dx.iloc[train_idx], y_dx.iloc[test_idx]
dy_train, dy_test = y_dy.iloc[train_idx], y_dy.iloc[test_idx]

# 5. Define The Challengers
# -------------------------

# Model A: Random Forest (Baseline)
rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1)

# Model B: Gradient Boosting (Specialist)
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42)

# Model C: SVR (Mathematician)
svr = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVR(C=10, epsilon=0.01, kernel='rbf'))
])

# Model D: Gaussian Process (The Professor's Choice) 
# Kernel: Constant * RBF (smoothness) + WhiteKernel (noise tolerance)
# We add WhiteKernel because sensor data is never perfect.
kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

gpr = Pipeline([
    ("scaler", StandardScaler()),
    ("gpr", GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True, random_state=42))
])

# Model E: Voting Regressor (The Team)
# Added GPR to the voting team as well
voting = VotingRegressor([
    ('rf', rf), 
    ('gb', gb), 
    ('gpr', gpr) 
])

models = {
    "Random Forest": (rf, rf),
    "Gradient Boosting": (gb, gb),
    "SVR": (svr, svr),
    "Gaussian Process": (gpr, gpr),
    "Voting Ensemble": (voting, voting)
}

# 6. The Tournament
# -----------------
print(f"{'Model Name':<20} | {'RMSE (cm)':<10}")
print("-" * 35)

best_rmse = float("inf")
best_model_name = ""

for name, (model_x, model_y) in models.items():
    # Clone models to ensure fresh training for X and Y axes
    m_x = clone(model_x)
    m_y = clone(model_y)
    
    # Train
    m_x.fit(X_train, dx_train)
    m_y.fit(X_train, dy_train)
    
    # Predict residuals
    p_dx = m_x.predict(X_test)
    p_dy = m_y.predict(X_test)
    
    # Reconstruct absolute positions (Prediction = Current_Pos + Predicted_Residual)
    x_pred = X_test["x_raw"] + p_dx
    y_pred = X_test["y_raw"] + p_dy
    
    # Truth
    x_true = df_ml.iloc[test_idx]["target_x_m"]
    y_true = df_ml.iloc[test_idx]["target_y_m"]
    
    # Calculate RMSE (Total Euclidean Error)
    error_dist = np.sqrt(((x_true - x_pred)*100)**2 + ((y_true - y_pred)*100)**2)
    # Note: Your metric effectively averages the component variances. 
    # Standard 2D RMSE is usually sqrt(mean(error_dist^2)). 
    # I kept your specific calculation: sqrt(mean(error_dist^2)/2)
    rmse_total = np.sqrt(np.mean(error_dist**2)/2)
    
    print(f"{name:<20} | {rmse_total:.2f}")
    
    if rmse_total < best_rmse:
        best_rmse = rmse_total
        best_model_name = name

print("-" * 35)
print(f"WINNER: {best_model_name} with {best_rmse:.2f} cm RMSE")