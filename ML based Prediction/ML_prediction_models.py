import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. Load data
# ----------------------------
df_velocity = pd.read_csv("beer_pong_velocity_output.csv").dropna()
df_impact = pd.read_csv("impact_log.csv")

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

# ----------------------------
# 2. Use last N frames only
# ----------------------------
N = 5
df_ml = (
    df_ml
    .sort_values("t")
    .groupby("throw_id", group_keys=False)
    .tail(N)
)

# ----------------------------
# 3. Relative (ML-safe) features
# ----------------------------
for axis in ["x", "y", "z"]:
    df_ml[f"d{axis}"] = df_ml.groupby("throw_id")[f"{axis}_raw"].diff()

for axis in ["x", "y", "z"]:
    df_ml[f"dv_{axis}"] = df_ml.groupby("throw_id")[f"vel_{axis}_measured"].diff()

df_ml = df_ml.dropna()

# ----------------------------
# 4. Features & targets
# ----------------------------
FEATURES = [
    "x_raw", "y_raw", "z_raw", 
    #"vel_x_measured", "vel_y_measured", "vel_z_measured",
    "dx", "dy", "dz",      # Your new relative features
    "dv_x", "dv_y", "dv_z", # Your new velocity delta features
    "v_err_x", "v_err_y", "v_err_z" # ADD THESE BACK for the rubric!
]
X = df_ml[FEATURES]

# Predict offset from last observation (ML-normalized)
df_ml["dx_target"] = df_ml["target_x_m"] - df_ml["x_raw"]
df_ml["dy_target"] = df_ml["target_y_m"] - df_ml["y_raw"]

y_dx = df_ml["dx_target"]
y_dy = df_ml["dy_target"]

groups = df_ml["throw_id"]

# ----------------------------
# 5. Group split
# ----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y_dx, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
dx_train, dx_test = y_dx.iloc[train_idx], y_dx.iloc[test_idx]
dy_train, dy_test = y_dy.iloc[train_idx], y_dy.iloc[test_idx]

# ----------------------------
# 6. Models
# ----------------------------
rf_x = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_y = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1)

mlp_x = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=3000, early_stopping=True, random_state=42))
])

mlp_y = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=3000, early_stopping=True, random_state=42))
])

# ----------------------------
# 7. Train
# ----------------------------
rf_x.fit(X_train, dx_train)
rf_y.fit(X_train, dy_train)

mlp_x.fit(X_train, dx_train)
mlp_y.fit(X_train, dy_train)

# ----------------------------
# 8. Predict final position
# ----------------------------
x_rf = X_test["x_raw"].to_numpy() + rf_x.predict(X_test)
y_rf = X_test["y_raw"].to_numpy() + rf_y.predict(X_test)

x_mlp = X_test["x_raw"].to_numpy() + mlp_x.predict(X_test)
y_mlp = X_test["y_raw"].to_numpy() + mlp_y.predict(X_test)

y_true = df_ml.loc[X_test.index, ["target_x_m", "target_y_m"]].to_numpy()

# ----------------------------
# 9. Evaluation (cm)
# ----------------------------
def plot_landing_point_results(y_true, x_rf, y_rf, x_mlp, y_mlp):
    """
    y_true : ndarray (N,2)  -> ground truth landing points [m]
    x_rf, y_rf : ndarray   -> RF predictions [m]
    x_mlp, y_mlp : ndarray -> MLP predictions [m]
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # Convert to cm for presentation
    y_true_cm = y_true * 100
    rf_pred_cm = np.column_stack([x_rf, y_rf]) * 100
    mlp_pred_cm = np.column_stack([x_mlp, y_mlp]) * 100

    # --------------------------------------------------
    # 1. Ground truth vs predictions (XY scatter)
    # --------------------------------------------------
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true_cm[:, 0], y_true_cm[:, 1], c="black", label="Ground Truth", s=30)
    plt.scatter(rf_pred_cm[:, 0], rf_pred_cm[:, 1], c="tab:blue", alpha=0.7, label="Random Forest")
    plt.scatter(mlp_pred_cm[:, 0], mlp_pred_cm[:, 1], c="tab:orange", alpha=0.7, label="Neural Network")
    plt.xlabel("X position [cm]")
    plt.ylabel("Y position [cm]")
    plt.title("Landing Point Prediction (XY Plane)")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    # --------------------------------------------------
    # 2. Error vectors (direction + magnitude)
    # --------------------------------------------------
    plt.figure(figsize=(7, 7))
    plt.quiver(
        y_true_cm[:, 0], y_true_cm[:, 1],
        rf_pred_cm[:, 0] - y_true_cm[:, 0],
        rf_pred_cm[:, 1] - y_true_cm[:, 1],
        angles="xy", scale_units="xy", scale=1,
        color="tab:blue", alpha=0.6, label="RF Error"
    )
    plt.quiver(
        y_true_cm[:, 0], y_true_cm[:, 1],
        mlp_pred_cm[:, 0] - y_true_cm[:, 0],
        mlp_pred_cm[:, 1] - y_true_cm[:, 1],
        angles="xy", scale_units="xy", scale=1,
        color="tab:orange", alpha=0.6, label="MLP Error"
    )
    plt.xlabel("X position [cm]")
    plt.ylabel("Y position [cm]")
    plt.title("Prediction Error Vectors")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    # --------------------------------------------------
    # 3. Error magnitude distribution
    # --------------------------------------------------
    rf_error = np.linalg.norm(rf_pred_cm - y_true_cm, axis=1)
    mlp_error = np.linalg.norm(mlp_pred_cm - y_true_cm, axis=1)

    plt.figure(figsize=(8, 4))
    plt.hist(rf_error, bins=25, alpha=0.7, label="Random Forest")
    plt.hist(mlp_error, bins=25, alpha=0.7, label="Neural Network")
    plt.xlabel("Landing Point Error [cm]")
    plt.ylabel("Count")
    plt.title("Error Distribution")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --------------------------------------------------
    # 4. Per-sample RMSE comparison
    # --------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(rf_error, label="Random Forest Error", linewidth=2)
    plt.plot(mlp_error, label="Neural Network Error", linewidth=2)
    plt.xlabel("Test Sample Index")
    plt.ylabel("Error [cm]")
    plt.title("Per-Sample Landing Point Error")
    plt.legend()
    plt.grid(True)
    plt.show()

def eval_model(name, x_pred, y_pred):
    preds = np.column_stack([x_pred, y_pred]) * 100
    truth = y_true * 100

    total_rmse = np.sqrt(mean_squared_error(truth, preds))
    axis_rmse = np.sqrt(((truth - preds) ** 2).mean(axis=0))

    print(f"{name}:")
    print(f"  Total RMSE: {total_rmse:.2f} cm")
    print(f"  X RMSE: {axis_rmse[0]:.2f} cm | Y RMSE: {axis_rmse[1]:.2f} cm\n")

eval_model("Random Forest (Pure ML)", x_rf, y_rf)
eval_model("Neural Network (Pure ML)", x_mlp, y_mlp)

# plot_landing_point_results(y_true, x_rf, y_rf, x_mlp, y_mlp)