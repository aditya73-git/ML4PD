import pandas as pd
import numpy as np
import cv2
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

# =========================================================
# 1. CAMERA SETUP (Parameters for Cam 1 and Cam 2)
# =========================================================
print("Loading calibration parameters...")

# --- CAMERA 1 SETUP ---
# Load intrinsic matrix (mtx) and distortion coefficients (dist)
cam1 = np.load("Calibration matrices/cam1_calib.npz")
K1, D1 = cam1["mtx"], cam1["dist"]
image_pts_cam1 = np.load("Calibration matrices/cam1_target_image_pts.npy").astype(np.float32)

# --- CAMERA 2 SETUP (With Zoom/Homography logic) ---
cam2 = np.load("Calibration matrices/cam2_calib.npz")
K2_pre, D2 = cam2["mtx"], cam2["dist"]
H2_pre = np.load("Calibration matrices/cam2_homography_prezoom.npy")
H2_post = np.load("Calibration matrices/cam2_homography.npy")
image_pts_cam2 = np.load("Calibration matrices/cam2_target_image_pts_postzoom.npy").astype(np.float32)

# Calculate focal scale adjustment for Cam 2 based on homography change
# H_tilde maps the relationship between the pre-zoom and post-zoom states
H_tilde = np.linalg.inv(K2_pre) @ H2_post @ np.linalg.inv(H2_pre)
scale = (np.linalg.norm(H_tilde[:, 0]) + np.linalg.norm(H_tilde[:, 1])) / 2

# Create the post-zoom intrinsic matrix by scaling focal lengths (fx, fy)
K2_post = K2_pre.copy()
scale_correction = 12.0  # Manual correction factor from physical table measurements
K2_post[0, 0] *= scale * scale_correction
K2_post[1, 1] *= scale * scale_correction

# Define 3D reference points of the calibration target (in meters)
world_pts_target = np.array([
    [-0.3, -0.3, 0.0], [0.3, -0.3, 0.0],
    [0.3,  0.3, 0.0], [-0.3,  0.3, 0.0]
], dtype=np.float32)

# Solve PnP (Perspective-n-Point) to find Extrinsic parameters (Rotation and Translation)
# This aligns the 3D world coordinate system with each camera's 2D view
_, rvec1, tvec1 = cv2.solvePnP(world_pts_target, image_pts_cam1, K1, D1, flags=cv2.SOLVEPNP_IPPE)
_, rvec2, tvec2 = cv2.solvePnP(world_pts_target, image_pts_cam2, K2_post, D2, flags=cv2.SOLVEPNP_IPPE)

def get_pixel_coords(x_m, y_m, z_m, K, D, rvec, tvec):
    """Projects 3D world coordinates (meters) into 2D pixel coordinates."""
    point_3d = np.array([[x_m, y_m, z_m]], dtype=np.float32)
    # cv2.projectPoints handles the perspective transformation and lens distortion
    points_2d, _ = cv2.projectPoints(point_3d, rvec, tvec, K, D)
    u, v = points_2d.ravel()
    return int(u), int(v)

# =========================================================
# 2. ML MODEL TRAINING (Causal Prediction Logic)
# =========================================================
print("Training ML models...")
# Load velocity tracking data and impact labels
df_velocity = pd.read_csv("Speed Tracking/beer_pong_velocity_output.csv").dropna()
df_impact = pd.read_csv("Labelling/impact_log.csv").assign(
    velocity_id=lambda d: d["ID"] - 20,
    target_x_m=lambda d: d["X_cm"] / 100.0, 
    target_y_m=lambda d: d["Y_cm"] / 100.0,
)

# Merge datasets and ensure chronological order for causal processing
df_ml = df_velocity.merge(df_impact[["velocity_id", "target_x_m", "target_y_m"]],
                         left_on="throw_id", right_on="velocity_id", how="inner").sort_values(["throw_id", "t"])

# Feature Engineering: Calculate causal deltas (current - previous) for position and velocity
for axis in ["x", "y", "z"]:
    df_ml[f"d{axis}"] = df_ml.groupby("throw_id")[f"{axis}_raw"].diff().fillna(0)
    df_ml[f"dv_{axis}"] = df_ml.groupby("throw_id")[f"vel_{axis}_measured"].diff().fillna(0)

# Define features used for prediction (instantaneous state + recent deltas)
FEATURES = ["x_raw", "y_raw", "z_raw", "vel_x_measured", "vel_y_measured", "vel_z_measured", "dx", "dy", "dz", "dv_x", "dv_y", "dv_z"]
X = df_ml[FEATURES]
# Target: The offset from current position to the final impact point (Residual Prediction)
y_dx = df_ml["target_x_m"] - df_ml["x_raw"]
y_dy = df_ml["target_y_m"] - df_ml["y_raw"]

# Split data by throw_id to avoid data leakage (don't train on frames from the test throw)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, _ = next(gss.split(X, y_dx, df_ml["throw_id"]))

# Train Random Forest Regressors for X and Y impact offsets
rf_x = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X.iloc[train_idx], y_dx.iloc[train_idx])
rf_y = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X.iloc[train_idx], y_dy.iloc[train_idx])

# =========================================================
# 3. VIDEO RENDERING (Dual Camera + Slow-Mo 5x)
# =========================================================
# Precise frame intervals for the specific throw recorded in the synchronized videos
throw_intervals = [
    (6964, 6990), (7403, 7430), (7606, 7628), (7795, 7822), (7992, 8017),
    (8289, 8317), (8502, 8528), (8954, 8980), (9174, 9198), (9380, 9410),
    (9878, 9904), (10062, 10088), (10665, 10690), (10847, 10875), (11477, 11506),
    (11654, 11679), (11998, 12023), (12180, 12206), (12395, 12425), (12701, 12729),
    (13113, 13141), (13297, 13327), (13487, 13517), (13676, 13706), (13864, 13894),
    (14151, 14181), (14333, 14363), (14529, 14559), (14722, 14752), (14916, 14944)
]

TARGET_THROW_ID = 26
SLOW_MO_FACTOR = 5.0

# Configuration dictionary to loop through both cameras
configs = [
    {"name": "CAM1", "file": "Location Tracking/syncronized_videos/CV_SYNC_IMG_0362.MOV", "K": K1, "D": D1, "rvec": rvec1, "tvec": tvec1},
    {"name": "CAM2", "file": "Location Tracking/syncronized_videos/CV_SYNC_IMG_6590.MOV", "K": K2_post, "D": D2, "rvec": rvec2, "tvec": tvec2}
]

# Get the ML tracking data for the target throw
test_data = df_ml[df_ml['throw_id'] == TARGET_THROW_ID].sort_values('t')



for cfg in configs:
    print(f"Processing {cfg['name']} for Throw {TARGET_THROW_ID}...")
    cap = cv2.VideoCapture(cfg['file'])
    start_frame, _ = throw_intervals[TARGET_THROW_ID - 1]
    
    # Calculate output FPS for slow motion effect
    slow_fps = cap.get(cv2.CAP_PROP_FPS) / SLOW_MO_FACTOR
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Configure VideoWriter with H.264 codec (avc1) for mobile compatibility
    out_path = f"Video predictions/throw_{TARGET_THROW_ID}_{cfg['name']}_5x.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), slow_fps, 
                          (int(cap.get(3)), int(cap.get(4))))

    for i in range(len(test_data)):
        ret, frame = cap.read()
        if not ret: break
        
        # Get ML features for the current frame
        row = test_data.iloc[i]
        feat = row[FEATURES].values.reshape(1, -1)
        
        # Predict the impact point using the current frame's state
        p_x = row['x_raw'] + rf_x.predict(feat)[0]
        p_y = row['y_raw'] + rf_y.predict(feat)[0]
        
        # Project the 3D predicted landing point (z=0) into 2D camera pixels
        u, v = get_pixel_coords(p_x, p_y, 0.0, cfg['K'], cfg['D'], cfg['rvec'], cfg['tvec'])

        # Draw UI Elements: Crosshair and Circle at the predicted impact point
        cv2.drawMarker(frame, (u, v), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 40, 5)
        cv2.circle(frame, (u, v), 30, (0, 0, 255), 4)

        # Draw a massive UI header for visibility on mobile devices
        cv2.rectangle(frame, (0,0), (950, 160), (0,0,0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"{cfg['name']} | THROW {TARGET_THROW_ID} | SLOWMO 5X", (25, 65), font, 2.3, (255, 255, 255), 5)
        cv2.putText(frame, f"PRED: X={p_x:.2f}m Y={p_y:.2f}m", (25, 135), font, 2.1, (0, 255, 255), 4)
        
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video {cfg['name']} saved successfully.")

print("\nOperation completed for both cameras.")