import pandas as pd
import numpy as np
import cv2
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.base import clone
from ultralytics import YOLO

# ---------------------------
# PATH SETUP
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
# project_root_1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # already defined
yolo_model_path = os.path.join(project_root, "ML4PD", "runs", "detect", "train", "weights", "best.pt")

cam1_path = os.path.join(project_root, "Recording", "Setup 3", "CV_SYNC_IMG_0362.MOV")
cam2_path = os.path.join(project_root, "Recording", "Setup 3", "CV_SYNC_IMG_6590.MOV")

# ---------------------------
# 1. CAMERA CALIBRATION
# ---------------------------
print("Loading calibration parameters...")

cam1 = np.load("Calibration matrices/cam1_calib.npz")
K1, D1 = cam1["mtx"], cam1["dist"]
image_pts_cam1 = np.load("Calibration matrices/cam1_target_image_pts.npy").astype(np.float32)

cam2 = np.load("Calibration matrices/cam2_calib.npz")
K2_pre, D2 = cam2["mtx"], cam2["dist"]
H2_pre = np.load("Calibration matrices/cam2_homography_prezoom.npy")
H2_post = np.load("Calibration matrices/cam2_homography.npy")
image_pts_cam2 = np.load("Calibration matrices/cam2_target_image_pts_postzoom.npy").astype(np.float32)

# Zoom scale correction
H_tilde = np.linalg.inv(K2_pre) @ H2_post @ np.linalg.inv(H2_pre)
scale = (np.linalg.norm(H_tilde[:, 0]) + np.linalg.norm(H_tilde[:, 1])) / 2
K2_post = K2_pre.copy()
scale_correction = 12.0
K2_post[0, 0] *= scale * scale_correction
K2_post[1, 1] *= scale * scale_correction

# Solve PnP
world_pts_target = np.array([[-0.3, -0.3, 0.0], [0.3, -0.3, 0.0], [0.3, 0.3, 0.0], [-0.3, 0.3, 0.0]], dtype=np.float32)
_, rvec1, tvec1 = cv2.solvePnP(world_pts_target, image_pts_cam1, K1, D1, flags=cv2.SOLVEPNP_IPPE)
_, rvec2, tvec2 = cv2.solvePnP(world_pts_target, image_pts_cam2, K2_post, D2, flags=cv2.SOLVEPNP_IPPE)

def get_pixel_coords(x_m, y_m, z_m, K, D, rvec, tvec):
    point_3d = np.array([[x_m, y_m, z_m]], dtype=np.float32)
    points_2d, _ = cv2.projectPoints(point_3d, rvec, tvec, K, D)
    u, v = points_2d.ravel()
    return int(u), int(v)

# ---------------------------
# 2. LOAD DATA AND TRAIN ML MODELS
# ---------------------------
print("Loading data and training ML models...")

vel = pd.read_csv("Speed Tracking/beer_pong_velocity_output.csv")
impact = pd.read_csv("Labelling/impact_log.csv")
impact["throw_id"] = impact["ID"] - 20
vel["vel_mag"] = np.sqrt(vel["vel_x_measured"]**2 + vel["vel_y_measured"]**2 + vel["vel_z_measured"]**2)

def get_features(df_partial):
    res = {}
    res["duration"] = df_partial["t"].max() - df_partial["t"].min()
    for ax in ["x", "y", "z"]:
        col = f"{ax}_raw"
        res[f"{col}_first"] = df_partial[col].iloc[0]
        res[f"{col}_last"] = df_partial[col].iloc[-1]
        res[f"{col}_std"] = df_partial[col].std() if len(df_partial) > 1 else 0.0
    for ax in ["x", "y", "z"]:
        col = f"vel_{ax}_measured"
        res[f"{col}_mean"] = df_partial[col].mean()
        res[f"{col}_last"] = df_partial[col].iloc[-1]
        res[f"{col}_std"] = df_partial[col].std() if len(df_partial) > 1 else 0.0
    res["vel_mag_mean"] = df_partial["vel_mag"].mean()
    res["vel_mag_max"] = df_partial["vel_mag"].max()
    res["vel_mag_last"] = df_partial["vel_mag"].iloc[-1]
    return pd.Series(res)

TARGET_THROW_ID = 26
data_rows, targets_x, targets_y = [], [], []

common_ids = sorted(list(set(vel["throw_id"]).intersection(set(impact["throw_id"]))))
train_ids = [tid for tid in common_ids if tid != TARGET_THROW_ID]

for tid in train_ids:
    full_throw = vel[vel["throw_id"] == tid].sort_values("t")
    tgt = impact[impact["throw_id"] == tid]
    if len(full_throw) > 0 and len(tgt) > 0:
        data_rows.append(get_features(full_throw))
        targets_x.append(tgt["X_cm"].values[0] / 100.0)
        targets_y.append(tgt["Y_cm"].values[0] / 100.0)

X_train = pd.DataFrame(data_rows)
y_x_train = np.array(targets_x)
y_y_train = np.array(targets_y)

# Define ML pipelines
kernel = 1.0 * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-4)

gpr_base = Pipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(f_regression, k=15)),
    ("gpr", GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, random_state=42))
])

gb_base = Pipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(f_regression, k=15)),
    ("gb", GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42))
])

gpr_x, gpr_y = clone(gpr_base), clone(gpr_base)
gb_x, gb_y = clone(gb_base), clone(gb_base)

gpr_x.fit(X_train, y_x_train)
gpr_y.fit(X_train, y_y_train)
gb_x.fit(X_train, y_x_train)
gb_y.fit(X_train, y_y_train)

print("Models trained successfully.")

# ---------------------------
# 3. LOAD YOLO MODEL
# ---------------------------
yolo_model = YOLO(yolo_model_path)

# ---------------------------
# 4. VIDEO RENDERING LOOP
# ---------------------------
throw_intervals = [
    (6964, 6990), (7403, 7430), (7606, 7628), (7795, 7822), (7992, 8017),
    (8289, 8317), (8502, 8528), (8954, 8980), (9174, 9198), (9380, 9410),
    (9878, 9904), (10062, 10088), (10665, 10690), (10847, 10875), (11477, 11506),
    (11654, 11679), (11998, 12023), (12180, 12206), (12395, 12425), (12701, 12729),
    (13113, 13141), (13297, 13327), (13487, 13517), (13676, 13706), (13864, 13894),
    (14151, 14181), (14333, 14363), (14529, 14559), (14722, 14752), (14916, 14944)
]

configs = [
    {"name": "CAM1", "file": cam1_path, "K": K1, "D": D1, "rvec": rvec1, "tvec": tvec1},
    {"name": "CAM2", "file": cam2_path, "K": K2_post, "D": D2, "rvec": rvec2, "tvec": tvec2}
]

throw_data = vel[vel['throw_id'] == TARGET_THROW_ID].sort_values('t')
SLOW_MO_FACTOR = 5.0

for cfg in configs:
    print(f"Rendering {cfg['name']}...")
    cap = cv2.VideoCapture(cfg['file'])
    
    start_frame, end_frame = throw_intervals[TARGET_THROW_ID - 1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    out_path = f"Video predictions/Pipeline_Throw_{TARGET_THROW_ID}_{cfg['name']}.avi"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), fps / SLOW_MO_FACTOR, (width, height))
    
    frame_idx = 0
    while frame_idx < len(throw_data):
        ret, frame = cap.read()
        if not ret: break
        current_data_row = throw_data.iloc[frame_idx]

        # --- YOLO DETECTION ---
        results = yolo_model(frame)[0]
        if len(results.boxes) > 0:
            box = results.boxes.xyxy[0].cpu().numpy()
            u_curr = int((box[0] + box[2]) / 2)
            v_curr = int((box[1] + box[3]) / 2)
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, "YOLO DETECTION", (u_curr - 40, v_curr - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # fallback to 3D projection
            u_curr, v_curr = get_pixel_coords(current_data_row['x_raw'], current_data_row['y_raw'], current_data_row['z_raw'],
                                              cfg['K'], cfg['D'], cfg['rvec'], cfg['tvec'])

        # --- VELOCITY HUD ---
        vel_mag = current_data_row['vel_mag']
        cv2.putText(frame, f"VEL: {vel_mag:.2f} m/s", (u_curr + 30, v_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # --- ML PREDICTION ---
        if frame_idx > 3:
            features = pd.DataFrame([get_features(throw_data.iloc[:frame_idx+1])])
            # GPR
            pred_x_m = gpr_x.predict(features)[0]
            pred_y_m = gpr_y.predict(features)[0]
            u_pred, v_pred = get_pixel_coords(pred_x_m, pred_y_m, 0.0, cfg['K'], cfg['D'], cfg['rvec'], cfg['tvec'])
            cv2.drawMarker(frame, (u_pred, v_pred), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 3)
            cv2.circle(frame, (u_pred, v_pred), 20, (0, 0, 255), 2)

            # Gradient Boosting
            gb_x_m = gb_x.predict(features)[0]
            gb_y_m = gb_y.predict(features)[0]
            u_gb, v_gb = get_pixel_coords(gb_x_m, gb_y_m, 0.0, cfg['K'], cfg['D'], cfg['rvec'], cfg['tvec'])
            cv2.circle(frame, (u_gb, v_gb), 10, (255, 200, 0), -1)

        # --- HUD ---
        cv2.rectangle(frame, (0, 0), (700, 120), (0, 0, 0), -1)
        cv2.putText(frame, "PIPELINE VISUALIZATION", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, f"GREEN: YOLO | RED: GPR | BLUE: GB", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved {out_path}")

print("Done.")
