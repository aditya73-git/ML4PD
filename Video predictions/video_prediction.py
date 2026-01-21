import pandas as pd
import numpy as np
import cv2
import os
import sys
import joblib
import tensorflow as tf

# =========================================================
# 0. PATH SETUP AND MATRIX IMPORTS
# =========================================================
BASE_DIR = os.getcwd()

# Add paths to locate tracking and PINN modules
sys.path.append(os.path.join(BASE_DIR, "Location Tracking"))
sys.path.append(os.path.join(BASE_DIR, "Physics Informed ML"))

# Import P1 and P2 projection matrices from your tracking file
from tracking_code_trial2 import P1, P2 
# Import the PINN class from Aditya's project
from models.pinn_model import SimplePINN

# =========================================================
# 1. PROJECTION AND FEATURE ENGINEERING FUNCTIONS
# =========================================================

def project_3d_to_2d(x, y, z, P):
    """Uses the 3x4 projection matrix P to obtain pixel coordinates (u,v)"""
    point_3d = np.array([x, y, z, 1.0])
    point_2d = P @ point_3d
    u = int(point_2d[0] / point_2d[2])
    v = int(point_2d[1] / point_2d[2])
    return u, v

def get_gpr_features(df_partial):
    """Generates the 22 features in the exact order required by the GPR model"""
    res = {}
    # Calculate scalar velocity (magnitude)
    v_mag = np.sqrt(df_partial["vel_x_measured"]**2 + df_partial["vel_y_measured"]**2 + df_partial["vel_z_measured"]**2)
    
    # 1. TEMPORAL FEATURE
    res["duration"] = df_partial["t"].max() - df_partial["t"].min()
    
    # 2. POSITION & VELOCITY STATS
    for ax in ["x", "y", "z"]:
        # Raw Position stats
        res[f"{ax}_raw_first"] = df_partial[f"{ax}_raw"].iloc[0]
        res[f"{ax}_raw_last"] = df_partial[f"{ax}_raw"].iloc[-1]
        res[f"{ax}_raw_std"] = df_partial[f"{ax}_raw"].std() if len(df_partial) > 1 else 0.0
        # Measured Velocity stats
        res[f"vel_{ax}_measured_mean"] = df_partial[f"vel_{ax}_measured"].mean()
        res[f"vel_{ax}_measured_last"] = df_partial[f"vel_{ax}_measured"].iloc[-1]
        res[f"vel_{ax}_measured_std"] = df_partial[f"vel_{ax}_measured"].std() if len(df_partial) > 1 else 0.0
        
    # 3. ENERGY STATS
    res["vel_mag_mean"] = v_mag.mean()
    res["vel_mag_max"] = v_mag.max()
    res["vel_mag_last"] = v_mag.iloc[-1]

    # Mandatory column order for GPR StandardScaler
    cols = ["duration", "x_raw_first", "x_raw_last", "x_raw_std", "y_raw_first", "y_raw_last", "y_raw_std", 
            "z_raw_first", "z_raw_last", "z_raw_std", "vel_x_measured_mean", "vel_x_measured_last", 
            "vel_x_measured_std", "vel_y_measured_mean", "vel_y_measured_last", "vel_y_measured_std", 
            "vel_z_measured_mean", "vel_z_measured_last", "vel_z_measured_std", "vel_mag_mean", 
            "vel_mag_max", "vel_mag_last"]
    
    return pd.DataFrame([res])[cols].fillna(0.0)

# =========================================================
# 2. MODEL LOADING
# =========================================================
print("Loading AI models...")

# PINN Model (6 features)
pinn_model = SimplePINN()
# Dummy pass to initialize weights and layers
_ = pinn_model(tf.convert_to_tensor(np.zeros((1, 6)), dtype=tf.float32)) 
# Load only the weights from the H5 file
pinn_model.load_weights(r"Physics Informed ML/final_validation_results/pinn_model.h5")
scaler_X_pinn = joblib.load(r"Physics Informed ML/final_validation_results/scaler_X.pkl")
scaler_y_pinn = joblib.load(r"Physics Informed ML/final_validation_results/scaler_y.pkl")

# GPR Models (22 features)
gpr_x = joblib.load('ML based Prediction/Trained Models/beer_pong_gpr_x.pkl')
gpr_y = joblib.load('ML based Prediction/Trained Models/beer_pong_gpr_y.pkl')

# =========================================================
# 3. DUAL-CAMERA RENDERING LOOP
# =========================================================
TARGET_THROW_ID = 26
SLOW_MO_FACTOR = 5.0

# Sync intervals (start_frame, end_frame)
throw_intervals = [
    (6964, 6990), (7403, 7430), (7606, 7628), (7795, 7822), (7992, 8017),
    (8289, 8317), (8502, 8528), (8954, 8980), (9174, 9198), (9380, 9410),
    (9878, 9904), (10062, 10088), (10665, 10690), (10847, 10875), (11477, 11506),
    (11654, 11679), (11998, 12023), (12180, 12206), (12395, 12425), (12701, 12729),
    (13113, 13141), (13297, 13327), (13487, 13517), (13676, 13706), (13864, 13894),
    (14151, 14181), (14333, 14363), (14529, 14559), (14722, 14752), (14916, 14944)
]

df_vel = pd.read_csv("Speed Tracking/beer_pong_velocity_output.csv")
test_data = df_vel[df_vel['throw_id'] == TARGET_THROW_ID].sort_values('t')
start_f, end_f = throw_intervals[TARGET_THROW_ID - 1]

camera_configs = [
    {"name": "CAM1", "path": "Location Tracking/syncronized_videos/CV_SYNC_IMG_0362.MOV", "P": P1},
    {"name": "CAM2", "path": "Location Tracking/syncronized_videos/CV_SYNC_IMG_6590.MOV", "P": P2}
]

for cfg in camera_configs:
    print(f"\nProcessing {cfg['name']}...")
    cap = cv2.VideoCapture(cfg['path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    width, height = int(cap.get(3)), int(cap.get(4))
    out_fps = cap.get(cv2.CAP_PROP_FPS) / SLOW_MO_FACTOR
    
    # Video output using MP4 codec
    output_name = f"Video predictions/Comparison_Throw_{TARGET_THROW_ID}_{cfg['name']}.mp4"
    out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (width, height))

    try:
        for i in range(len(test_data)):
            ret, frame = cap.read()
            if not ret or (start_f + i) > end_f: break
            
            sub = test_data.iloc[:i+1]
            row = test_data.iloc[i]

            # 1. PINN Prediction
            cur_pinn = np.array([[row['x_raw'], row['y_raw'], row['z_raw'],
                                  row['vel_x_measured'], row['vel_y_measured'], row['vel_z_measured']]])
            y_p_raw = pinn_model.predict(scaler_X_pinn.transform(cur_pinn), verbose=0)
            y_pinn = scaler_y_pinn.inverse_transform(y_p_raw)[0]

            # 2. GPR Prediction
            cur_gpr = get_gpr_features(sub)
            px_g = gpr_x.predict(cur_gpr)[0]
            py_g = gpr_y.predict(cur_gpr)[0]

            # 3. Coordinate Projection (z=0 table plane)
            up, vp = project_3d_to_2d(y_pinn[0], y_pinn[1], 0.0, cfg['P'])
            ug, vg = project_3d_to_2d(px_g, py_g, 0.0, cfg['P'])

            # 4. Drawing Overlay
            cv2.drawMarker(frame, (up, vp), (255, 255, 0), cv2.MARKER_CROSS, 40, 3) # PINN (Cyan)
            cv2.drawMarker(frame, (ug, vg), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 40, 3) # GPR (Magenta)

            # UI Graphics
            cv2.rectangle(frame, (0,0), (950, 160), (0,0,0), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"{cfg['name']} | THROW {TARGET_THROW_ID} | SLOW-MO", (30, 60), font, 1.8, (255, 255, 255), 3)
            cv2.putText(frame, f"PINN (Cyan): X={y_pinn[0]:.2f} Y={y_pinn[1]:.2f}", (30, 110), font, 1.3, (255, 255, 0), 2)
            cv2.putText(frame, f"GPR (Magenta): X={px_g:.2f} Y={py_g:.2f}", (30, 150), font, 1.3, (255, 0, 255), 2)

            out.write(frame)
            
        print(f"✅ Saved: {output_name}")
    finally:
        cap.release()
        out.release()

print("\nProcessing completed for both cameras.")