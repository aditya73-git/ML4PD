import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
# Importing matrices and triangulation function from your original file
from tracking_code_trial2 import P1, P2, triangulate_point

# --- CONFIGURATION ---
VIDEO_CAM1 = "Location Tracking/syncronized_videos/CV_SYNC_IMG_0362.MOV"
VIDEO_CAM2 = "Location Tracking/syncronized_videos/CV_SYNC_IMG_6590.MOV"
MODEL_PATH = "runs/detect/train/weights/best.pt"
OUTPUT_CSV = "Location Tracking/beer_pong_trajectories.csv"

# Annotated throw intervals (start_frame, end_frame)
throw_intervals = [
    (6964, 6990), # throw 1
    (7403, 7430), # throw 2
    (7606, 7628), # throw 3
    (7795, 7822), # throw 4
    (7992, 8017), # throw 5
    (8289, 8317), # throw 6
    (8502, 8528), # throw 7
    (8954, 8980), # throw 8
    (9174, 9198), # throw 9
    (9380, 9410), # throw 10
    (9878, 9904), # throw 11
    (10062, 10088), # throw 12
    (10665, 10690), # throw 13
    (10847, 10875), # throw 14
    (11477, 11506), # throw 15
    (11654, 11679), # throw 16
    (11998, 12023), # throw 17
    (12180, 12206), # throw 18
    (12395, 12425), # throw 19
    (12701, 12729), # throw 20
    (13113, 13141), # throw 21
    (13297, 13327), # throw 22
    (13487, 13517), # throw 23
    (13676, 13706), # throw 24
    (13864, 13894), # throw 25
    (14151, 14181), # throw 26
    (14333, 14363), # throw 27
    (14529, 14559), # throw 28
    (14722, 14752), # throw 29
    (14916, 14944), # throw 30
]

def run_trajectory_extraction():
    model = YOLO(MODEL_PATH)
    cap1 = cv2.VideoCapture(VIDEO_CAM1)
    cap2 = cv2.VideoCapture(VIDEO_CAM2)
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    raw_data = []

    print("--- Phase 1: Position Tracking & Triangulation ---")
    for throw_id, (start, end) in enumerate(throw_intervals, 1):
        cap1.set(cv2.CAP_PROP_POS_FRAMES, start)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, start)

        for current_f in range(start, end + 1):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2: break

            res1 = model.predict(frame1, conf=0.4, verbose=False)[0]
            res2 = model.predict(frame2, conf=0.4, verbose=False)[0]

            t_rel = (current_f - start) / fps
            
            if len(res1.boxes) > 0 and len(res2.boxes) > 0:
                c1 = res1.boxes.xywh[0].cpu().numpy()[:2]
                c2 = res2.boxes.xywh[0].cpu().numpy()[:2]
                X_3d = triangulate_point(P1, P2, c1, c2)
                
                raw_data.append({
                    'throw_id': throw_id, 't': t_rel,
                    'x_raw': X_3d[0], 'y_raw': X_3d[1], 'z_raw': X_3d[2]
                })
            else:
                raw_data.append({
                    'throw_id': throw_id, 't': t_rel,
                    'x_raw': np.nan, 'y_raw': np.nan, 'z_raw': np.nan
                })

    cap1.release()
    cap2.release()
    
    df = pd.DataFrame(raw_data)

    print("--- Phase 2: Interpolation, Trimming, and Coefficient Calculation ---")
    
    # 1. Preliminary Interpolation to find the bounce point reliably
    df[['x_raw', 'y_raw', 'z_raw']] = df.groupby('throw_id')[['x_raw', 'y_raw', 'z_raw']].transform(
        lambda x: x.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    )

    processed_list = []
    for tid, group in df.groupby('throw_id'):
        # TO REMOVE THE TAIL: We look for the FIRST time the ball reaches the minimum height
        # This prevents including frames where the ball is rolling on the table
        min_z = group['z_raw'].min()
        # Find the first index where Z is close to the minimum (with a small tolerance for noise)
        bounce_idx = group[group['z_raw'] <= (min_z + 0.001)].index[0]
        
        # Trim: keep data from start until the first impact point
        trimmed_group = group.loc[:bounce_idx].copy()
        t = trimmed_group['t'].values
        
        # 2. Regression on Trimmed Data
        # Linear fit for horizontal (X, Y), Quadratic (parabolic) for vertical (Z)
        cx = np.polyfit(t, trimmed_group['x_raw'], 1)
        cy = np.polyfit(t, trimmed_group['y_raw'], 1)
        cz = np.polyfit(t, trimmed_group['z_raw'], 2)
        
        trimmed_group['coeff_x_1'], trimmed_group['coeff_x_0'] = cx[0], cx[1]
        trimmed_group['coeff_y_1'], trimmed_group['coeff_y_0'] = cy[0], cy[1]
        trimmed_group['coeff_z_2'], trimmed_group['coeff_z_1'], trimmed_group['coeff_z_0'] = cz[0], cz[1], cz[2]
        
        processed_list.append(trimmed_group)

    final_df = pd.concat(processed_list)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset successfully created and trimmed: {OUTPUT_CSV}")

if __name__ == "__main__":
    # Ensure the extraction actually runs
    run_trajectory_extraction()

    # --- DEBUG PLOTTING SECTION ---
    if os.path.exists(OUTPUT_CSV):
        df_debug = pd.read_csv(OUTPUT_CSV)
        target_throw_id = 26 

        if target_throw_id in df_debug['throw_id'].unique():
            group = df_debug[df_debug['throw_id'] == target_throw_id]
            t = group['t'].values
            
            # 1. Reconstruct fitted trajectories (in meters)
            x_m = group['coeff_x_1'].iloc[0] * t + group['coeff_x_0'].iloc[0]
            y_m = group['coeff_y_1'].iloc[0] * t + group['coeff_y_0'].iloc[0]
            z_m = group['coeff_z_2'].iloc[0] * t**2 + group['coeff_z_1'].iloc[0] * t + group['coeff_z_0'].iloc[0]

            # 2. CONVERSION TO CM FOR PLOTTING
            # We multiply both raw data and fitted lines by 100
            plt.figure(figsize=(15, 5))
            
            # X Axis (cm)
            plt.subplot(1, 3, 1)
            plt.plot(t, group['x_raw'] * 100, 'o', label='Raw (cm)', alpha=0.5)
            plt.plot(t, x_m * 100, '-', label='Fit (cm)', color='red')
            plt.xlabel('Time (s)')
            plt.ylabel('X Position (cm)')
            plt.title(f'X - Throw {target_throw_id}')
            plt.legend()

            # Y Axis (cm)
            plt.subplot(1, 3, 2)
            plt.plot(t, group['y_raw'] * 100, 'o', label='Raw (cm)', alpha=0.5)
            plt.plot(t, y_m * 100, '-', label='Fit (cm)', color='green')
            plt.xlabel('Time (s)')
            plt.ylabel('Y Position (cm)')
            plt.title(f'Y - Throw {target_throw_id}')
            plt.legend()

            # Z Axis (cm)
            plt.subplot(1, 3, 3)
            plt.plot(t, group['z_raw'] * 100, 'o', label='Raw (cm)', alpha=0.5)
            plt.plot(t, z_m * 100, '-', label='Fit (cm)', color='blue')
            plt.xlabel('Time (s)')
            plt.ylabel('Z Position (cm)')
            plt.title(f'Z - Throw {target_throw_id}')
            plt.legend()

            plt.tight_layout()
            plt.show()