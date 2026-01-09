import cv2
import numpy as np
import json
import os
from collections import deque
from ball_tracking import get_ball_coords 

# --- CONFIG ---
# Use HW acceleration if available (DirectShow or FFMPEG)
# On Windows/NVIDIA, CAP_FFMPEG is usually the fastest
BACKEND = cv2.CAP_FFMPEG 

# cv2.namedWindow("3D Test Tracker", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("3D Test Tracker", 800, 450)
# cv2.namedWindow("top side", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("top side", 800, 450)
# 1. LOAD CALIBRATIONS
def build_projection_matrix(path):
    data = np.load(path)
    R, _ = cv2.Rodrigues(data['rvecs'])
    Rt = np.hstack((R, data['tvecs'].reshape(3,1)))
    return data['mtx'] @ Rt

P_TOP = build_projection_matrix('calib_top_final_t1.npz')
P_SIDE = build_projection_matrix('calib_side_final_t1.npz')

# 2. VIDEO SETUP WITH HW ACCEL
cap_top = cv2.VideoCapture(r"Recording\Setup 1\SYNC_IMG_6567_c1_t1.MOV", BACKEND)
cap_side = cv2.VideoCapture(r"Recording\Setup 1\SYNC_IMG_0349_c2_t1.MOV", BACKEND)

# --- SEQUENTIAL SYNC VARIABLES ---
top_buffer = deque(maxlen=10) # Holding tank for Top frames
SYNC_RATIO = 59.97 / 59.94
DT = 1.0 / 59.94
Z_OFFSET = 5.49

dataset = []
impact_count, cooldown = 0, 0
prev_pos, prev_z_vel = None, 0

def triangulate_point(pt_top, pt_side):
    p1 = np.array([[pt_top[0]], [pt_top[1]]], dtype=np.float32)
    p2 = np.array([[pt_side[0]], [pt_side[1]]], dtype=np.float32)
    pts4d = cv2.triangulatePoints(P_TOP, P_SIDE, p1, p2)
    pts3d = pts4d[:3] / pts4d[3]
    return [float(pts3d[0][0]), float(pts3d[1][0]), float(pts3d[2][0] - Z_OFFSET)]

print("🚀 Starting GPU-Optimized Sequential Extraction...")

# Start processing
while True:
    ret_s, f_side = cap_side.read()
    if not ret_s: break
    
    side_frame_idx = int(cap_side.get(cv2.CAP_PROP_POS_FRAMES))
    
    # --- SEQUENTIAL TOP-SYNC LOGIC ---
    # Target frame for Top camera based on Side index
    target_top_idx = int(side_frame_idx * SYNC_RATIO)
    
    # Read Top frames until we reach or pass the target index
    # (Usually reads 1 frame per side frame, occasionally skips 1)
    current_top_idx = int(cap_top.get(cv2.CAP_PROP_POS_FRAMES))
    while current_top_idx <= target_top_idx + 2:
        ret_t, f_top = cap_top.read()
        if not ret_t: break
        top_buffer.append((current_top_idx, f_top))
        current_top_idx += 1

    # Find the frame in our buffer closest to target_top_idx
    f_top_match = None
    for idx, frame in top_buffer:
        if idx == target_top_idx:
            f_top_match = frame
            break
    
    # --- BALL PROCESSING ---
    if f_top_match is not None:
        pixel_side = get_ball_coords(f_side, relaxed=False)
        pixel_top = get_ball_coords(f_top_match, relaxed=False)
        
        if pixel_side and pixel_top:
            X, Y, Z = triangulate_point(pixel_top, pixel_side)
            
            # Physics Math
            vx, vy, vz = 0.0, 0.0, 0.0
            if prev_pos:
                vx, vy, vz = (X-prev_pos[0])/DT, (Y-prev_pos[1])/DT, (Z-prev_pos[2])/DT
            
            # Look-Back Impact Logic
            if cooldown == 0 and prev_pos and vz > 50 and prev_z_vel < 0:
                if prev_pos[2] < 2.0:
                    impact_count += 1
                    cooldown = 100
                    if dataset: dataset[-1]["is_impact"] = True
                    print(f"🎯 IMPACT {impact_count} | Frame: {dataset[-1]['frame']} | X: {dataset[-1]['pos']['x']:.2f} | Y: {dataset[-1]['pos']['y']:.2f}| Z: {dataset[-1]['pos']['z']:.2f}")

            dataset.append({
                "frame": side_frame_idx,
                "pos": {"x": round(X, 2), "y": round(Y, 2), "z": round(Z, 2)},
                "vel": {"vx": round(vx, 2), "vy": round(vy, 2), "vz": round(vz, 2)},
                "is_impact": False
            })
            
            # Visuals
            cv2.drawMarker(f_side, (int(pixel_side[0]), int(pixel_side[1])), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(f_side, f"3D X:{X:.1f} Y:{Y:.1f} Z:{Z:.1f}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            prev_pos, prev_z_vel = (X, Y, Z), vz

    #cv2.imshow("3D Test Tracker", f_side)
    #cv2.imshow("top side", f_top)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    if cooldown > 0: cooldown -= 1
    if side_frame_idx % 500 == 0: print(f"Processing... Frame {side_frame_idx}")
    
# SAVE
with open("sequential_dataset_t1_3D.json", 'w') as f:
    json.dump(dataset, f, indent=2)

print(f"\n✅ Finished! {len(dataset)} points recorded.")
cap_top.release()
cap_side.release()