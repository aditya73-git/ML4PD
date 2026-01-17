import os
import sys
import numpy as np
import cv2
from ultralytics import YOLO
import csv

# --------------------------------------------------
# Project path & Calibration Loading
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def load_calib(path):
    data = np.load(path)
    return data["mtx"], data["dist"]

K1, D1 = load_calib(r"ML4PD/Calibration matrices/cam1_calib.npz")
K2_pre, D2 = load_calib(r"ML4PD/Calibration matrices/cam2_calib.npz")

# Zoom correction for Camera 2
H2_pre = np.load(r"ML4PD/Calibration matrices/cam2_homography_prezoom.npy")
H2_post = np.load(r"ML4PD/Calibration matrices/cam2_homography.npy")

H_tilde = np.linalg.inv(K2_pre) @ H2_post @ np.linalg.inv(H2_pre)
scale = (np.linalg.norm(H_tilde[:, 0]) + np.linalg.norm(H_tilde[:, 1])) / 2

K2_post = K2_pre.copy()
K2_post[0, 0] *= (scale * 12.0) # Applying your empirical correction
K2_post[1, 1] *= (scale * 12.0)

# --------------------------------------------------
# Solve PnP and Projection Matrices
# --------------------------------------------------
world_pts = np.array([[-0.3, -0.3, 0.0], [0.3, -0.3, 0.0], 
                     [0.3, 0.3, 0.0], [-0.3, 0.3, 0.0]], dtype=np.float32)

img_pts_cam1 = np.load(r"ML4PD/Calibration matrices/cam1_target_image_pts.npy").astype(np.float32)
img_pts_cam2 = np.load(r"ML4PD/Calibration matrices/cam2_target_image_pts_postzoom.npy").astype(np.float32)

_, rvec1, tvec1 = cv2.solvePnP(world_pts, img_pts_cam1, K1, D1, flags=cv2.SOLVEPNP_IPPE)
_, rvec2, tvec2 = cv2.solvePnP(world_pts, img_pts_cam2, K2_post, D2, flags=cv2.SOLVEPNP_IPPE)

def get_projection_matrix(K, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    Rt = np.hstack((R, tvec))
    return K @ Rt

P1 = get_projection_matrix(K1, rvec1, tvec1)
P2 = get_projection_matrix(K2_post, rvec2, tvec2)

# --------------------------------------------------
# Triangulation Logic
# --------------------------------------------------


def triangulate_point(pt1, pt2, K1, D1, K2, D2, P1, P2):
    # 1. Undistort points
    # Points must be in shape (1, 1, 2) for undistortPoints
    p1 = np.array([[pt1]], dtype=np.float32)
    p2 = np.array([[pt2]], dtype=np.float32)
    
    # We pass P=None to get normalized coordinates (needed for triangulation)
    # but since we are using Projection Matrices P1/P2 that include K, 
    # we should ideally provide K here to get ideal coordinates.
    p1_u = cv2.undistortPoints(p1, K1, D1, P=K1)
    p2_u = cv2.undistortPoints(p2, K2, D2, P=K2)
    
    # 2. Triangulate (Expects 2xN arrays)
    X_hom = cv2.triangulatePoints(P1, P2, p1_u.reshape(2,1), p2_u.reshape(2,1))
    
    # 3. Convert from Homogeneous to 3D
    X = X_hom[:3] / X_hom[3]
    return X.flatten()

# --------------------------------------------------
# YOLO and Processing Setup
# --------------------------------------------------
model = YOLO(r"ML4PD/runs/detect/train/weights/best.pt")
cap1 = cv2.VideoCapture(r"Recording/Setup 3/CV_SYNC_IMG_0362.MOV")
cap2 = cv2.VideoCapture(r"Recording/Setup 3/CV_SYNC_IMG_6590.MOV")

trajectory = []
is_playing = True
total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
cv2.namedWindow("Stereo Ball Tracking (XYZ in cm)", cv2.WINDOW_NORMAL)

while True:
    # 1. READ FRAMES
    # We only read automatically if playing. 
    # If paused, we only read if a seek key was pressed (handled below).
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        # If we hit the end, stop playing but don't close the window
        is_playing = False
        print("End of video reached.")
        # Optional: reset to beginning or break
        # break 
    
    curr_idx = int(cap1.get(cv2.CAP_PROP_POS_FRAMES))

    # 2. DETECTION & TRIANGULATION (Only run if frames were successfully read)
    if ret1 and ret2:
        res1 = model(frame1, verbose=False)[0]
        res2 = model(frame2, verbose=False)[0]

        pt1 = pt2 = None
        box1 = box2 = None

        if len(res1.boxes) > 0:
            box1 = res1.boxes.xyxy[0].cpu().numpy()
            pt1 = ((box1[0]+box1[2])/2, (box1[1]+box1[3])/2)
        
        if len(res2.boxes) > 0:
            box2 = res2.boxes.xyxy[0].cpu().numpy()
            pt2 = ((box2[0]+box2[2])/2, (box2[1]+box2[3])/2)

        xyz_cm = None
        if pt1 and pt2:
            xyz_cm = triangulate_point(pt1, pt2, K1, D1, K2_post, D2, P1, P2) * 100.0
            trajectory.append([curr_idx, *xyz_cm])

        # 3. VISUALIZATION
        h_disp, w_disp = 480, 480
        f1_res = cv2.resize(frame1, (w_disp, h_disp))
        f2_res = cv2.resize(frame2, (w_disp, h_disp))
        
        def draw_info(img, box, xyz, scale_w, scale_h):
            if box is not None:
                bx = [box[0]*scale_w, box[1]*scale_h, box[2]*scale_w, box[3]*scale_h]
                cv2.rectangle(img, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0,255,0), 2)
                if xyz is not None:
                    txt = f"X:{xyz[0]:.1f} Y:{xyz[1]:.1f} Z:{xyz[2]:.1f}"
                    cv2.putText(img, txt, (int(bx[0]), int(bx[1])-10), 0, 0.5, (0,255,0), 2)

        draw_info(f1_res, box1, xyz_cm, w_disp/frame1.shape[1], h_disp/frame1.shape[0])
        draw_info(f2_res, box2, xyz_cm, w_disp/frame2.shape[1], h_disp/frame2.shape[0])

        combined = np.hstack((f1_res, f2_res))
        cv2.putText(combined, f"Frame: {curr_idx}/{total_frames} {'[PLAY]' if is_playing else '[PAUSE]'}", 
                    (20, 30), 0, 0.7, (0, 0, 255) if not is_playing else (0, 255, 0), 2)
        cv2.imshow("Stereo Ball Tracking (XYZ in cm)", combined)

    # 4. KEYBOARD LOGIC
    # WaitKey logic: if playing, wait 1ms; if paused, wait indefinitely until a key is pressed
    key = cv2.waitKey(1 if is_playing else 0) & 0xFF

    if key == ord('q'):
        break
    elif key == 32: # SPACE to toggle play/pause
        is_playing = not is_playing
    elif key == ord('d'): # Step forward 1 frame
        # cap.read() moves the pointer forward, so we don't strictly need to set it 
        # unless we want to jump multiple frames. To 'refresh' while paused:
        pass # The next loop iteration will naturally read the next frame
    elif key == ord('a'): # Step backward 1 frame
        # To step back, we must set the pointer to 2 frames back (because read() advances it)
        new_pos = max(0, curr_idx - 2)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
    
    # If paused and no key was pressed, we continue the loop 
    # but we need to prevent cap.read() from advancing if we didn't press a 'step' key.
    if not is_playing and key == 255: # 255 means no key was pressed
        # Re-set to current frame so the next read() shows the same frame
        cap1.set(cv2.CAP_PROP_POS_FRAMES, curr_idx - 1)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, curr_idx - 1)

cap1.release()
cap2.release()
cv2.destroyAllWindows()

# Save CSV
with open("trajectory_3d_cm.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "x_cm", "y_cm", "z_cm"])
    writer.writerows(trajectory)