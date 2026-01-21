import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from test2 import detect_two_images
import glob
import numpy as np
import cv2
from ultralytics import YOLO


CHECKERBOARD = (10, 7)       # inner corners
SQUARE_SIZE = 0.015         # meters


# Camera 1 (already valid)
cam1 = np.load(
    "Calibration matrices/cam1_calib.npz")
K1 = cam1["mtx"]
D1 = cam1["dist"]

# Camera 2 pre-zoom
cam2 = np.load(
    "Calibration matrices/cam2_calib.npz")
K2_pre = cam2["mtx"]
D2 = cam2["dist"]

# Homographies
H2_pre = np.load(
    "Calibration matrices/cam2_homography_prezoom.npy")
H2_post = np.load(
    "Calibration matrices/cam2_homography.npy")
H_tilde = np.linalg.inv(K2_pre) @ H2_post @ np.linalg.inv(H2_pre)

s1 = np.linalg.norm(H_tilde[:, 0])
s2 = np.linalg.norm(H_tilde[:, 1])
scale = (s1 + s2) / 2

print("Estimated focal scale:", scale)

K2_post = K2_pre.copy()
K2_post[0, 0] *= scale   # fx
K2_post[1, 1] *= scale   # fy

scale_correction = 12.0  # estimated from table size comparison

K2_post[0, 0] *= scale_correction
K2_post[1, 1] *= scale_correction


print("Camera 2 post-zoom intrinsics:\n", K2_post)

world_pts_target = np.array([
    [-0.3, -0.3, 0.0],
    [0.3, -0.3, 0.0],
    [0.3,  0.3, 0.0],
    [-0.3,  0.3, 0.0]
], dtype=np.float32)

image_pts_cam2 = np.load(
    "Calibration matrices/cam2_target_image_pts_postzoom.npy").astype(np.float32)

_, rvec2, tvec2 = cv2.solvePnP(
    world_pts_target,
    image_pts_cam2,
    K2_post,
    D2,
    flags=cv2.SOLVEPNP_IPPE
)

image_pts_cam1 = np.load("Calibration matrices/cam1_target_image_pts.npy").astype(np.float32)

_, rvec1, tvec1 = cv2.solvePnP(
    world_pts_target,
    image_pts_cam1,
    K1,
    D1,
    flags=cv2.SOLVEPNP_IPPE
)


def projection_matrix(K, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    return K @ np.hstack((R, tvec))


P1 = projection_matrix(K1, rvec1, tvec1)
P2 = projection_matrix(K2_post, rvec2, tvec2)


(pt1, _), (pt2, _) = detect_two_images(
   model_path="runs/detect/train/weights/best.pt",
   img1_path="Location Tracking/Test_images/frame_07610.jpg",
   img2_path="Location Tracking/Test_images/frame_07743.jpg",
   out1_path="detected_cam1.jpg",
   out2_path="detected_cam2.jpg"
)

print("Cam1 pixel center:", pt1)
print("Cam2 pixel center:", pt2)

def triangulate_point(P1, P2, pt1, pt2):
    pt1 = np.array(pt1).reshape(2, 1).astype(float)
    pt2 = np.array(pt2).reshape(2, 1).astype(float)

    X_h = cv2.triangulatePoints(P1, P2, pt1, pt2)
    X = X_h[:3] / X_h[3]
    return X.flatten()

# X = triangulate_point(P1, P2, pt1, pt2)
# print("Triangulated 3D position (meters):", X)

if __name__ == "__main__":
    (pt1, _), (pt2, _) = detect_two_images(...)
    X = triangulate_point(P1, P2, pt1, pt2)
    print("Triangulated 3D position:", X)