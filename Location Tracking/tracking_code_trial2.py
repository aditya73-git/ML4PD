
import glob
import numpy as np
import cv2
from Object_Detection_Model.test2 import detect_two_images

CHECKERBOARD = (10, 7)       # inner corners
SQUARE_SIZE = 0.015         # meters


# Camera 1 (already valid)
cam1 = np.load(
    "C:/Users/LEGION7/OneDrive/Desktop/PERSONAL/TUM/Semester 1/Machine Learning for Product Development/ML4PD/cam1_calib.npz")
K1 = cam1["K"]
D1 = cam1["D"]

# Camera 2 pre-zoom
cam2 = np.load(
    "C:/Users/LEGION7/OneDrive/Desktop/PERSONAL/TUM/Semester 1/Machine Learning for Product Development/ML4PD/cam2_calib.npz")
K2_pre = cam2["K"]
D2 = cam2["D"]

# Homographies
H2_pre = np.load(
    "C:/Users/LEGION7/OneDrive/Desktop/PERSONAL/TUM/Semester 1/Machine Learning for Product Development/ML4PD/cam2_homography_prezoom.npy")
H2_post = np.load(
    "C:/Users/LEGION7/OneDrive/Desktop/PERSONAL/TUM/Semester 1/Machine Learning for Product Development/ML4PD/cam2_homography.npy")

H_tilde = np.linalg.inv(K2_pre) @ H2_post @ np.linalg.inv(H2_pre)

s1 = np.linalg.norm(H_tilde[:, 0])
s2 = np.linalg.norm(H_tilde[:, 1])
scale = (s1 + s2) / 2

print("Estimated focal scale:", scale)

K2_post = K2_pre.copy()
K2_post[0, 0] *= scale   # fx
K2_post[1, 1] *= scale   # fy

print("Camera 2 post-zoom intrinsics:\n", K2_post)

world_pts_target = np.array([
    [-0.3, -0.3, 0.0],
    [0.3, -0.3, 0.0],
    [0.3,  0.3, 0.0],
    [-0.3,  0.3, 0.0]
], dtype=np.float32)

image_pts_cam2 = np.load(
    "cam2_target_image_pts_postzoom.npy").astype(np.float32)

_, rvec2, tvec2 = cv2.solvePnP(
    world_pts_target,
    image_pts_cam2,
    K2_post,
    D2,
    flags=cv2.SOLVEPNP_IPPE
)

image_pts_cam1 = np.load("cam1_target_image_pts.npy").astype(np.float32)

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

detections_cam1 = {frame_id: (u1, v1)}
detections_cam2 = {frame_id: (u2, v2)}


def triangulate_point(P1, P2, pt1, pt2):
    pt1 = np.array(pt1).reshape(2, 1).astype(float)
    pt2 = np.array(pt2).reshape(2, 1).astype(float)

    X_h = cv2.triangulatePoints(P1, P2, pt1, pt2)
    X = X_h[:3] / X_h[3]
    return X.flatten()


trajectory = []

for frame_id in sorted(detections_cam1.keys()):
    if frame_id not in detections_cam2:
        continue

    pt1 = detections_cam1[frame_id]
    pt2 = detections_cam2[frame_id]

    X = triangulate_point(P1, P2, pt1, pt2)
    trajectory.append(X)

trajectory = np.array(trajectory)

trajectory[:, 0]  # X (meters)
trajectory[:, 1]  # Y (meters)
trajectory[:, 2]  # Z (meters)
