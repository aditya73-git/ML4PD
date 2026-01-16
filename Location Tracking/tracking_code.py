import numpy as np
import cv2
import matplotlib.pyplot as plt

# Camera 1 calibration
cam1 = np.load(
    "C:/Users/LEGION7/OneDrive/Desktop/PERSONAL/TUM/Semester 1/Machine Learning for Product Development/ML4PD/cam1_calib.npz")
mtx1 = cam1["mtx"]
dist1 = cam1["dist"]

# Camera 2 homography
H2 = np.load("C:/Users/LEGION7/OneDrive/Desktop/PERSONAL/TUM/Semester 1/Machine Learning for Product Development/ML4PD/cam2_homography.npy")


def cam2_pixel_to_world(u, v, H):
    p = np.array([u, v, 1.0])
    Pw = H @ p
    Pw /= Pw[2]
    return Pw[0], Pw[1]


def pixel_to_ray(u, v, mtx, dist):
    pts = np.array([[[u, v]]], dtype=np.float32)
    undist = cv2.undistortPoints(pts, mtx, dist)
    x, y = undist[0, 0]
    ray = np.array([x, y, 1.0])
    return ray / np.linalg.norm(ray)


def estimate_xyz_single_frame(u1, v1, u2, v2, mtx, dist, H):
    # X,Y from camera 2 (world frame)
    X, Y = cam2_pixel_to_world(u2, v2, H)

    # Ray from camera 1
    ray = pixel_to_ray(u1, v1, mtx, dist)

    # Solve for lambda (least-squares)
    lambdas = []
    if abs(ray[0]) > 1e-6:
        lambdas.append(X / ray[0])
    if abs(ray[1]) > 1e-6:
        lambdas.append(Y / ray[1])

    lam = np.mean(lambdas)

    Z = lam * ray[2]

    return X, Y, Z


# Example inputs (replace with real detector outputs)
u1, v1 = 640, 360
u2, v2 = 512, 420

X, Y, Z = estimate_xyz_single_frame(
    u1, v1,
    u2, v2,
    mtx1, dist1,
    H2
)

print(f"Estimated 3D position:")
print(f"X = {X:.2f} cm")
print(f"Y = {Y:.2f} cm")
print(f"Z = {Z:.2f} cm")
