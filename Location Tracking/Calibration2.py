import cv2
import numpy as np
import glob
import os

# Updated to match your image
CHECKERBOARD = (10, 7)
SQUARE_SIZE_CM = 1.5
IMAGE_DIR = "C:/Users/LEGION7/OneDrive/Desktop/PERSONAL/TUM/Semester 1/Machine Learning for Product Development/ML4PD/Location Tracking/Checkerboard_frames_cam2"
IMAGE_EXT = "*.jpg"
STEP = 5

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_CM

objpoints = []
imgpoints = []

images = glob.glob(os.path.join(IMAGE_DIR, IMAGE_EXT))
print(f"Checking {len(images)} images with pattern {CHECKERBOARD}...")

for i, fname in enumerate(images[::STEP]):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- PRE-PROCESSING FOR TABLET SCREENS ---
    # Increase contrast to make corners sharper against screen glow
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Use SB detector - it is much better for screens
    ret, corners = cv2.findChessboardCornersSB(
        gray, CHECKERBOARD, cv2.CALIB_CB_EXHAUSTIVE)

    if ret:
        print(f"✅ SUCCESS: {os.path.basename(fname)}")
        objpoints.append(objp)
        imgpoints.append(corners)

        if len(imgpoints) >= 50:
            print("Reached 50 usable frames. Stopping detection to begin calibration...")
            break
        # VISUAL CHECK: See the corners detected
        # cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        # cv2.imshow('Detection', cv2.resize(img, (960, 540)))
        # cv2.waitKey(100)
    else:
        print(f"❌ FAILED: {os.path.basename(fname)}")

if len(imgpoints) >= 10:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    print("\nCalibration successful! Matrix saved.")
    np.savez("cam2_calib.npz", mtx=mtx, dist=dist)
else:
    print(f"\nOnly found {len(imgpoints)} usable images. Need at least 10.")

# Calculate Reprojection Error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"Total Reprojection Error: {mean_error/len(objpoints):.4f} pixels")
