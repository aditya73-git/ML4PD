import cv2
import numpy as np

# Target paper dimensions (cm)
TARGET_SIZE = 0.6

world_pts = np.array([
    [-0.3, -0.3],   # bottom-left
    [0.3, -0.3],   # bottom-right
    [0.3,  0.3],   # top-right
    [-0.3,  0.3]    # top-left
], dtype=np.float32)

image_pts = []


def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        image_pts.append([x, y])
        print(f"Clicked: {x}, {y}")


img = cv2.imread("C:/Users/LEGION7/OneDrive/Desktop/PERSONAL/TUM/Semester 1/Machine Learning for Product Development/ML4PD/Location Tracking/TargetFrame_cam1/frame_01865.jpg")
if img is None:
    raise ValueError("Image not found!")

cv2.namedWindow("Click target corners BL → BR → TR → TL", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Click target corners BL → BR → TR → TL", 960, 540)

cv2.imshow("Click target corners BL → BR → TR → TL", img)
cv2.setMouseCallback("Click target corners BL → BR → TR → TL", mouse_cb)


while len(image_pts) < 4:
    cv2.waitKey(1)

cv2.destroyAllWindows()

image_pts = np.array(image_pts, dtype=np.float32)
np.save("cam1_target_image_pts.npy", image_pts)

H2, _ = cv2.findHomography(image_pts, world_pts)
np.save("cam1_homography.npy", H2)
