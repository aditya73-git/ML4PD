import cv2
import numpy as np

def get_ball_coords(frame, relaxed=False):
    # Expanded HSV for the "blurred" look in the top view
    lower = np.array([5, 100, 100]) if relaxed else np.array([5, 150, 100])
    upper = np.array([30, 255, 255]) 

    gpu_frame = cv2.UMat(frame)
    blurred = cv2.GaussianBlur(gpu_frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    
    cnts, _ = cv2.findContours(mask.get(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if peri == 0: return None
        
        circularity = 4 * np.pi * (area / (peri * peri))
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if relaxed:
            # RELAXED FILTERS for the blurred impact streak
            if circularity > 0.1 and 5 < radius < 100: 
                return (int(x), int(y))
        else:
            # STRICT FILTERS for normal tracking
            if 0.6 < circularity < 1.2 and 8 < radius < 40:
                return (int(x), int(y))
    return None

def get_ball_coords_bottom(frame, relaxed=True):
    # Expanded HSV for the "blurred" look in the top view
    lower = np.array([5, 100, 100]) if relaxed else np.array([5, 150, 100])
    upper = np.array([30, 255, 255]) 

    gpu_frame = cv2.UMat(frame)
    blurred = cv2.GaussianBlur(gpu_frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 20:
            x, y, w, h = cv2.boundingRect(cnt)
            # Return center-x and bottom-y
            return (x + w//2, y + h) 
    return None

# ---- SELF TEST SECTION ----
if __name__ == "__main__":
    start_frame = 11743
    #cap_top.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_side = cv2.VideoCapture(r"Recording/Setup 1/SYNC_IMG_6567_c1_t1.MOV")
    cap_side.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    #cap_side = cv2.VideoCapture(r"Recording/Setup 1/SYNC_IMG_0349_c2_t1.MOV")
    while True:
        ret, frame = cap_side.read()
        if not ret:
            break

        coords = get_ball_coords(frame)
        if coords:
            x, y = coords
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
            print("Ball at:", coords)

        cv2.imshow("Tracking Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap_side.release()
    cv2.destroyAllWindows()
