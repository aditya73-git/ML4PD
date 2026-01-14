# pyright: reportPrivateImportUsage=false
import cv2
from ultralytics import YOLO
import os

def detect_two_images(model_path, img1_path, img2_path,
                      out1_path="output1.jpg", out2_path="output2.jpg"):
    """
    Detect a ping-pong ball in two images using YOLOv8.
    Returns normalized YOLO xywh for both images
    and saves the images with bounding boxes drawn.
    """

    # Load model once
    model = YOLO(model_path)

    # Helper to process a single image
    def process_single_image(image_path, output_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        H, W = img.shape[:2]

        # Run YOLO detection
        results = model(img)[0]

        if len(results.boxes) == 0:
            print(f"No object detected in {image_path}")
            return None, None

        # Get YOLO normalized bounding box (cx, cy, w, h) in 0–1 range
        cx, cy, w, h = results.boxes.xywhn[0].cpu().numpy()

        # Convert normalized → pixel coordinates for drawing
        cx_pix = int(cx * W)
        cy_pix = int(cy * H)
        w_pix = int(w * W)
        h_pix = int(h * H)

        x1 = int(cx_pix - w_pix / 2)
        y1 = int(cy_pix - h_pix / 2)
        x2 = int(cx_pix + w_pix / 2)
        y2 = int(cy_pix + h_pix / 2)

        # Draw bounding box
        drawn = img.copy()
        cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(drawn, (cx_pix, cy_pix), 4, (0, 0, 255), -1)

        # Save
        cv2.imwrite(output_path, drawn)

        return (cx, cy, w, h), output_path

    # Process both images
    bbox1, saved1 = process_single_image(img1_path, out1_path)
    bbox2, saved2 = process_single_image(img2_path, out2_path)

    return (bbox1, saved1), (bbox2, saved2)

result1, result2 = detect_two_images(
    model_path="runs/detect/train/weights/best.pt",
    img1_path="C:/Users/Youssef Sabry/Desktop/MLPD/Project/ML4PD/Object Detection Model/test images/frame_08093.jpg",
    img2_path="C:/Users/Youssef Sabry/Desktop/MLPD/Project/ML4PD/Object Detection Model/test images/frame_08101.jpg",
    out1_path="C:/Users/Youssef Sabry/Desktop/MLPD/Project/ML4PD/Object Detection Model/test images/detected_test1.jpg",
    out2_path="C:/Users/Youssef Sabry/Desktop/MLPD/Project/ML4PD/Object Detection Model/test images/detected_test2.jpg"
)

print("Image 1 normalized bbox:", result1[0])
print("Saved image 1:", result1[1])

print("Image 2 normalized bbox:", result2[0])
print("Saved image 2:", result2[1])
