# Youssef Ibrahim - go52xac - 03804537

# pyright: reportPrivateImportUsage=false
from ultralytics import YOLO

# Load the pre-trained YOLOv8 SMALL model
model_s = YOLO("yolov8s.pt")

# Train the model on the dataset
model_s.train(
    data="C:/Users/Youssef Sabry/Desktop/MLPD/Project/ML4PD/Object Detection Model/data.yaml",
    epochs=50, # 50 full passes through the entire training dataset
    imgsz=640, # input image size or resolution
    # Reducing image augmentation strength (because maybe if augmentations are too strong the ping-pong ball
    # can become distorted or the bounding box becomes inaccurate and training becomes unstable)
    hsv_h=0.005,
    hsv_s=0.3,
    hsv_v=0.3,
    fliplr=0.3,
    flipud=0.0,
    mosaic=0.5,
    # Using a smaller batch size on CPU (because if the no. of images processed at a time is smaller training becomes more stable)
    batch=4
)

