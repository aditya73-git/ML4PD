# Youssef Ibrahim - go52xac - 03804537

# pyright: reportPrivateImportUsage=false
from ultralytics import YOLO

# Load the YOLOv8 NANO model
model_n = YOLO("yolov8n.pt")

# Train the model on the dataset
model_n.train(
    data="C:/Users/Youssef Sabry/Desktop/MLPD/Project/ML4PD/Object Detection Model/data.yaml",
    epochs=50, # 50 full passes through the entire training dataset
    imgsz=640 # input image size or resolution
)
