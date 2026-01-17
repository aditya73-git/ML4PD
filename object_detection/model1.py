# pyright: reportPrivateImportUsage=false
from ultralytics import YOLO

# Load the small nano model
model_n = YOLO("yolov8n.pt")

# Train
model_n.train(
    data="C:/Users/Youssef Sabry/Desktop/MLPD/Project/ML4PD/Object Detection Model/data.yaml",
    epochs=50,
    imgsz=640
)
