# pyright: reportPrivateImportUsage=false
from ultralytics import YOLO

model_s = YOLO("yolov8s.pt")

model_s.train(
    data="C:/Users/Youssef Sabry/Desktop/MLPD/Project/ML4PD/Object Detection Model/data.yaml",
    epochs=50,
    imgsz=640,
    hsv_h=0.005,
    hsv_s=0.3,
    hsv_v=0.3,
    fliplr=0.3,
    flipud=0.0,
    mosaic=0.5,
    batch=4
)

