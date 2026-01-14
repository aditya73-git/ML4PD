# pyright: reportPrivateImportUsage=false
from ultralytics import YOLO

# Load the trained models
model_n = YOLO("runs/detect/train/weights/best.pt")
model_s = YOLO("runs/detect/train2/weights/best.pt")

model_n.val()
model_s.val()
