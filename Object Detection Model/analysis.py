# Youssef Ibrahim - go52xac - 03804537

# pyright: reportPrivateImportUsage=false
from ultralytics import YOLO

# Load the trained models (this loads the BEST weights saved after training the models)
model_n = YOLO("runs/detect/train/weights/best.pt")
model_s = YOLO("runs/detect/train2/weights/best.pt")

# Validate the YOLOv8 models on val dataset (YOLO compares predictions with ground-truth labels and computes performance metrics)
model_n.val()
model_s.val()
