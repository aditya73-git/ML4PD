# Object Detection Model
## Author: Youssef Ibrahim - go52xac - 03804537

### Structure:

- `train/images/`: Contains training images used to train the object detection models.

- `train/labels/`: Contains YOLO-format label files for training images.

- `val/images/`: Contains validation images used to evaluate the model during training.

- `val/labels/`: Contains ground-truth labels for validation images.

- `runs/`: This directory is automatically created by YOLO during training and inference.

- `runs/detect/train/`: Results from training YOLOv8n

- `runs/detect/train2/`: Results from training YOLOv8s

- `runs/detect/train3/`: Results from training YOLOv8s with stricter configurations.

- `data.yaml`: Defines the dataset configuration for YOLO.

- `model1.py`: Trains YOLOv8n (Nano) model

- `model2.py`: Trains YOLOv8s (Small) model

- `analysis.py`: Used to validate and compare trained models.

- `test.py`: Used for testing the trained model on new images and prints out normalized YOLO xywh for both images and saves the images with bounding boxes drawn under `test images`.

- `test2.py`: Created to modify `test.py` to be able to integrate with the Location Tracking.

### Important Notes:
- To run you need to have openCV installed:
```
pip install opencv-python
```
- To train the models run `model1.py` for yolov8n and run `model2.py`for yolov8s.
- Run `test.py` for verifying the object detection model for yolov8n and solution of task (d).
- The labelled training data and validation data are removed from the repository for max. storage submission purposes.
- To be able to run the object detection model, kindly download the folder named `train` and the folder named `val` from the **group 15 Sharepoint drive** and place both under `Object Detection Model` folder so that the path is as follows:
```
Object Detection Model/train
```
and
```
Object Detection Model/val
```

# Location Tracking
## Author: Lina Ghonim - go54sen - 03821777

### Important Notes:
- The synced videos are removed from the repository for max. storage submission purposes.
- To be able to run the `linear_regression.py` file under Location Tracking folder, kindly download the folder named `syncronized_videos` from the group 15 Sharepoint drive and place it under `Location Tracking` folder so that the path is as follows:
```
Location Tracking/syncronized_videos
```
- Only **run** `linear_regression.py` because `tracking_code_trial2.py` is just a helper file.

- To get the error plots run `error_calculation.py`.

- To get the calibration matrices, run `Calibration.py`, `Calibration2.py`, `Camera1_homogrophy.py`, `Camera2_homogrophy.py` and  `Camera1_homogrophy_prezoom.py`.

# Physics Based Modelling
## Author: Balazs Horvath - ge78yoj - 03741091

- `landing_point_predictor.py`: Main script

- `landing_point_plotter.py`: Definition for the plotting functions

### Important Note:
- The generated plots are removed from the repository for max. storage submission purposes. They can be restored by running `landing_point_predictor.py`

# ML based prediction
## Authour: Venkata Aditya Sai Prasanna Kotte - go56xur - 03786342

- `ML_models.py` : Main script that trains 2 models and validates the results

- `Plots.py` : Helper module that plots the validated results 

# Useful Links

- [Link to the presentation (Google Slides)](https://docs.google.com/presentation/d/1DE0DZpC3-iIcxX1wmQhgHlwFM3O0JKoiveR2z3y3ClQ/edit?usp=sharing)

- [Link to the git repository](https://github.com/aditya73-git/ML4PD)
