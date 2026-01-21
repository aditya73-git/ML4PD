# Physics-Informed Neural Network for Beer Pong Trajectory Prediction

## Author: Sri Aditya Yaddanapudi (03812104)

## Achieved Results: 1.56 cm Average Error with Only 20 Training Examples!

This project demonstrates a **Physics-Informed Neural Network (PINN)** that predicts beer pong ball landing positions with **1.56 cm average accuracy**, representing a **93.3% improvement** over pure physics models while using only **20 training examples**.

### Key Achievements
- **Average Error**: 1.56 cm 
- **Success Rate**: 100% (all 10 test throws < 3.5 cm target)
- **Best Prediction**: 0.34 cm (near perfect!)
- **Data Efficiency**: Only 20 training throws required
- **Real-time Ready**: Predictions in milliseconds

## Project Structure
physics_informed_ml_project/
│
├── data/ # Raw and processed data files
│ ├── beer_pong_velocity_output.csv # 3D trajectory measurements at 60Hz
│ ├── impact_log.csv # Ground truth landing positions
│ └── Physics_Errors.csv # Physics-only model errors for comparison
│
├── models/ # Neural network and physics components
│ ├── init.py
│ ├── pinn_model.py # SimplePINN architecture (64→32 neurons)
│ ├── physics_loss.py # Physics constraints with loss scaling
│ └── data_loader.py # Data preprocessing and splitting
│
├── training/ # Training utilities
│ ├── init.py
│ ├── trainer.py # Training loop with validation
│ 
│
├── prediction/ # Inference and prediction
│ ├── init.py
│ └── predictor.py # Smart prediction with pattern adjustments
│
├── visualization/ # Plotting and analysis
│ ├── init.py
│ └── plots.py # All required visualization plots
│
├── main.py # Main execution script
├── config.yaml # Comprehensive configuration (see below)
├── requirements.txt # Python dependencies
└── README.md # This documentation file

## Configuration File (`config.yaml`)

The project is fully configured through `config.yaml` which contains all Comprehensive configuration about project.

## final_validation_results has results of the project.