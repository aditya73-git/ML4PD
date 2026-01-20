"""
DATA LOADING AND PREPROCESSING MODULE
This module handles data loading, cleaning, and preparation for the Physics-Informed
Neural Network (PINN) training pipeline. It converts raw sensor data into properly
formatted training and validation sets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(velocity_path, impact_path):
    """
    Loads and preprocesses raw trajectory and impact point data from CSV files.
    
    This function performs three main tasks:
    1. Reads velocity trajectory data (time-series positions and velocities)
    2. Reads impact/landing point data (final ball positions)
    3. Handles unit conversion (cm to meters) for consistency with physics equations
    
    Parameters:
    -----------
    velocity_path : str
        Path to CSV file containing velocity/position time-series data
    impact_path : str
        Path to CSV file containing ball impact/landing positions
    
    Returns:
    --------
    tuple of (pd.DataFrame, pd.DataFrame)
        Returns (velocity_data, impact_data) DataFrames ready for further processing
    """
    
    # Load raw data from CSV files
    vel_data = pd.read_csv(velocity_path)
    impact_data = pd.read_csv(impact_path)
    
    # Debug output to verify column names in impact data
    print(f"Impact data columns: {impact_data.columns.tolist()}")
    
    # Create throw identifier for impact data
    # Subtracting 20 aligns with velocity data throw IDs
    impact_data['throw_id'] = impact_data['ID'] - 20
    
    # Check for different possible column naming conventions and handle unit conversion
    # Physics-informed models require SI units (meters) for consistent physics calculations
    if 'X_m' in impact_data.columns and 'Y_m' in impact_data.columns:
        print("✓ Using X_m and Y_m columns (meters)")
        # Data already in meters - use directly
        impact_data['actual_x_m'] = impact_data['X_m']
        impact_data['actual_y_m'] = impact_data['Y_m']
    elif 'X_cm' in impact_data.columns and 'Y_cm' in impact_data.columns:
        print("✓ Using X_cm and Y_cm columns (converting to meters)")
        # Convert centimeters to meters (1 cm = 0.01 m)
        impact_data['actual_x_m'] = impact_data['X_cm'] / 100.0
        impact_data['actual_y_m'] = impact_data['Y_cm'] / 100.0
    else:
        raise ValueError("No X/Y columns found in impact data!")
    
    return vel_data, impact_data


def prepare_data_splits(vel_data, impact_data):
    """
    Creates training, validation, and test splits from the velocity data.
    
    The function implements a time-based split strategy:
    - Training: First 16 throws (throws 1-16)
    - Validation: Next 4 throws (throws 17-20) for hyperparameter tuning
    - Testing: Final 10 throws (throws 21-30) for final evaluation
    
    Each sample consists of current state → next state pairs for trajectory learning.
    
    Parameters:
    -----------
    vel_data : pd.DataFrame
        DataFrame containing time-series velocity and position data
    impact_data : pd.DataFrame
        DataFrame containing impact/landing positions (used for reference only)
    
    Returns:
    --------
    tuple of (X_train, y_train, X_val, y_val)
        Training and validation sets as numpy arrays
    """
    
    print("\nPreparing data splits:")
    print("Training: Throws 1-16")
    print("Validation: Throws 17-20") 
    print("Testing: Throws 21-30")
    
    # Initialize empty lists to store samples
    X_train, y_train, X_val, y_val = [], [], [], []
    
    # Process each throw to create state transition samples
    for throw_id in range(1, 21):
        # Filter data for current throw and sort by time
        throw_data = vel_data[vel_data['throw_id'] == throw_id].sort_values('t')
        
        # Skip throws with insufficient data points
        if len(throw_data) < 2:
            continue
            
        # Create consecutive state pairs for trajectory learning
        # For each time step, predict the next position from current state
        for i in range(len(throw_data) - 1):
            current = throw_data.iloc[i]  # Current state
            next_row = throw_data.iloc[i + 1]  # Next state (target)
            
            # Input features: position + velocity (6-dimensional state vector)
            sample = [
                current['x_raw'], current['y_raw'], current['z_raw'],  # Current position
                current['vel_x_measured'], current['vel_y_measured'], current['vel_z_measured']  # Current velocity
            ]
            
            # Target: next position (3-dimensional)
            target = [next_row['x_raw'], next_row['y_raw'], next_row['z_raw']]
            
            # Split into training (throws 1-16) and validation (throws 17-20)
            if throw_id <= 16:
                X_train.append(sample)
                y_train.append(target)
            else:  # throws 17-20
                X_val.append(sample)
                y_val.append(target)
    
    # Convert lists to numpy arrays for efficient tensor operations
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    return X_train, y_train, X_val, y_val


def scale_data(X_train, y_train, X_val, y_val):
    """
    Applies standardization to input features and target variables.
    
    Standardization (z-score normalization) is critical for neural network training:
    1. Centers data around zero mean
    2. Scales to unit variance
    3. Improves training stability and convergence speed
    
    Important: The validation data is transformed using the training statistics
    to prevent data leakage from validation/test sets into training.
    
    Parameters:
    -----------
    X_train, y_train : np.array
        Training features and targets
    X_val, y_val : np.array
        Validation features and targets
    
    Returns:
    --------
    tuple of scaled arrays and scaler objects
        Returns (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, scaler_X, scaler_y)
        The scaler objects are included for inverse transformations during prediction
    """
    
    # Create separate scalers for features (X) and targets (y)
    scaler_X = StandardScaler()  # Scaler for input features (6D state vector)
    scaler_y = StandardScaler()  # Scaler for output targets (3D next position)
    
    # Fit scalers on training data only, then transform both training and validation
    X_train_scaled = scaler_X.fit_transform(X_train)  # Fit and transform training features
    y_train_scaled = scaler_y.fit_transform(y_train)  # Fit and transform training targets
    
    X_val_scaled = scaler_X.transform(X_val)  # Transform validation features (using training statistics)
    y_val_scaled = scaler_y.transform(y_val)  # Transform validation targets (using training statistics)
    
    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, scaler_X, scaler_y