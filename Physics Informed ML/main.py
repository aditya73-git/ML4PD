"""
MAIN EXECUTION SCRIPT
Orchestrates the complete Physics-Informed Neural Network pipeline:
1. Data loading and preprocessing
2. Model training with physics constraints
3. Testing on unseen throws
4. Performance evaluation and visualization
"""

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import joblib

# Import custom modules
from models.data_loader import load_data, prepare_data_splits, scale_data
from models.pinn_model import SimplePINN
from models.physics_loss import physics_loss_original, scaled_physics_loss
from training.trainer import train_with_validation
from prediction.predictor import smart_predict, test_predictions
from visualization.plots import create_visualizations, plot_loss_history

def main():
    """Main execution function - runs the complete PINN pipeline"""
    
    # Header display
    print("=" * 80)
    print("FINAL OPTIMIZED PINN WITH VALIDATION AND SCALED LOSS")
    print("=" * 80)
    
    # Create directory for saving results
    os.makedirs('Physics Informed ML/final_validation_results', exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # --- STEP 1: DATA LOADING & PREPROCESSING ---
    print("Loading data...")
    # Load velocity trajectory data and impact point data
    vel_data, impact_data = load_data(
        "Physics Informed ML/data/beer_pong_velocity_output.csv",
        "Physics Informed ML/data/impact_log.csv"
    )
    
    # Split data: Training (throws 1-16), Validation (17-20), Test (21-30)
    X_train, y_train, X_val, y_val = prepare_data_splits(vel_data, impact_data)
    
    # Scale/normalize data for neural network training
    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, scaler_X, scaler_y = scale_data(
        X_train, y_train, X_val, y_val
    )
    
    # --- STEP 2: MODEL CREATION ---
    print("\nBuilding model (64 → 32 neurons with LeakyReLU)...")
    pinn = SimplePINN()  # Simple neural network architecture
    
    # --- STEP 3: TRAINING WITH PHYSICS CONSTRAINTS ---
    print("\nTraining model (150 epochs)...")
    # Train with custom loss combining physics and data
    train_losses, val_losses, train_scaled_losses, val_scaled_losses = train_with_validation(
        pinn, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
        physics_loss_original, scaled_physics_loss, epochs=150
    )
    
    # Display training results
    print(f"\nTraining complete!")
    print(f"Final Training Loss (scaled): {train_scaled_losses[-1]:.4f}")
    print(f"Final Validation Loss (scaled): {val_scaled_losses[-1]:.4f}")
    print(f"Best Validation Loss (scaled): {min(val_scaled_losses):.4f}")
    
    # Explain the loss scaling system used
    print("\n" + "="*80)
    print("LOSS SCALING EXPLANATION:")
    print("="*80)
    print("For display purposes, we scale the loss components:")
    print("- Gravity/Horizontal acceleration terms: ÷10,000 (from m²/s⁴ to readable scale)")
    print("- Negative coordinate terms: ÷100 (from m² to readable scale)")
    print("- Data loss (MSE): Already small (normalized data)")
    print("\nThis scaling makes the loss values easier to read and compare.")
    print("The actual training uses the original (unscaled) physics constraints.")
    print("="*80)
    
    # --- STEP 4: TESTING ON UNSEEN THROWS (21-30) ---
    print("\n" + "=" * 80)
    print("TESTING ON THROWS 21-30")
    print("=" * 80)
    
    # Test the trained model on completely unseen data
    results = test_predictions(pinn, vel_data, impact_data, scaler_X, scaler_y, smart_predict)
    
    # --- STEP 5: RESULTS ANALYSIS ---
    if results:
        # Convert results to DataFrame for analysis
        df = pd.DataFrame(results)
        avg_error = df['error_cm'].mean()
        std_error = df['error_cm'].std()
        
        # Display comprehensive test results
        print("\n" + "=" * 80)
        print("FINAL TEST RESULTS")
        print("=" * 80)
        print(f"Average Error: {avg_error:.2f} cm")
        print(f"Standard Deviation: {std_error:.2f} cm")
        print(f"Best Error: {df['error_cm'].min():.2f} cm")
        print(f"Worst Error: {df['error_cm'].max():.2f} cm")
        print(f"\nTraining Loss (scaled): {train_scaled_losses[-1]:.4f}")
        print(f"Validation Loss (scaled): {val_scaled_losses[-1]:.4f}")
        
        # Categorize performance by error thresholds
        excellent = len(df[df['error_cm'] < 2.0])
        good = len(df[(df['error_cm'] >= 2.0) & (df['error_cm'] < 3.5)])
        poor = len(df[df['error_cm'] >= 3.5])
        
        print(f"\nPerformance Categories:")
        print(f"★ Excellent (<2.0 cm):  {excellent}")
        print(f"✓ Good (2.0-3.5 cm):    {good}")
        print(f"✗ Poor (≥3.5 cm):       {poor}")
        
        # Calculate success rate (throws with error < 3.5 cm)
        success_rate = (excellent + good) / len(df) * 100
        print(f"\nSuccess Rate (<3.5 cm): {success_rate:.1f}%")
        
        # --- STEP 6: SAVE RESULTS & ARTIFACTS ---
        # Save predictions to CSV
        df.to_csv('Physics Informed ML/final_validation_results/predictions.csv', index=False)
        
        # Save loss history for analysis
        loss_df = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss_raw': train_losses,
            'val_loss_raw': val_losses,
            'train_loss_scaled': train_scaled_losses,
            'val_loss_scaled': val_scaled_losses
        })
        loss_df.to_csv('Physics Informed ML/final_validation_results/loss_history.csv', index=False)
        
        # Save scalers for future predictions
        joblib.dump(scaler_X, 'Physics Informed ML/final_validation_results/scaler_X.pkl')
        joblib.dump(scaler_y, 'Physics Informed ML/final_validation_results/scaler_y.pkl')
        
        # --- STEP 7: CREATE VISUALIZATIONS ---
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)
        
        # 1. Plot training/validation loss curves
        plot_loss_history(train_scaled_losses, val_scaled_losses, 'Physics Informed ML/final_validation_results')
        
        # 2. Create scatter plots comparing predictions to actuals
        create_visualizations(results, "Physics Informed ML/data/Physics_Errors.csv", 'Physics Informed ML/final_validation_results')
        
        print(f"\n All results saved to 'final_validation_results/' directory")
        
        # --- STEP 8: FINAL VERDICT ---
        print("\n" + "=" * 80)
        if avg_error <= 3.5:
            print(" TARGET ACHIEVED! Average error ≤ 3.5 cm")
            if avg_error <= 2.5:
                print(" EXCELLENT! Average error < 2.5 cm")
        else:
            print(f" Average error: {avg_error:.2f} cm (target: ≤3.5 cm)")
        
        # Compare with pure physics model baseline
        print("\n" + "=" * 80)
        print("COMPARISON WITH PURE PHYSICS MODEL:")
        print("=" * 80)
        print("Pure Physics Model Average Error: ~23.4 cm")
        print(f"Our PINN Average Error: {avg_error:.1f} cm")
        improvement = ((23.4 - avg_error) / 23.4 * 100) if avg_error < 23.4 else 0
        print(f"Improvement: {improvement:.1f}% better!")
        print("=" * 80)
    else:
        print("\n No predictions were made")

# Entry point for script execution
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()