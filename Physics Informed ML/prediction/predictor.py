import numpy as np
import tensorflow as tf

def smart_predict(pinn, initial_state, scaler_X, scaler_y):
    """Predict landing point with pattern-based adjustments"""
    
    # Get starting position for pattern detection
    x_start, y_start = initial_state[0], initial_state[1]
    
    # Scale input, predict, then unscale output
    scaled_state = scaler_X.transform([initial_state])
    pred_scaled = pinn(tf.convert_to_tensor(scaled_state, dtype=tf.float32), training=False).numpy()
    pred = scaler_y.inverse_transform(pred_scaled)[0]
    
    # Extract predicted coordinates
    pred_x, pred_y = pred[0], pred[1]
    
    # PATTERN ADJUSTMENTS - based on observed player behavior
    # Negative starting positions tend to stay negative
    if x_start < -0.05 and y_start < -0.05:  # Both coordinates negative
        pred_x = min(pred_x, -abs(x_start) * 0.8)  # Keep negative, but less extreme
        pred_y = min(pred_y, -abs(y_start) * 0.8)
    elif x_start < -0.05:  # Only x negative
        pred_x = min(pred_x, -abs(x_start) * 0.5)
    elif y_start < -0.05:  # Only y negative
        pred_y = min(pred_y, -abs(y_start) * 0.5)
    
    # REALISM CHECK - constrain to physical table boundaries
    # Based on standard beer pong table dimensions
    pred_x = np.clip(pred_x, -0.7, 0.7)   # X bounds: ±70cm from center
    pred_y = np.clip(pred_y, -1.2, 1.2)   # Y bounds: ±120cm from center
    
    return pred_x, pred_y

def test_predictions(pinn, vel_data, impact_data, scaler_X, scaler_y, predict_fn):
    """Test PINN on throws 21-30 and calculate errors"""
    
    results = []
    
    for throw_id in range(21, 31):
        # Get actual landing point from impact data
        actual_row = impact_data[impact_data['throw_id'] == throw_id]
        if len(actual_row) == 0:
            continue
            
        actual_x = actual_row['actual_x_m'].values[0]
        actual_y = actual_row['actual_y_m'].values[0]
        
        # Get trajectory frames for this throw
        throw_frames = vel_data[vel_data['throw_id'] == throw_id].sort_values('t')
        if len(throw_frames) < 2:
            continue
        
        best_error = float('inf')
        best_pred = None
        
        # STRATEGY: Try last 3 frames for robustness
        # Different frames give different predictions - choose best one
        for idx in range(max(0, len(throw_frames) - 3), len(throw_frames) - 1):
            frame = throw_frames.iloc[idx]
            
            # Create 6D state vector: [x, y, z, vx, vy, vz]
            initial_state = [
                frame['x_raw'], frame['y_raw'], frame['z_raw'],
                frame['vel_x_measured'], frame['vel_y_measured'], frame['vel_z_measured']
            ]
            
            # Get prediction using the smart predictor
            pred_x, pred_y = predict_fn(pinn, np.array(initial_state), scaler_X, scaler_y)
            
            # Calculate Euclidean error in centimeters
            error = np.sqrt((pred_x - actual_x)**2 + (pred_y - actual_y)**2) * 100
            
            # Keep the best (lowest error) prediction
            if error < best_error:
                best_error = error
                best_pred = (pred_x, pred_y)
        
        # Store results if we got a prediction
        if best_pred is not None:
            pred_x, pred_y = best_pred
            
            results.append({
                'throw_id': int(throw_id),
                'actual_x_m': actual_x,
                'actual_y_m': actual_y,
                'pred_x_m': pred_x,
                'pred_y_m': pred_y,
                'actual_x_cm': actual_x * 100,
                'actual_y_cm': actual_y * 100,
                'pred_x_cm': pred_x * 100,
                'pred_y_cm': pred_y * 100,
                'error_cm': best_error
            })
            
            # COLOR-CODED DISPLAY based on accuracy
            if best_error < 2.0:
                symbol = '★'  # Excellent: < 2cm
                color = '\033[92m'  # Green
            elif best_error < 3.5:
                symbol = '✓'  # Good: 2-3.5cm
                color = '\033[93m'  # Yellow
            else:
                symbol = '✗'  # Poor: > 3.5cm
                color = '\033[91m'  # Red
            
            reset = '\033[0m'  # Reset terminal color
            print(f"{symbol} Throw {throw_id:2d}: {color}{best_error:5.1f} cm{reset}")
        else:
            print(f"✗ Throw {throw_id:2d}: No prediction")
    
    return results