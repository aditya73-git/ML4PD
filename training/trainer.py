import tensorflow as tf
import numpy as np

def train_with_validation(pinn, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
                         physics_loss_fn, scale_loss_fn,
                         epochs=150, batch_size=32, lr=0.001):
    """
    Train Physics-Informed Neural Network with validation monitoring.
    
    Uses custom training loop to combine physics loss with data loss.
    Implements dual loss system: actual physics loss for training + 
    scaled loss for human-readable monitoring.
    """
    
    # Adam optimizer - good default for most neural networks
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    # Get dataset sizes
    n_train = len(X_train_scaled)
    n_val = len(X_val_scaled)
    
    # Track losses for analysis
    train_losses = []      # Actual (unscaled) training loss
    val_losses = []        # Actual (unscaled) validation loss
    train_scaled_losses = []  # Scaled training loss (for display)
    val_scaled_losses = []    # Scaled validation loss (for display)
    
    for epoch in range(epochs):
        # Shuffle training data each epoch to prevent order bias
        indices = tf.random.shuffle(tf.range(n_train))
        X_shuffled = tf.gather(X_train_scaled, indices)
        y_shuffled = tf.gather(y_train_scaled, indices)
        
        # --- TRAINING PHASE ---
        epoch_train_loss = 0.0
        epoch_train_scaled = 0.0
        train_batches = 0
        
        # Process training data in batches
        for i in range(0, n_train, batch_size):
            batch_x = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            with tf.GradientTape() as tape:
                # Forward pass: get predictions from neural network
                predictions = pinn(batch_x, training=True)
                
                # Convert to tensors for physics loss computation
                batch_x_denorm = tf.convert_to_tensor(batch_x, dtype=tf.float32)
                predictions_denorm = tf.convert_to_tensor(predictions, dtype=tf.float32)
                
                # Calculate physics loss (main regularization)
                phys_loss, g_loss, h_loss, n_loss = physics_loss_fn(batch_x_denorm, predictions_denorm)
                
                # Data loss: MSE between predictions and actual next positions
                # Weighted at 10% to balance with physics constraints
                data_loss = tf.reduce_mean(tf.square(batch_y - predictions)) * 0.1
                
                # Total loss = physics loss + data loss
                total_loss = phys_loss + data_loss
                
                # Calculate scaled version for display/monitoring
                scaled_phys, scaled_g, scaled_h, scaled_n = scale_loss_fn(g_loss, h_loss, n_loss)
                scaled_total = scaled_phys + data_loss  # data loss already small
            
            # Backward pass: calculate gradients and update weights
            grads = tape.gradient(total_loss, pinn.trainable_variables)
            optimizer.apply_gradients(zip(grads, pinn.trainable_variables))
            
            # Accumulate losses for epoch average
            epoch_train_loss += total_loss
            epoch_train_scaled += scaled_total
            train_batches += 1
        
        # Calculate average training loss for this epoch
        avg_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0
        avg_train_scaled = epoch_train_scaled / train_batches if train_batches > 0 else 0
        
        train_losses.append(avg_train_loss.numpy())
        train_scaled_losses.append(avg_train_scaled.numpy())
        
        # --- VALIDATION PHASE ---
        epoch_val_loss = 0.0
        epoch_val_scaled = 0.0
        val_batches = 0
        
        # Process validation data (no gradient computation)
        for i in range(0, n_val, batch_size):
            batch_x = X_val_scaled[i:i+batch_size]
            batch_y = y_val_scaled[i:i+batch_size]
            
            # Forward pass only (no training flag)
            predictions = pinn(batch_x, training=False)
            
            # Prepare tensors for physics loss
            batch_x_denorm = tf.convert_to_tensor(batch_x, dtype=tf.float32)
            predictions_denorm = tf.convert_to_tensor(predictions, dtype=tf.float32)
            
            # Calculate losses (same as training but no gradients)
            phys_loss, g_loss, h_loss, n_loss = physics_loss_fn(batch_x_denorm, predictions_denorm)
            data_loss = tf.reduce_mean(tf.square(batch_y - predictions)) * 0.1
            total_loss = phys_loss + data_loss
            
            # Scaled version for monitoring
            scaled_phys, scaled_g, scaled_h, scaled_n = scale_loss_fn(g_loss, h_loss, n_loss)
            scaled_total = scaled_phys + data_loss
            
            epoch_val_loss += total_loss
            epoch_val_scaled += scaled_total
            val_batches += 1
        
        # Calculate average validation loss
        avg_val_loss = epoch_val_loss / val_batches if val_batches > 0 else 0
        avg_val_scaled = epoch_val_scaled / val_batches if val_batches > 0 else 0
        
        val_losses.append(avg_val_loss.numpy())
        val_scaled_losses.append(avg_val_scaled.numpy())
        
        # Progress reporting every 30 epochs
        if (epoch + 1) % 30 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss = {avg_train_scaled:.4f}, "
                  f"Val Loss = {avg_val_scaled:.4f}")
    
    # Return all loss histories for analysis
    return (train_losses, val_losses, 
            train_scaled_losses, val_scaled_losses)