"""
PHYSICS LOSS MODULE
This module defines the physics-informed loss functions that enforce physical 
constraints on the neural network's predictions. These constraints ensure the 
model learns trajectories that obey real-world physics laws.
"""

import tensorflow as tf

def physics_loss_original(inputs, predictions, g=9.81, dt=0.0167):
    """
    Computes the physics-informed loss that enforces three physical constraints.
    
    This is the EXACT loss function from the successful model run that achieved
    2.94 cm average error. It combines:
    1. Gravity constraint: Enforces -9.81 m/s² vertical acceleration
    2. Horizontal motion constraint: Minimizes horizontal acceleration
    3. Negative coordinate pattern: Domain-specific constraint for throw patterns
    
    Parameters:
    -----------
    inputs : tf.Tensor
        Input state vector [x0, y0, z0, vx0, vy0, vz0] - current position and velocity
    predictions : tf.Tensor
        Neural network predictions for next position [x_pred, y_pred, z_pred]
    g : float, default=9.81
        Gravitational acceleration constant in m/s²
    dt : float, default=0.0167
        Time step between measurements in seconds (≈60 Hz sampling rate)
    
    Returns:
    --------
    tuple of (total_loss, gravity_loss, horizontal_loss, negative_loss)
        Total weighted loss and individual component losses for monitoring
    """
    
    # Extract position and velocity components from input state vector
    # Positions: x0, y0, z0 (current position in meters)
    # Velocities: vx0, vy0, vz0 (current velocity in m/s)
    x0, y0, z0 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    vx0, vy0, vz0 = inputs[:, 3], inputs[:, 4], inputs[:, 5]
    
    # Extract predicted next positions from neural network output
    x_pred, y_pred, z_pred = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    
    # Calculate predicted velocities using finite difference approximation
    # Velocity = (position change) / (time step)
    vx_pred = (x_pred - x0) / dt  # Predicted x-velocity
    vy_pred = (y_pred - y0) / dt  # Predicted y-velocity
    vz_pred = (z_pred - z0) / dt  # Predicted z-velocity
    
    # 1. GRAVITY CONSTRAINT - Most important physical law
    # Enforces Newton's second law: F = ma → vertical acceleration = -g
    # Computes: (vz_pred - vz0)/dt = acceleration in z-direction
    # Penalizes deviations from -9.81 m/s² (negative because gravity pulls down)
    gravity_loss = tf.reduce_mean(tf.square((vz_pred - vz0)/dt + g))
    
    # 2. HORIZONTAL MOTION CONSTRAINT - Air resistance approximation
    # Assumes minimal horizontal acceleration (negligible air resistance)
    # Penalizes any horizontal acceleration (changes in vx or vy)
    # This constraint captures the near-inertial horizontal motion in beer pong
    horizontal_loss = tf.reduce_mean(
        tf.square((vx_pred - vx0)/dt) +  # x-acceleration
        tf.square((vy_pred - vy0)/dt)    # y-acceleration
    )
    
    # 3. NEGATIVE COORDINATES CONSTRAINT - Domain-specific pattern
    # Based on observation that throws starting in negative x/y coordinates
    # tend to stay in negative territory (thrower's release pattern)
    # -0.05 meter threshold was determined empirically from training data
    
    # Create binary masks for negative starting positions
    negative_x_mask = tf.cast(x0 < -0.05, tf.float32)  # 1 if x0 < -0.05, else 0
    negative_y_mask = tf.cast(y0 < -0.05, tf.float32)  # 1 if y0 < -0.05, else 0
    
    # Penalize predictions that go positive when starting negative
    # relu(x_pred + 0.05) = 0 if x_pred < -0.05, positive otherwise
    # Only applies when mask = 1 (starting position was negative)
    negative_loss = tf.reduce_mean(
        negative_x_mask * tf.square(tf.nn.relu(x_pred + 0.05)) +  # X-direction penalty
        negative_y_mask * tf.square(tf.nn.relu(y_pred + 0.05))   # Y-direction penalty
    )
    
    # ORIGINAL WEIGHTS THAT WORKED - Determined through extensive experimentation
    # These weights balance the three constraints effectively:
    total_loss = (
        1.0 * gravity_loss +      # Strongest weight: fundamental physics
        0.2 * horizontal_loss +   # Medium weight: air resistance approximation
        0.5 * negative_loss       # High weight: important domain pattern
    )
    
    return total_loss, gravity_loss, horizontal_loss, negative_loss


def scaled_physics_loss(gravity_loss, horizontal_loss, negative_loss):
    """
    Scales physics loss components for human-readable display and monitoring.
    
    PROBLEM: Raw physics loss values are in different units and scales:
    - Acceleration terms: m²/s⁴ (very large numbers, often 10,000+)
    - Position terms: m² (moderate numbers)
    
    SOLUTION: Apply different scaling factors to bring all components to
    similar magnitude for easier monitoring during training.
    
    IMPORTANT: This scaling is ONLY for display/monitoring. The actual
    training uses the unscaled losses to maintain proper physics relationships.
    
    Parameters:
    -----------
    gravity_loss : tf.Tensor
        Unscaled gravity constraint loss (in m²/s⁴ units)
    horizontal_loss : tf.Tensor
        Unscaled horizontal constraint loss (in m²/s⁴ units)
    negative_loss : tf.Tensor
        Unscaled negative coordinates loss (in m² units)
    
    Returns:
    --------
    tuple of (scaled_total, scaled_gravity, scaled_horizontal, scaled_negative)
        All losses scaled to similar magnitude for monitoring
    """
    
    # Apply scaling factors determined empirically from typical loss magnitudes
    scaled_gravity = gravity_loss / 10000.0    # Scale from m²/s⁴ → ~0-10 range
    scaled_horizontal = horizontal_loss / 10000.0  # Same scaling for acceleration terms
    scaled_negative = negative_loss / 100.0    # Scale from m² → ~0-10 range
    
    # Recombine with original weights for consistency
    scaled_total = (
        1.0 * scaled_gravity +      # Maintains relative importance
        0.2 * scaled_horizontal +   # Same weight relationships
        0.5 * scaled_negative       # As original model
    )
    
    return scaled_total, scaled_gravity, scaled_horizontal, scaled_negative