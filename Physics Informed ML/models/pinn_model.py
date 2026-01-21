"""
NEURAL NETWORK ARCHITECTURE MODULE
This module defines the Physics-Informed Neural Network (PINN) model architecture.
The design follows a simple, effective structure that was proven successful in the project.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

class SimplePINN(Model):
    """
    Simple Physics-Informed Neural Network architecture.
    
    This is the EXACT architecture that achieved 2.94 cm average error in the project.
    The simplicity is intentional - it was found that more complex architectures 
    overfitted to the limited training data (only 20 throws).
    
    Architecture: 6 inputs → 64 neurons → 32 neurons → 3 outputs
    Activation: LeakyReLU throughout (avoids dead neurons in ReLU)
    """
    
    def __init__(self):
        """
        Initialize the neural network layers.
        
        Layer structure:
        - Input: 6 features (x, y, z position + vx, vy, vz velocity)
        - Hidden Layer 1: 64 neurons with LeakyReLU activation
        - Hidden Layer 2: 32 neurons with LeakyReLU activation  
        - Output: 3 neurons (next x, y, z position predictions)
        
        This specific architecture (64→32) was determined through experimentation
        to provide sufficient capacity while avoiding overfitting.
        """
        super(SimplePINN, self).__init__()
        
        # First hidden layer: 64 neurons
        # Takes 6D input vector (position + velocity) and expands to 64 dimensions
        # This allows the network to learn complex feature combinations
        self.dense1 = layers.Dense(64)
        
        # LeakyReLU activation prevents "dead neurons" that can occur with standard ReLU
        # negative_slope=0.1 means: f(x) = x if x > 0, else 0.1*x
        # This small slope for negative values keeps gradients flowing during training
        self.activation1 = layers.LeakyReLU(negative_slope=0.1)
        
        # Second hidden layer: 32 neurons (compression layer)
        # Reduces dimensionality from 64 to 32, forcing the network to learn
        # the most important features for trajectory prediction
        self.dense2 = layers.Dense(32)
        
        # Second LeakyReLU activation maintains gradient flow
        self.activation2 = layers.LeakyReLU(negative_slope=0.1)
        
        # Output layer: 3 neurons (next x, y, z positions)
        # Linear activation (default) since we're predicting continuous position values
        # The physics constraints in the loss function handle the non-linear dynamics
        self.output_layer = layers.Dense(3)
    
    def call(self, inputs, training=False):
        """
        Forward pass through the neural network.
        
        This defines the computational graph: how data flows through the layers.
        
        Parameters:
        -----------
        inputs : tf.Tensor
            Input tensor of shape [batch_size, 6] containing:
            [x, y, z, vx, vy, vz] - current state vector
        training : bool, default=False
            Training flag (unused in this simple model but included for compatibility
            with TensorFlow's Model API for features like dropout/batch normalization)
        
        Returns:
        --------
        tf.Tensor
            Output tensor of shape [batch_size, 3] containing predicted next positions:
            [x_next, y_next, z_next]
        """
        # Layer 1: 6 → 64 dimensions with LeakyReLU activation
        x = self.dense1(inputs)
        x = self.activation1(x)
        
        # Layer 2: 64 → 32 dimensions with LeakyReLU activation
        x = self.dense2(x)
        x = self.activation2(x)
        
        # Output layer: 32 → 3 dimensions (linear activation)
        return self.output_layer(x)