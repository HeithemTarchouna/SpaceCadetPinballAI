"""
Custom CNN Feature Extractor for Space Cadet Pinball AI.
Optimized for the pinball game's visual characteristics.
Supports hybrid observation with game state from shared memory.
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor
from gymnasium import spaces
import numpy as np


class PinballCNN(BaseFeaturesExtractor):
    """
    Custom CNN for pinball that focuses on:
    1. Ball detection (small, fast-moving object)
    2. Flipper positions
    3. Table features and targets
    
    The network uses larger initial filters to capture the ball,
    and includes residual-like connections for better gradient flow.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]  # Should be 3 for RGB
        
        # Convolutional layers optimized for 320x240 RGB input
        self.cnn = nn.Sequential(
            # First layer: Large kernel to capture overall table structure
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Second layer: Medium kernel for table features
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Third layer: Small kernel for fine details (ball position)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # Fourth layer: Even finer details
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Flatten(),
        )
        
        # Calculate the size of flattened features
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, *observation_space.shape[1:])
            n_flatten = self.cnn(sample).shape[1]
        
        # Fully connected layer to desired feature dimension
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Light dropout for regularization
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalize input to [0, 1] if it's in [0, 255]
        if observations.max() > 1.0:
            observations = observations / 255.0
        return self.linear(self.cnn(observations))


class PinballCNNSmall(BaseFeaturesExtractor):
    """
    Smaller/faster CNN for pinball - good for testing and faster iteration.
    Uses fewer parameters but maintains key detection capabilities.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Flatten(),
        )
        
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, *observation_space.shape[1:])
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.max() > 1.0:
            observations = observations / 255.0
        return self.linear(self.cnn(observations))


class PinballHybridExtractor(BaseFeaturesExtractor):
    """
    Hybrid feature extractor that combines:
    1. CNN for image processing (visual input)
    2. MLP for game state (ball position, speed from shared memory)
    
    This leverages your custom game interface for direct ball state!
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        # We extract features from a Dict space with 'image' and 'state' keys
        super().__init__(observation_space, features_dim)
        
        image_space = observation_space.spaces['image']
        state_space = observation_space.spaces['state']
        
        n_input_channels = image_space.shape[0]
        state_dim = state_space.shape[0]
        
        # CNN for image features (same architecture as PinballCNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, *image_space.shape[1:])
            cnn_output_dim = self.cnn(sample).shape[1]
        
        # MLP for game state features (ball x, y, speed, etc.)
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # Combined features dimension
        combined_dim = cnn_output_dim + 64
        
        # Final combination layer
        self.combine = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
    def forward(self, observations) -> torch.Tensor:
        # Extract image and state from dict observation
        image = observations['image']
        state = observations['state']
        
        # Normalize image
        if image.max() > 1.0:
            image = image / 255.0
        
        # Process through respective networks
        cnn_features = self.cnn(image)
        state_features = self.state_mlp(state)
        
        # Concatenate and combine
        combined = torch.cat([cnn_features, state_features], dim=1)
        return self.combine(combined)


# Policy kwargs for using custom CNNs
PINBALL_POLICY_KWARGS = {
    "features_extractor_class": PinballCNN,
    "features_extractor_kwargs": {"features_dim": 512},
    "net_arch": dict(pi=[256, 128], vf=[256, 128]),  # Separate actor/critic networks
}

PINBALL_POLICY_KWARGS_SMALL = {
    "features_extractor_class": PinballCNNSmall,
    "features_extractor_kwargs": {"features_dim": 256},
    "net_arch": dict(pi=[128, 64], vf=[128, 64]),
}

# Hybrid policy with game state integration
PINBALL_POLICY_KWARGS_HYBRID = {
    "features_extractor_class": PinballHybridExtractor,
    "features_extractor_kwargs": {"features_dim": 512},
    "net_arch": dict(pi=[256, 128], vf=[256, 128]),
}
