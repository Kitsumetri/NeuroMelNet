import torch

import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Implements a residual block.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride value for the first convolutional layer. Default is 1.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer with batch normalization
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer with batch normalization
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to match dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = F.relu(self.bn1(self.conv1(x)))  # First convolution followed by ReLU activation and batch normalization
        out = self.bn2(self.conv2(out))        # Second convolution followed by batch normalization
        out += self.shortcut(x)                # Add shortcut connection
        out = F.relu(out)                      # Apply ReLU activation
        return out

class NeuroMelNet(nn.Module):
    """
    Implements the NeuroMelNet model for classification.
    
    Args:
        num_classes (int, optional): Number of classes. Default is 2.
    """

    def __init__(self, num_classes: int = 2):
        super(NeuroMelNet, self).__init__()
        
        # Initial convolutional layer with ReLU activation
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        
        # Residual blocks
        self.block1 = ResidualBlock(32, 64)
        self.block2 = ResidualBlock(64, 128)

        self.dropout1 = nn.Dropout2d(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
        
        # GRU layer for sequential processing
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NeuroMelNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Add singleton dimensions to input tensor for compatibility with convolutional layers
        x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        
        # Initial convolution followed by batch normalization and ReLU activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # Max pooling
        x = self.dropout1(x)  # Dropout
        
        # Residual blocks
        x = self.block1(x)
        x = self.block2(x)
        
        # Global average pooling
        x = x.mean([2, 3])
        
        # Reshape for input to GRU layer
        x = x.view(x.size(0), 1, 128)
        
        # GRU layer
        x, _ = self.gru(x)
        
        # Extract last hidden state
        x = x[:, -1, :]
        
        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # Sigmoid activation for binary classification
        x = F.sigmoid(x)
        
        return x
