"""
File contains: Model architecture for a simple convolution network
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

class SimpleConvNet(nn.Module):

    def __init__(self):

        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size = 3),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Flatten(),
            nn.Linear(26 * 26 * 10, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.Softmax(1)
        )

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
    
    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":

    random_tensor = torch.randn(1, 1, 28, 28)
    print(random_tensor.shape)
    model = SimpleConvNet()
    y = model(random_tensor)
    print(y.shape)
