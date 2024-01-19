"""
File contains: Model for LeNet architecture
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

class LeNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, stride = 1, padding = "same")
        self.act1 = nn.Tanh()
        self.avg_pool_1 = nn.AvgPool2d(kernel_size = 2)        
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1)
        self.act2 = nn.Tanh()
        self.avg_pool_2 = nn.AvgPool2d(kernel_size = 2)        
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5, stride = 1)
        self.act3 = nn.Tanh()

        self.conv_layers = nn.Sequential(
            self.conv1,
            self.act1,
            self.avg_pool_1,            
            self.conv2,
            self.act2,
            self.avg_pool_2,            
            self.conv3,
            self.act3
        )
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features = 120, out_features = 84)
        self.act4 = nn.Tanh()
        self.fc2 = nn.Linear(in_features = 84, out_features = 10)

        self.linear_layers = nn.Sequential(
            self.fc1,
            self.act4,
            self.fc2            
        )
        self.softmax = nn.Softmax(1)
    
    def forward(self, x):

        embedding = self.conv_layers(x)
        flattened_emb = self.flatten(embedding)
        linear_emb = self.linear_layers(flattened_emb)
        probabilities = self.softmax(linear_emb)

        return probabilities

if __name__ == "__main__":

    x = torch.randn(4, 1, 28, 28)
    model = LeNet()
    y = model(x)
    print(y.shape)


