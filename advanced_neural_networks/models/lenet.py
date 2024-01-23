"""
File contains: Model for LeNet architecture
#TODO:
-> Number of conv blocks and max pooling layers must be a hyperparameter
-> The optimizer should be implemented with learning rate scheduler and other hyperparameters
-> implement the above using a config file
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

class LeNet(nn.Module):

    def __init__(self, input_shape: tuple, n_conv_blocks: int, out_channel_list: list,
                 kernel_size: int, pool_size: int):
        super().__init__()
       
        self.input_shape = input_shape
        self.n_conv_blocks = n_conv_blocks        
        self.out_channels_list = out_channel_list

        self.conv_blocks = []

        in_channels = 1
        bs, _, height, width = self.input_shape
        
        for itr in range(self.n_conv_blocks):
            
            out_channels = self.out_channels_list[itr]
            if itr == 0:
                # first conv block
                conv_block = self._conv_blocks(in_channels, out_channels, kernel_size = kernel_size, pool_size = pool_size, padding = "same")
                padding_size = (kernel_size - 1) / 2


            else:
                conv_block = self._conv_blocks(in_channels, out_channels, kernel_size = kernel_size, pool_size = pool_size, padding = 0)
                padding_size = 0

            height, width = self._compute_ft_map_shape(height, width, kernel_size, 1, padding_size, pool_size)
            
            self.conv_blocks.append(conv_block)            
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*self.conv_blocks)
        self.flatten = nn.Flatten()
        output_shape = (bs, out_channel_list[-1], height, width)
        self.fc1 = nn.Linear(in_features = int(out_channel_list[-1] * height * width), out_features = 10)
        self.lin_activation = nn.ReLU()
        

        self.linear_layers = nn.Sequential(
            self.fc1,
            self.lin_activation                       
        )
        self.softmax = nn.Softmax(1)
    
    def _conv_blocks(self, in_channels: int, out_channels: int,
                     kernel_size: int, pool_size: int,
                     padding: int = 0):
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride= 1, padding = padding)
        relu = nn.ReLU()
        pool_layer = nn.MaxPool2d(kernel_size = pool_size, stride = pool_size)

        conv_block = nn.Sequential(
                        conv_layer,
                        relu,
                        pool_layer
        )

        return conv_block
    
    def _compute_ft_map_shape(self, height: int, width: int,
                              kernel_size: int, stride_size: int,
                              padding: int, pool_size: int):
        
        ft_height = ((height - kernel_size + 2 * padding) // stride_size) + 1
        ft_width = ((width - kernel_size + 2 * padding) // stride_size) + 1
        pool_height = ((ft_height - pool_size) // pool_size) + 1
        pool_width = ((ft_width - pool_size) // pool_size) + 1
        return pool_height, pool_width
    
    def forward(self, x):
        ft_map = self.conv_layers(x)
        flatten = self.flatten(ft_map)
        linear_emb = self.linear_layers(flatten)
        probabilities = self.softmax(linear_emb)
        return probabilities

if __name__ == "__main__":

    x = torch.randn(1, 1, 28, 28)
    n_conv_blocks = 2
    out_channel_list = [6, 16]
    kernel_size = 5
    pool_size = 2
    model = LeNet(list(x.shape), n_conv_blocks, out_channel_list, kernel_size, pool_size, )
    y = model(x)
    print(y.shape)


