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
                 kernel_size: int, pool_size: int, n_linear_layers: int,
                 n_neurons_list: list,
                 dropout_prob: list):
        super().__init__()
       
        self.input_shape = input_shape
        self.n_conv_blocks = n_conv_blocks        
        self.out_channels_list = out_channel_list

        self.conv_blocks = []
        
        bs, in_channels, height, width = self.input_shape
        
        for itr in range(self.n_conv_blocks):
            
            out_channels = self.out_channels_list[itr]
            if itr == 0:
                # first conv block
                conv_block = self._conv_blocks(in_channels,
                                               out_channels,
                                               kernel_size = kernel_size,
                                               pool_size = pool_size,
                                               padding = "same",
                                               dropout = dropout_prob[itr])
                padding_size = (kernel_size - 1) / 2


            else:
                conv_block = self._conv_blocks(in_channels,
                                               out_channels,
                                               kernel_size = kernel_size,
                                               pool_size = pool_size,
                                               padding = 0,
                                               dropout = dropout_prob[itr])
                padding_size = 0

            height, width = self._compute_ft_map_shape(height, width, kernel_size, 1, padding_size, pool_size)
            
            self.conv_blocks.append(conv_block)            
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*self.conv_blocks)
        self.flatten = nn.Flatten()
        output_shape = (bs, out_channel_list[-1], height, width)

        self.linear_blocks = []
        in_features = int(out_channel_list[-1] * height * width)
        for jtr in range(n_linear_layers):
            out_features = n_neurons_list[jtr]
            linear_block = self._linear_block(in_features, out_features)
            self.linear_blocks.append(linear_block)
            in_features = out_features
        
        self.linear_layers = nn.Sequential(*self.linear_blocks)        
        self.softmax = nn.Softmax(1)
    
    def _conv_blocks(self, in_channels: int, out_channels: int,
                     kernel_size: int, pool_size: int,
                     padding: int = 0,
                     dropout: float = 0.2):
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride= 1, padding = padding)
        relu = nn.ReLU()
        pool_layer = nn.MaxPool2d(kernel_size = pool_size, stride = pool_size)
        dropout_layer = nn.Dropout(dropout)
        conv_block = nn.Sequential(
                        conv_layer,
                        relu,
                        pool_layer,
                        dropout_layer
        )

        return conv_block
    
    def _linear_block(self, in_features: int, out_features: int):
        linear_layer = nn.Linear(in_features, out_features)
        relu_layer = nn.ReLU()

        linear_block = nn.Sequential(linear_layer, relu_layer)
        return linear_block
    
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

    x = torch.randn(64, 3, 32, 32)
    n_conv_blocks = 3
    out_channel_list = [2, 8, 16]
    kernel_size = 3
    pool_size = 2
    n_linear_layers = 2
    n_neurons_list = [84, 10]
    dropout_prob = [0.2, 0.2, 0.2]
    model = LeNet(list(x.shape), n_conv_blocks, out_channel_list, kernel_size, pool_size, n_linear_layers, n_neurons_list, dropout_prob)
    y = model(x)
    print(y.shape)


