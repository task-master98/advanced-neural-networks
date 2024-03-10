"""
File contains: Implementation of RBF and RBF network
"""
import torch
import torch.nn as nn
import numpy as np

class RBF_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, num_centers, centers):
        super(RBF_Layer, self).__init__()
        self.num_centers = num_centers
        self.centers = nn.Parameter(centers)  # Learnable RBF centers
        self.weights = nn.Parameter(torch.randn(output_dim, num_centers))  # Learnable weights
        self.bias = nn.Parameter(torch.randn(output_dim))
        self.sigma = self.compute_rbf_sigma()

    def compute_rbf_sigma(self):
        # Assuming you have your input data X_train as a numpy array
        print(f"Computing sigma...")
        X_train = np.array(self.centers.detach().cpu())  # Detach tensor and move to CPU
        
        # Compute the standard deviation parameter for RBF
        pairwise_distances = np.linalg.norm(X_train[:, np.newaxis] - X_train, axis=-1)
        rbf_sigma = np.median(pairwise_distances)  # Using the median trick
        return rbf_sigma

    def forward(self, x):
        # Compute distances between input and centers
        dists = torch.cdist(x, self.centers)
        # Apply RBF activation function (e.g., Gaussian)
        activations = torch.exp(-dists / (2 * self.sigma))  # Example Gaussian RBF
        # print(activations.shape)
        # Apply trainable weights
        weighted_activations = torch.matmul(activations, self.weights.t()) + self.bias
        return weighted_activations

class RBFNetwork(nn.Module):
    def __init__(self, input_dim, num_centers, centers, output_dim):
        super(RBFNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.output_dim = output_dim
        self.rbf_layer = RBF_Layer(input_dim, output_dim, num_centers, centers)
        self.activation = nn.Softmax()
        

    def forward(self, x):
        rbf_output = self.rbf_layer(x)
        output = self.activation(rbf_output)
        return output