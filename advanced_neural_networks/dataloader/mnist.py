"""
File contains: dataset class for MNIST dataset
"""
import torch
import torchvision
import numpy as np
import yaml
import os

default_config_file = "mnist_config.yaml"

class MNISTDataset(torch.utils.data.Dataset):

    def __init__(self, config_file: str = default_config_file, location: str = "cloud",
                 train: str = False, transforms: list = []):

        with open(config_file, "rb") as f:
            self.config = yaml.load(f, Loader = yaml.FullLoader)
        
        self.root_dir = self.config["root_dir"][location]
        self.download = self.config["download"]
        if not os.path.isdir(self.root_dir):
            self.download = False
        
        self.dataset_statistics = self.config["dataset_statistics"]
        self.dataset_mean = self.dataset_statistics["mean"]
        self.dataset_std = self.dataset_statistics["std"]
        
        tensor_transform = torchvision.transforms.ToTensor()
        normalize_transform = torchvision.transforms.Normalize((self.dataset_mean,),
                                                               (self.dataset_std,))

        transforms.append(tensor_transform)
        transforms.append(normalize_transform)
        self.transforms = torchvision.transforms.Compose(transforms)

        self.dataset = torchvision.datasets.MNIST(root = self.root_dir,
                                                  train = train,
                                                  transform = self.transforms,
                                                  download = self.download)
    
    def __getitem__(self, idx):

        img_tensor, label = self.dataset[idx]
        return (img_tensor, label)
        

if __name__ == "__main__":

    mnist_dataset = MNISTDataset()
    img, label = mnist_dataset[0]