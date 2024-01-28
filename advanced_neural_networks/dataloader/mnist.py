"""
File contains: dataset class for MNIST dataset
#TODO:
-> Implement stratified cross validation on the training dataset
-> Hyperparameter tuning is to be done for each k-fold cross validation
    i.e., each combination of the hyperparameters should have a different cross
    validation
"""
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import yaml
import os

default_config_file = "mnist_config.yaml"

class MNISTDataset(torch.utils.data.Dataset):

    def __init__(self, config: str = default_config_file, location: str = "cloud",
                 train: str = False, transforms: list = [], one_hot: bool = True):
        if not isinstance(config, dict):
            with open(config, "rb") as f:
                self.config = yaml.load(f, Loader = yaml.FullLoader)
        else:
            self.config = config
        
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
        self.one_hot = one_hot
    
    def __getitem__(self, idx):

        img_tensor, label = self.dataset[idx]
        if self.one_hot:
            label_tensor = torch.Tensor([label]).long()
            label = F.one_hot(label_tensor, num_classes = 10).double()
        return (img_tensor, label)

    def __len__(self):
        return len(self.dataset)
        

if __name__ == "__main__":

    mnist_dataset = MNISTDataset(location="local", one_hot=False)
    img, label = mnist_dataset[0]
    print(label)