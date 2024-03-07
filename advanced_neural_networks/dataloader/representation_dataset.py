"""
File contains: Dataset to load representations
"""
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import yaml
import os

METADATA_NAMES = {
    "MNIST": "mnist",
    "FashionMNIST": "fashion_mnist",
    "CIFAR10": "cifar_10"
}

class Representations(torch.utils.data.Dataset):

    def __init__(self, config, data_type: str, train: bool, location: str = "cloud"):
        if not isinstance(config, dict):
            with open(config, "rb") as f:
                self.config = yaml.load(f, Loader = yaml.FullLoader)
        else:
            self.config = config

        self.root_dir = self.config["root_dir"][location]
        self.data_dir = os.path.join(self.root_dir, f"{data_type}_representations")
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError("No such dataset available")
        
        self.metadata_dir = os.path.join(self.root_dir, "metadata")
        metadata_filename = METADATA_NAMES[data_type]
        train_meta_suffix = self.config["train_metadata"]
        val_meta_suffix = self.config["test_metadata"]       
        

        if train:
            self.rep_dir = os.path.join(self.data_dir, "train")
            self.metadata_df = pd.read_csv(os.path.join(self.metadata_dir, f"{metadata_filename}_{train_meta_suffix}"))
        else:
            self.rep_dir = os.path.join(self.data_dir, "val")
            self.metadata_df = pd.read_csv(os.path.join(self.metadata_dir, f"{metadata_filename}_{val_meta_suffix}"))
    
    def load_representation(self, batch_idx: int):
        file_path = os.path.join(self.rep_dir, f"img_{batch_idx}.npz")
        representation_data = np.load(file_path)
        return representation_data["x"], representation_data["y"]
    
    def __getitem__(self, idx):
        x, y = self.load_representation(idx)
        metadata_label = self.metadata_df.iloc[idx].to_dict()["label"]
        assert metadata_label == y[0]

        x = torch.Tensor(x)
        y = torch.Tensor(y)
        return x, y



