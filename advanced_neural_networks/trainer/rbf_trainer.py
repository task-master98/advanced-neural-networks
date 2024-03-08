"""
File contains: Train method for rbf based network
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import torch
from torch.utils.data import Subset, DataLoader
import yaml
from tqdm import tqdm, tqdm_notebook

import advanced_neural_networks
from advanced_neural_networks.trainer.trainer import MNISTTrainer
from advanced_neural_networks.dataloader.representation_dataset import Representations
from advanced_neural_networks.models.rbf_layer import RBFLayer
from advanced_neural_networks.models.rbf_utils import *

class RBFTrainer(MNISTTrainer):

    OPTIMIZER_DICT = {"adam": torch.optim.Adam,
                      "adamax": torch.optim.Adamax,
                      "rmsprop": torch.optim.RMSprop}

    DATA_PREFIX = {"MNIST": "mnist",
                   "FashionMNIST": "fashion_mnist",
                   "CIFAR10": "cifar10"}
    

    def __init__(self, config_file: str, location: str = "cloud", data_type: str = "MNIST"):
        with open(config_file, "rb") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.dataset_config = self.config["dataset"]        
        self.trainer_config = self.config["train_config"]

        self.dataset = Representations(config = self.dataset_config,
                                    data_type = data_type,
                                    train = True,
                                    location = "cloud")
        
        self.val_dataset = Representations(config = self.dataset_config,
                                    data_type = data_type,
                                    train = False,
                                    location = "cloud")
        
        self.kfolds = self.dataset_config["kfolds"]    

        self.max_epochs = self.trainer_config["max_epochs"]
        self.learning_rate = self.trainer_config["lr"]
        self.batch_size = self.trainer_config["batch_size"]
        self.model = None
    
    # def find_cluster_centers(self, feature_vec, n_clusters):        
    #     k_means = KMeans(n_clusters = n_clusters)
    #     k_means.fit(feature_vec)
    #     cluster_centers = torch.Tensor(k_means.cluster_centers_)
    #     return cluster_centers 

    def train(self, rbf_params, optimizer_params):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        metrics_df = pd.DataFrame()

        train_iterator = DataLoader(self.dataset,
                                    batch_size = self.batch_size,
                                    shuffle = True)
        val_iterator = DataLoader(self.val_dataset,
                                  batch_size = self.batch_size) 

        rbf_layer = RBFLayer(**rbf_params)
        model = model.to(device)
        self.configure_optimizers(model, **optimizer_params)
        self.criterion = self.criterion.to(device)        

        for epoch in range(self.max_epochs):

            train_loss, train_acc = self.train_epoch(train_iterator, rbf_layer, device)
            val_loss, val_acc = self.evaluate_epoch(val_iterator, rbf_layer, device)

            metrics = {"train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "epoch": epoch}
            
            metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index = True)
        
        return metrics_df, rbf_layer
        




