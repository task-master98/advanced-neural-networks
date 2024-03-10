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
import torch.nn.functional as F
import yaml
from tqdm import tqdm, tqdm_notebook

import advanced_neural_networks
from advanced_neural_networks.trainer.trainer import MNISTTrainer
from advanced_neural_networks.dataloader.representation_dataset import Representations
from advanced_neural_networks.models.rbf_layer import RBFNetwork
from advanced_neural_networks.trainer.callbacks import EarlyStopping

class RBFTrainer(MNISTTrainer):

    OPTIMIZER_DICT = {"adam": torch.optim.Adam,
                      "adamax": torch.optim.Adamax,
                      "rmsprop": torch.optim.RMSprop}
    
    LOSS_DICT = {"bce": torch.nn.BCEWithLogitsLoss,
                 "cross_entropy": torch.nn.CrossEntropyLoss,
                 "mse": torch.nn.MSELoss}

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

        self.train_metadata_df = self.dataset.metadata_df
        
        self.kfolds = self.dataset_config["kfolds"]    

        self.max_epochs = self.trainer_config["max_epochs"]
        self.learning_rate = self.trainer_config["lr"]
        self.batch_size = self.trainer_config["batch_size"]
        self.model = None
        self.num_classes = 10
        self.x_train_numpy = self.convert_dataset_to_numpy()
    
    def find_cluster_centers(self, feature_vec, n_clusters, max_iter = 100):        
        k_means = KMeans(n_clusters = n_clusters, max_iter = max_iter)
        k_means.fit(feature_vec)
        cluster_centers = torch.tensor(k_means.cluster_centers_, dtype=torch.float32)
        return cluster_centers 

    def train_epoch(self, iterator, model, device):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0       
        model.train()

        for (x, y) in tqdm_notebook(iterator, desc="Training", leave=False):

            x = x.to(device)
            y_pred = model(x)
            y_onehot = F.one_hot(y.long(), num_classes = self.num_classes).squeeze()
            y_onehot = y_onehot.to(device)            
            loss = self.criterion(y_pred, y_onehot.float()) 

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            _, predicted = torch.max(y_pred, 1)
            correct_predictions += (predicted == torch.argmax(y_onehot, 1)).sum().item()
            total_samples += y_onehot.size(0) 
        
        return epoch_loss / total_samples, correct_predictions / total_samples
    
    def evaluate_epoch(self, iterator, model, device):

        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0       
        model.eval()

        with torch.no_grad():            
            for (x, y) in tqdm_notebook(iterator, desc="Evaluating", leave=False):

                x = x.to(device)
                y_pred = model(x)
                y_onehot = F.one_hot(y.long(), num_classes = self.num_classes).squeeze()
                y_onehot = y_onehot.to(device)            
                loss = self.criterion(y_pred, y_onehot.float())

                epoch_loss += loss.item() * x.size(0)
                _, predicted = torch.max(y_pred, 1)
                correct_predictions += (predicted == torch.argmax(y_onehot, 1)).sum().item()
                total_samples += y_onehot.size(0)
                

        return epoch_loss / total_samples, correct_predictions / total_samples
    
    def convert_dataset_to_numpy(self):
        x_train = []
        for itr in tqdm_notebook(range(len(self.dataset))):
            x, _ = self.dataset[itr]
            x_train.append(x.numpy())

        x_train = np.array(x_train)
        return x_train


    def train(self, rbf_params, optimizer_params):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        metrics_df = pd.DataFrame()

        train_iterator = DataLoader(self.dataset,
                                    batch_size = self.batch_size,
                                    shuffle = True)
        val_iterator = DataLoader(self.val_dataset,
                                  batch_size = self.batch_size) 

        n_clusters = rbf_params["kmeans"]["n_clusters"]
        max_iter = rbf_params["kmeans"]["max_iter"]
        print(f"Finding Cluster Centers...")
        cluster_centers = self.find_cluster_centers(self.x_train_numpy, n_clusters, max_iter)

        model_params = rbf_params["model"]
        model_params["centers"] = cluster_centers

        rbf_layer = RBFNetwork(**model_params)
        rbf_layer = rbf_layer.to(device)
        self.configure_optimizers(rbf_layer, **optimizer_params)
        self.criterion = self.criterion.to(device)        

        for epoch in tqdm_notebook(range(self.max_epochs), desc = "Epoch", leave = True):

            train_loss, train_acc = self.train_epoch(train_iterator, rbf_layer, device)
            val_loss, val_acc = self.evaluate_epoch(val_iterator, rbf_layer, device)

            metrics = {"train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "epoch": epoch}
            
            metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index = True)
        
        return metrics_df, rbf_layer
    
    def cross_validate(self, rbf_params, optimizer_params):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        

        fold_dict = self.initialize_folds(self.kfolds)
        metrics_df = pd.DataFrame()
        best_valid_loss = float("inf")
        best_model = None

        for fold_idx in tqdm_notebook(fold_dict, desc = "Cross Validation", leave = True):
            
            fold_info = fold_dict[fold_idx]
            val_fold_idx = fold_info["val"][0]

            train_indices = self.train_metadata_df.loc[self.train_metadata_df["fold"] != val_fold_idx].index.to_numpy()
            val_indices = self.train_metadata_df.loc[self.train_metadata_df["fold"] == val_fold_idx].index.to_numpy()

            train_dataset = Subset(self.dataset, train_indices)
            val_dataset = Subset(self.dataset, val_indices)

            train_iterator = DataLoader(train_dataset,
                                        batch_size = self.batch_size,
                                        shuffle = True)
            val_iterator = DataLoader(val_dataset,
                                      batch_size = self.batch_size)
            
            n_clusters = rbf_params["kmeans"]["n_clusters"]
            max_iter = rbf_params["kmeans"]["max_iter"]
            print(f"Finding Cluster Centers...")
            cluster_centers = self.find_cluster_centers(self.x_train_numpy, n_clusters, max_iter)

            model_params = rbf_params["model"]
            model_params["centers"] = cluster_centers

            rbf_layer = RBFNetwork(**model_params)
            rbf_layer = rbf_layer.to(device)
            self.configure_optimizers(rbf_layer, **optimizer_params)
            self.criterion = self.criterion.to(device)    

            early_stopping = EarlyStopping(patience = 3)

            for epoch in tqdm_notebook(range(self.max_epochs), desc = "Epoch", leave = False):

                train_loss, train_acc = self.train_epoch(train_iterator, rbf_layer, device)
                val_loss, val_acc = self.evaluate_epoch(val_iterator, rbf_layer, device)

                metrics = {"train_loss": train_loss,
                            "val_loss": val_loss,
                            "train_acc": train_acc,
                            "val_acc": val_acc,
                            "epoch": epoch,
                            "fold_idx": fold_idx}
                
                metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index = True)

                if early_stopping.should_stop(val_loss):
                    print(f"Early stopping at epoch {epoch}, validation loss: {val_loss}")
                    break

                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    best_model = model
            
        return metrics_df, best_model




