"""
File contains: Trainer class that contains the main training pipeline
#TODO:
-> include test to check if there is overlap between train and validation folds
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, DataLoader
import yaml
from tqdm import tqdm, tqdm_notebook

import advanced_neural_networks
from advanced_neural_networks.dataloader.mnist import MNISTDataset
from advanced_neural_networks.models.lenet import LeNet

module_dir = advanced_neural_networks.__path__[0]

class MNISTTrainer:

    OPTIMIZER_DICT = {"adam": torch.optim.Adam,
                      "adamax": torch.optim.Adamax,
                      "rmsprop": torch.optim.RMSprop}

    LOSS_DICT = {"bce": torch.nn.BCEWithLogitsLoss,
                 "cross_entropy": torch.nn.CrossEntropyLoss}

    def __init__(self, config_file: str, location: str = "cloud"):
        with open(config_file, "rb") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.dataset_config = self.config["dataset"]        
        self.trainer_config = self.config["train_config"]

        self.dataset = MNISTDataset(config = self.dataset_config, location = location, 
                                    train = True, transforms = [],
                                    one_hot = True)

        train_metadata_path = os.path.join(module_dir, "metadata", self.dataset_config["train_metadata"])
        self.train_metadata_df = pd.read_csv(train_metadata_path)
        self.kfolds = self.dataset_config["kfolds"]    

        self.max_epochs = self.trainer_config["max_epochs"]
        self.learning_rate = self.trainer_config["lr"]
        self.batch_size = self.trainer_config["batch_size"]
        self.model = None
        # self.configure_optimizers()    
    
    
    def configure_optimizers(self, model, optimizer_type: str, lr: float):        
        
        self.optimizer = getattr(torch.optim, optimizer_type)(model.parameters(), lr = lr)
        loss_type = self.trainer_config["loss"]
        self.criterion = self.LOSS_DICT[loss_type]()
    
    def train_epoch(self, iterator, model, device):
        epoch_loss = 0.0
        epoch_acc = 0.0        
        model.train()

        for (x, y) in tqdm_notebook(iterator, desc="Training", leave=False):

            x = x.to(device)
            y = y.squeeze().to(device)
            
            self.optimizer.zero_grad()

            y_pred = model(x)

            loss = self.criterion(y_pred, y)
            acc = self.calculate_accuracy(y_pred, y)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc
        
        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    def evaluate_epoch(self, iterator, model, device):

        epoch_loss = 0.0
        epoch_acc = 0.0        
        model.eval()

        with torch.no_grad():            
            for (x, y) in tqdm_notebook(iterator, desc="Evaluating", leave=False):

                x = x.to(device)
                y = y.squeeze().to(device)

                y_pred = model(x)
                loss = self.criterion(y_pred, y)
                acc = self.calculate_accuracy(y_pred, y)

                epoch_loss += loss.item()
                epoch_acc += acc

        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    @staticmethod
    def calculate_accuracy(y_pred, y):
        top_pred = torch.argmax(y_pred, dim = 1)
        label = torch.argmax(y, dim = 1)
        correct = torch.sum(top_pred == label)
        total_samples = len(label)
        acc = correct.item() / total_samples
        return acc
    
    @staticmethod
    def initialize_folds(kfolds: int):
        fold_array = np.arange(kfolds)
        fold_indices = {}
        for fold in range(kfolds):
            train_folds = fold_array[fold_array != fold]
            val_fold = np.array([fold])

            fold_dict = {"train": train_folds, "val": val_fold}
            fold_indices[f"fold_{fold}"] = fold_dict
        
        return fold_indices
    
    def cross_validate(self, model_params: dict, optimizer_params: dict):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        

        fold_dict = self.initialize_folds(self.kfolds)
        metrics_df = pd.DataFrame()
        best_valid_loss = {fold_idx: float("inf") for fold_idx in fold_dict}

        for fold_idx in tqdm_notebook(fold_dict, desc = "Cross Validation", leave = False):
            
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

            model = LeNet(**model_params)
            model = model.to(device)
            self.configure_optimizers(model, **optimizer_params)
            self.criterion = self.criterion.to(device)

            for epoch in range(self.max_epochs):

                train_loss, train_acc = self.train_epoch(train_iterator, model, device)
                val_loss, val_acc = self.evaluate_epoch(val_iterator, model, device)

                metrics = {"train_loss": train_loss,
                           "val_loss": val_loss,
                           "train_acc": train_acc,
                           "val_acc": val_acc,
                           "epoch": epoch,
                           "fold": fold_idx}
                
                metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index = True)

                if val_loss < best_valid_loss[fold_idx]:
                    best_valid_loss[fold_idx] = val_loss
                    torch.save(model.state_dict(), f"best-model-{fold_idx}.pt")
        
        return metrics_df

if __name__ == "__main_":
    config_file = os.path.join(module_dir, "trainer", "trainer_config.yaml")
    mnist_trainer = MNISTTrainer(config_file)
    fold_indices = MNISTTrainer.initialize_folds(10)
    print(fold_indices)


    






       

    
    
