"""
File contains: Trainer class that contains the main training pipeline
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, DataLoader
import yaml
from tqdm import tqdm

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
        self.model_config = self.config["model"]["lenet"]
        self.trainer_config = self.config["train_config"]

        self.dataset = MNISTDataset(config = self.dataset_config, location = location, 
                                    train = True, transforms = [],
                                    one_hot = True)

        train_metadata_path = os.path.join(module_dir, "metadata", self.dataset_config["train_metadata"])
        self.train_metadata_df = pd.read_csv(train_metadata_path)    

        self.max_epochs = self.trainer_config["max_epochs"]
        self.learning_rate = self.trainer_config["lr"]
        self.batch_size = self.trainer_config["batch_size"]
        self.configure_optimizers()
        
    
    def configure_optimizers(self):
        optimizer_type = self.trainer_config["optimizer"]
        self.optimizer = self.OPTIMIZER_DICT[optimizer_type](self.model.parameters(),
                                             lr = self.learning_rate)
        loss_type = self.trainer_config["loss"]
        self.criterion = self.LOSS_DICT[loss_type]()
    
    def train_epoch(self, iterator, model):
        epoch_loss = 0.0
        device = model.device
        model.train()

        for (x, y) in tqdm(iterator, desc="Training", leave=False):

            x = x.to(device)
            y = y.to(device)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss = self.criterion(y_pred, y)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
        
        return epoch_loss / len(iterator)
    
    def evaluate_epoch(self, iterator, model):

        epoch_loss = 0.0
        device = model.device
        model.eval()

        with torch.no_grad():            
            for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):

                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)
                loss = self.criterion(y_pred, y)

                epoch_loss += loss.item()

        return epoch_loss / len(iterator)

        
    
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
    
    def cross_validate(self, kfolds: int):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        

        fold_dict = self.initialize_folds(kfolds)
        metrics_df = pd.DataFrame()
        best_valid_loss = {fold_idx: float("inf") for fold_idx in fold_dict}

        for fold_idx in tqdm(fold_dict, desc = "Cross Validation", leave = False):
            
            fold_info = fold_dict[fold_idx]
            val_fold_idx = fold_info["val"][0]

            train_indices = self.train_metadata_df.loc[self.train_metadata_df["fold"] != val_fold_idx]
            val_indices = self.train_metadata_df.loc[self.train_metadata_df["fold"] == val_fold_idx]

            train_dataset = Subset(self.dataset, train_indices)
            val_dataset = Subset(self.dataset, val_indices)

            train_iterator = DataLoader(train_dataset,
                                        batch_size = self.batch_size,
                                        shuffle = True)
            val_iterator = DataLoader(val_dataset,
                                      batch_size = self.batch_size)

            model = LeNet(**self.model_config)
            model = model.to(device)
            self.criterion = self.criterion.to(device)

            for epoch in range(self.max_epochs):

                train_loss = self.train_epoch(train_iterator, model)
                val_loss = self.evaluate_epoch(val_iterator, model)

                metrics = {"train_loss": train_loss,
                           "val_loss": val_loss,
                           "epoch": epoch,
                           "fold": fold_idx}
                
                metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index = True)

                if val_loss < best_valid_loss[fold_idx]:
                    best_valid_loss[fold_idx] = val_loss
                    torch.save(model.state_dict(), f"best-model-epoch{epoch}-{fold_idx}.pt")
        
        return metrics_df

if __name__ == "__main_":
    config_file = os.path.join(module_dir, "trainer", "trainer_config.yaml")
    mnist_trainer = MNISTTrainer(config_file)
    fold_indices = MNISTTrainer.initialize_folds(10)
    print(fold_indices)


    






       

    
    
