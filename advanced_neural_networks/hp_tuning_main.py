"""
File contains: pipeline for hyperparameter tuning using optuna
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from optuna.trial import TrialState
import datetime
import json

import advanced_neural_networks
from advanced_neural_networks.models.lenet import LeNet
from advanced_neural_networks.trainer.trainer import MNISTTrainer
from advanced_neural_networks.trainer.rbf_trainer import RBFTrainer

module_dir = advanced_neural_networks.__path__[0]
results_dir = os.path.join(module_dir, "results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
trainer_config = os.path.join(module_dir, "trainer", "trainer_config.yaml")

INPUT_SHAPE = {
    "MNIST": [1, 1, 28, 28],
    "FashionMNIST": [1, 1, 28, 28],
    "CIFAR10": [1, 3, 32, 32]
}

REPRESENTATION_SHAPE = {
    "MNIST": [720],
    "FashionMNIST": [720],
    "CIFAR10": [512]
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_best_metrics(metrics_df: pd.DataFrame, train_mode: str):
    if train_mode == "cross_validate":
        n_epochs = len(metrics_df.loc[metrics_df["fold"] == "fold_0"])
        last_epoch_df = metrics_df.loc[metrics_df["epoch"] == n_epochs - 1]
        mean_val_acc = last_epoch_df["val_acc"].mean()
        std_val_acc = last_epoch_df["val_acc"].std()
        return mean_val_acc, std_val_acc
    elif train_mode == "simple":        
        last_acc = metrics_df["val_acc"].to_numpy()[-1]
        return last_acc
    
    

def save_metrics_df(metrics_df: pd.DataFrame, trial_datetime: datetime.datetime, trial_number: int):
    datetime_string = trial_datetime.strftime("%Y%m%d")
    res_path = os.path.join(results_dir, datetime_string)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    csv_path = os.path.join(res_path, f"optuna_trial_{trial_number}.csv")
    metrics_df.to_csv(csv_path, index = False)
    

def objective(trial: optuna.Trial, data_type: str, train_mode: str):

    # define model hyperparams
    input_shape = INPUT_SHAPE[data_type]
    num_conv_layers = trial.suggest_int("num_conv_layers", 2, 3)
    out_channels_list = [int(trial.suggest_float("num_filter_"+str(i), 16, 64, step=16))
                   for i in range(num_conv_layers)] 
    out_channels_list = sorted(out_channels_list)
    kernel_size = 3
    pool_size = 2
    n_linear_layers = trial.suggest_int("n_linear_layers", 0, 1)
    n_neurons_list = [int(trial.suggest_float("n_neurons", 50, 100, step=10)) for _ in range(n_linear_layers)]
    n_neurons_list.append(10)
    n_linear_layers += 1

    model_params = {"input_shape": input_shape,
                    "n_conv_blocks": num_conv_layers,
                    "out_channel_list": out_channels_list,
                    "kernel_size": kernel_size,
                    "pool_size": pool_size,
                    "n_linear_layers": n_linear_layers,
                    "n_neurons_list": n_neurons_list}
    
    # define optimizers
    optimizer_type = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log = True)
    optimizer_params = {"optimizer_type": optimizer_type, "lr": learning_rate}    

    # intialize trainer
    mnist_trainer = MNISTTrainer(config_file = trainer_config, location = "cloud", data_type = data_type)    
    if train_mode == "cross_validate":
        print(f"Starting Cross Validation...")
        metrics_df, model = mnist_trainer.cross_validate(model_params, optimizer_params)
    elif train_mode == "simple":
        print(f"Starting simple training...")
        metrics_df, model = mnist_trainer.train(model_params, optimizer_params)

    start_date = trial.datetime_start
    metrics_df["optuna_trial"] = trial.number
    # set model parameters in trial
    model_state_dict = model.state_dict()
    model_state_dict_numpy = {key: value.detach().cpu().numpy() for key, value in model_state_dict.items()}
    trial.set_user_attr("best_model", value = json.dumps(model_state_dict_numpy, cls=NumpyEncoder))
    save_metrics_df(metrics_df, start_date, trial.number)   

    metrics = get_best_metrics(metrics_df, train_mode)

    return metrics

def rbf_objective(trial: optuna.Trial, data_type: str, train_mode: str):

    input_shape = REPRESENTATION_SHAPE[data_type]
    n_clusters = trial.suggest_int("n_clusters", 10, 400)

    rbf_params = {
    "kmeans": {"n_clusters": n_clusters, "max_iter": 100},
    "model": {
        "input_dim": input_shape,
        "num_centers": n_clusters,
        "output_dim": 10
    }}

    batch_size = 64
    optimizer_type = trial.suggest_categorical("optimizer", ["Adam", "Adamax"])
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log = True)
    optimizer_params = {"optimizer_type": optimizer_type, "lr": learning_rate}
    max_epochs = 20

    model_trainer = RBFTrainer(trainer_config, location = "cloud", data_type = data_type)
    model_trainer.max_epochs = max_epochs
    model_trainer.batch_size = batch_size
    model_trainer.criterion = nn.MSELoss()

    if train_mode == "cross_validate":
        print(f"Starting Cross Validation...")
        metrics_df, model = model_trainer.cross_validate(rbf_params, optimizer_params)
    elif train_mode == "simple":
        print(f"Starting simple training...")
        metrics_df, model = model_trainer.train(rbf_params, optimizer_params)
    
    start_date = trial.datetime_start
    metrics_df["optuna_trial"] = trial.number
    save_metrics_df(metrics_df, start_date, trial.number)

    metrics = get_best_metrics(metrics_df, train_mode)
    return metrics



if __name__ == "__main__":
    study = optuna.create_study(direction = "maximize")
    number_of_trials = 1
    study.optimize(objective, n_trials=number_of_trials)    


   
    


