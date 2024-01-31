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

import advanced_neural_networks
from advanced_neural_networks.models.lenet import LeNet
from advanced_neural_networks.trainer.trainer import MNISTTrainer

module_dir = advanced_neural_networks.__path__[0]
trainer_config = os.path.join(module_dir, "trainer", "trainer_config.yaml")

def get_best_metrics(metrics_df: pd.DataFrame):
    sorted_metrics = metrics_df.sort_values("val_loss")
    best_metrics = sorted_metrics.iloc[0].to_dict()
    return best_metrics

def objective(trial: optuna.Trial):

    # define model hyperparams
    input_shape = [1, 1, 28, 28]
    num_conv_layers = trial.suggest_int("num_conv_layers", 2, 3)
    out_channels_list = [int(trial.suggest_discrete_uniform("num_filters", 6, 32, 6)) for _ in range(num_conv_layers)]
    kernel_size = 3
    pool_size = trial.suggest_int("pool_size", 1, 2)
    n_linear_layers = trial.suggest_int("n_linear_layers", 1, 2)
    n_neurons_list = [int(trial.suggest_discrete_uniform("n_neurons", 50, 100, 10)) for _ in range(n_linear_layers)]
    n_neurons_list.append(10)
    n_linear_layers += 1

    model = LeNet(input_shape = input_shape,
                  n_conv_blocks = num_conv_layers,
                  out_channel_list = out_channels_list,
                  kernel_size = kernel_size,
                  pool_size = pool_size,
                  n_linear_layers = n_linear_layers,
                  n_neurons_list = n_neurons_list)
    
    # define optimizers
    optimizer_type = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log = True)
    optimizer_params = {"optimizer_type": optimizer_type, "lr": learning_rate}    

    # intialize trainer
    mnist_trainer = MNISTTrainer(config_file = trainer_config, location = "cloud")    

    metrics_df = mnist_trainer.cross_validate(model, optimizer_params)

    best_metrics = get_best_metrics(metrics_df)

    return best_metrics["val_loss"]

    


   
    


