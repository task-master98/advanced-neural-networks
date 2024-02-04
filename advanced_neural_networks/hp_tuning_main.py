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

import advanced_neural_networks
from advanced_neural_networks.models.lenet import LeNet
from advanced_neural_networks.trainer.trainer import MNISTTrainer

module_dir = advanced_neural_networks.__path__[0]
results_dir = os.path.join(module_dir, "results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
trainer_config = os.path.join(module_dir, "trainer", "trainer_config.yaml")

def get_best_metrics(metrics_df: pd.DataFrame):
    sorted_metrics = metrics_df.sort_values("val_loss")
    best_metrics = sorted_metrics.iloc[0].to_dict()
    return best_metrics

def save_metrics_df(metrics_df: pd.DataFrame, trial_datetime: datetime.datetime, trial_number: int):
    datetime_string = trial_datetime.strftime("%Y%m%d")
    res_path = os.path.join(results_dir, datetime_string)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    csv_path = os.path.join(res_path, f"optuna_trial_{trial_number}.csv")
    metrics_df.to_csv(csv_path, index = False)
    

def objective(trial: optuna.Trial):

    # define model hyperparams
    input_shape = [1, 1, 28, 28]
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
    mnist_trainer = MNISTTrainer(config_file = trainer_config, location = "cloud")    

    metrics_df = mnist_trainer.cross_validate(model_params, optimizer_params)
    start_date = trial.datetime_start
    metrics_df["optuna_trial"] = trial.number
    save_metrics_df(metrics_df, start_date, trial.number)

    best_metrics = get_best_metrics(metrics_df)

    # set model parameters in trial
    trial.set_user_attr("best_model", value = model)

    return best_metrics["val_acc"]

if __name__ == "__main__":
    study = optuna.create_study(direction = "maximize")
    number_of_trials = 1
    study.optimize(objective, n_trials=number_of_trials)    


   
    


