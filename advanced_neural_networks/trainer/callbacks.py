"""
File contains: early stopping callbacks
"""

import numpy as np

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_val_loss = np.inf
        self.no_improvement_count = 0

    def should_stop(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            return True
        else:
            return False

    def reset(self):
        self.best_val_loss = np.inf
        self.no_improvement_count = 0
