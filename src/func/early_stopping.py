import numpy as np
import torch
import os

from src.utils.util import proj_root_dir, make_sure_dir, print_and_write_log


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, checkpoint_file=None, trace_func=print_and_write_log):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf #np.Inf 正无穷大的浮点表示, 常用于数值比较当中的初始值
        self.delta = delta
        self.checkpoint_file = checkpoint_file if checkpoint_file is not None else str(
            proj_root_dir / "checkpoints/checkpoint.pt")
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}') #以f开头表示在字符串内支持大括号内的python表达式
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        make_sure_dir(os.path.dirname(self.checkpoint_file))
        torch.save(model.state_dict(), self.checkpoint_file)
        self.val_loss_min = val_loss
