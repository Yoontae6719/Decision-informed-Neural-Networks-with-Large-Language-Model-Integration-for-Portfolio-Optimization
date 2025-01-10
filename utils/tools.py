import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil

from tqdm import tqdm
import json
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_Pred, get_result_list
import os
plt.switch_backend('agg')
import argparse
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from utils.losses import cvx_layer
import time
import pandas as pd


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)



def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric, dfl_metric,denorm, set_name):
    total_loss = []
    total_mse_loss = []
    total_mae_loss = []
    total_dfl_loss = []

        
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_x_mac) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            seq_x_mac = seq_x_mac.float().to(accelerator.device)

            
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs, dfl_out, lambda_v,_, _ ,_= model(batch_x, dec_inp, batch_x_mark, batch_y_mark, seq_x_mac)
            else:
            
                outputs, dfl_out, lambda_v,_, _ ,_= model(batch_x, dec_inp, batch_x_mark, batch_y_mark, seq_x_mac)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            if args.is_price == True:    
                history = batch_x[:, -args.pred_len*3:, :]
            else:
                history = batch_x[:, -args.pred_len*3:, :]
                
            history = accelerator.gather_for_metrics(history)
            dfl_out = accelerator.gather_for_metrics(dfl_out)
            
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(accelerator.device)
            dfl_out = dfl_out[:, -args.pred_len:, :]
            
            pred = outputs.detach()
            true = batch_y.detach()
            history = history.detach()
            dfl_pred = dfl_out.detach()
            
            loss = criterion(pred, true)
            if args.is_price == True:
                dfl_out, batch_y, history = denorm(dfl_out, batch_y, history)
            else:
                pass
                
            loss2 = dfl_metric(dfl_out, batch_y, history, lambda_v, "mae")
                
                
            tot_loss = (args.loss_alpha * loss) + ((1-args.loss_alpha) * loss2) 

            mae_loss = mae_metric(pred, true)
            total_dfl_loss.append(loss2.item())
            total_mse_loss.append(loss.item())

            total_loss.append(tot_loss.item())
            total_mae_loss.append(mae_loss.item())

    total_loss = np.average(total_loss)
    total_dfl_loss = np.average(total_dfl_loss)
    total_mae_loss = np.average(total_mae_loss)
    total_mse_loss = np.average(total_mse_loss)

    model.train()
    return total_loss, total_mse_loss, total_mae_loss, total_dfl_loss

import gc

def get_result_list_tools(args, model, accelerator, test_input_loader, dfl_metric, lambda_v, settings):
    import torch
    import torch.nn as nn
    import gc
    import pandas as pd
    import numpy as np

    w_preds1 = []
    w_preds2 = []
    w_preds3 = []
    w_preds4 = []
    w_preds5 = []

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_x_mac) in enumerate(test_input_loader):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            seq_x_mac = seq_x_mac.float().to(accelerator.device)

            outputs, pre_, _, _, _, _ = model(
                batch_x, batch_y, batch_x_mark, batch_y_mark, seq_x_mac, None, None
            )
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :]
            history = batch_x[:, -args.pred_len * 3 :, :]
            pre_ = pre_[:, -args.pred_len:, :]

            # Ensure correct unpacking of returned values
            _, w1, _, _, _ = dfl_metric(
                pre_.detach(), batch_y.detach(), history.detach(), lambda_v, "mae"
            )


            w_preds1.append(w1)
           #w_preds2.append(w2)
           #w_preds3.append(w3)
           #w_preds4.append(w4)
           #w_preds5.append(w5)

    # Stack all w_preds
    w_preds1 = torch.cat(w_preds1, dim=0)


    # Gather w_preds from all processes
    w_preds1 = accelerator.gather(w_preds1)


    # On the main process, save the data
    if accelerator.is_main_process:
        # Convert tensors to NumPy arrays and reshape
        w_pred_np1 = w_preds1.cpu().numpy().reshape(-1, 50)


        print("Shape of w_pred_np1:", w_pred_np1.shape)

        # Read the date information
        w_shape = w_pred_np1.shape[0]
        w_date = pd.read_csv("w_infer_date.csv").iloc[:w_shape, :]

        # Create DataFrames and save each separately
        df_w1 = pd.concat([w_date.reset_index(drop=True), pd.DataFrame(w_pred_np1)], axis=1)


        # Save each DataFrame to its own CSV file
        df_w1.to_csv(f"./llama1/{settings}_w1.csv", index=False)

    torch.cuda.empty_cache()
    gc.collect()

    return w_preds1
