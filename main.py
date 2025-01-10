import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import DBLM, DBLM2

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import json

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
# CUDA_VISIBLE_DEVICES=1 
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, get_result_list_tools
from utils.losses import compute_dfl_loss

parser = argparse.ArgumentParser(description='DBLM')


fix_seed = 2029
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer', help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2024, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='SNP', help='dataset type')
parser.add_argument('--root_path', type=str, default='./preprocessing', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='SNP_data_0909.csv', help='data file')

parser.add_argument('--loader', type=str, default='modal', help='dataset type') # CHECK
parser.add_argument('--freq', type=str, default='d',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--is_price', type=bool, default=False, help='Use price data?')


# forecasting task
parser.add_argument('--seq_len', type=int, default=252, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=22, help='prediction sequence length')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
# view define
parser.add_argument('--num_heads', type=int, default=4, help='input sequence length')
parser.add_argument('--num_enc_l', type=int, default=1, help='input sequence length')
parser.add_argument('--num_hidden', type=int, default=256, help='input hidden')

parser.add_argument('--loss_alpha', type=float, default=0.4, help='Loss function alpha')

# model define
parser.add_argument('--moving_avg', type=int, default=65, help='window size of moving average')
parser.add_argument('--enc_in', type=int, default=50, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=50, help='decoder input size')

parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
parser.add_argument('--lambda_value', type=str, default='B', help='Lmabda e.g.("HA", "A", "B", "C", "HC")')
parser.add_argument('--d_ff', type=int, default= 32, help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768

# Prompt
parser.add_argument('--ticker_dict_path', type=str, default='./view_bank/ticker_dict2.json', help='Path to the ticker dictionary JSON file') #./view_bank/ticker_dict2.json ticker_dict_dow
parser.add_argument('--sector_dict_path', type=str, default='./view_bank/sector_dict2.json', help='Path to the ticker dictionary JSON file') #./view_bank/sector_dict2.json sector_dict_dow


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=3, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate') #type1 COS
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start') # Check
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)

args = parser.parse_args()

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

os.makedirs("./result/loss/", exist_ok=True)



for ii in range(args.itr-1,args.itr):
    # setting record of experiments
    setting = 'md{}_da{}_sq{}_pr{}_lo{}_he{}_en{}_ll{}_la{}_de{}_is{}_pr{}_dff{}_hi{}_ii{}_Noise_None2'.format(
        args.model,
        args.data,
        args.seq_len,
        args.pred_len,
        int(args.loss_alpha*10),
        args.num_heads,
        args.num_enc_l,
        args.llm_model,
        args.lambda_value,
        args.des, args.batch_size, args.is_price, args.d_ff, args.num_hidden, ii)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader   = data_provider(args, 'val')
    test_data, test_loader   = data_provider(args, 'test')
    
    
    if args.model == 'DBLM2':
        model = DBLM2.Model(args).float()
    elif args.model == 'DBLM2':
        model = DBLM2.Model(args).float()
    elif args.model == "DBLM2":
        model = DBLM2.Model(args).float()
    else:
        model = DBLM.Model(args).float()

    path = os.path.join(args.checkpoints,  setting + '-' + args.model_comment)  # unique checkpoint saving path
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    
    
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    neg_criterion= nn.MSELoss()
    dfl_metric = compute_dfl_loss
        
    #dfl_metric = compute_dfl_loss #price_compute_dfl_loss compute_dfl_loss
    
    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler) 
    
    def denormalize_fn(batch_x, batch_y, history):
        
        a = batch_x.device
        b = batch_y.device
        c = history.device

        batch_x = batch_x.detach().cpu().numpy()
        batch_y = batch_y.detach().cpu().numpy()
        history = history.detach().cpu().numpy()

        # Reshape from (batch_size, seq_length, feature_size) to (batch_size * seq_length, feature_size)
        batch_size, seq_length, feature_size = batch_x.shape
        batch_size2, seq_length2, feature_size2 = history.shape

        batch_x_reshaped = batch_x.reshape(-1, feature_size)
        batch_y_reshaped = batch_y.reshape(-1, feature_size)
        history_reshaped = history.reshape(-1, feature_size)

        # Apply inverse_transform on reshaped data
        batch_x_denorm = train_data.inverse_transform(batch_x_reshaped)
        batch_y_denorm = train_data.inverse_transform(batch_y_reshaped)
        history_denorm = train_data.inverse_transform(history_reshaped)

        # Reshape back to original dimensions

        batch_x_denorm = batch_x_denorm.reshape(batch_size, seq_length, feature_size)
        batch_y_denorm = batch_y_denorm.reshape(batch_size, seq_length, feature_size)
        history_denorm = history_denorm.reshape(batch_size2, seq_length2, feature_size2)
        
        batch_x_denorm = torch.tensor(batch_x_denorm, device=a)
        batch_y_denorm = torch.tensor(batch_y_denorm, device=b)
        history_denorm = torch.tensor(history_denorm, device=c)

        #batch_x_denorm, batch_y_denorm, history_denorm = accelerator.prepare(batch_x_denorm, batch_y_denorm, history_denorm)
        return batch_x_denorm, batch_y_denorm, history_denorm
    

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        train_dfl_loss = []

        model.train()
        
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_x_mac) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()
            
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            seq_x_mac = seq_x_mac.float().to(accelerator.device)

            
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs, dfl_out, lambda_v, _, _,_ = model(batch_x, dec_inp, batch_x_mark, batch_y_mark, seq_x_mac)
                    
                    if args.is_price == True:
                        history = batch_x[:, -args.pred_len*3:, :]
                    else:
                        history = batch_x[:, -args.pred_len*3:, :]
                    
                    outputs = outputs[:, -args.pred_len:, :]
                    batch_y = batch_y[:, -args.pred_len:, :].to(accelerator.device)
                    dfl_out = dfl_out[:, -args.pred_len:, :]
                    loss = criterion(outputs, batch_y)
                    if args.is_price == True:
                        dfl_out, batch_y, history = denormalize_fn(dfl_out, batch_y, history)
                    else:
                        pass
                    loss2 = dfl_metric(dfl_out, batch_y, history, lambda_v, "mae")
                    tot_loss = (args.loss_alpha * loss) + ((1-args.loss_alpha) * loss2) 
                    train_dfl_loss.append(loss2.item())
                    train_loss.append(tot_loss.item())
            else:
                outputs, dfl_out, lambda_v, _, _,_ = model(batch_x, dec_inp, batch_x_mark, batch_y_mark, seq_x_mac)
                if args.is_price == True:
                    history = batch_x[:, -args.pred_len*3:, :]
                else:
                    history = batch_x[:, -args.pred_len*3:, :]
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :]
                dfl_out = dfl_out[:, -args.pred_len:, :]
                loss = criterion(outputs, batch_y)
                if args.is_price == True:
                    dfl_out, batch_y, history = denormalize_fn(dfl_out, batch_y, history)
                else:
                    pass
                
                loss2 = dfl_metric(dfl_out, batch_y, history, lambda_v, "mae")

                    
                tot_loss = (args.loss_alpha * loss) + ((1-args.loss_alpha) * loss2) 
                train_dfl_loss.append(loss2.item())
                train_loss.append(tot_loss.item())

            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(tot_loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(tot_loss)
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        train_dfl_loss = np.average(train_dfl_loss)
        
        vali_loss, val_mse_loss, vali_mae_loss, vali_dfl_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric, dfl_metric, denormalize_fn, "val")
        test_loss, test_mse_loss, test_mae_loss, test_dfl_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric, dfl_metric, denormalize_fn, "test")
        
        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.3f} T_DFL: {2:.3f} Vali Loss: {3:.3f} V_DFL: {4:.3f} Test Loss: {5:.3f} MAE Loss: {6:.3f} MSE Loss: {7:.3f} T_DFL: {8:.3f}".format(
                epoch + 1, train_loss, train_dfl_loss, vali_loss, vali_dfl_loss, test_loss, test_mae_loss, test_mse_loss, test_dfl_loss))

        early_stopping(vali_loss, model, path) 
        if early_stopping.early_stop:
            
            #w_preds1 = get_result_list_tools(args,  model, accelerator, test_loader,  cpu_compute_dfl_loss, lambda_v, setting)
            filename = f"./result/loss/{setting}.txt"
    
            with open(filename, 'w') as f:
                f.write(f"test_loss: {test_loss}\n")
                f.write(f"test_mae_loss: {test_mae_loss}\n")
                f.write(f"test_mse_loss: {test_mse_loss}\n")
                f.write(f"test_dfl_loss: {test_dfl_loss}\n")

            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

