import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
import gc


class Dataset_Multi_SNP(Dataset):
    def __init__(self, root_path = "./preprocessing",
                       flag = "train",
                       size = None,
                       data_path = "SNP_data_0909.csv", 
                       scale=True,
                       timeenc=1,
                       freq='d'):
        
        if size == None:
            self.seq_len   = 5 * 4 * 12  # 1 yeras, 5 days * 4 weeks * 12 months
            self.label_len = 5 * 4 # a month
            self.pred_len  = 5 * 4 # a month 
        else:
            self.seq_len   = size[0]
            self.label_len = size[1]
            self.pred_len  = size[2]
        
        # Define Tran, valid, test. 
        assert flag in ['train', 'val', "test"]       # add
        type_map = {'train': 0,  'val': 1, 'test': 2}  # add
        self.set_type = type_map[flag]            
        
        
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()      
        
    
    def __read_data__(self):
        
        # Step 0: Get Price data with macro data. And then Define split range.
        self.scaler = StandardScaler() 
        self.scaler_macro = StandardScaler() 
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = df_raw[df_raw["Date"] >= "2010-01-01"].reset_index(drop = True)
        
        
        target = list(df_raw.columns)
        for col in ["Date", "ICSA", "UMCSENT", "HSN1F", "UNRATE", "HYBS"]:
            if col in target:  
                target.remove(col)
        df_raw = df_raw[["Date", "ICSA", "UMCSENT", "HSN1F", "UNRATE", "HYBS"] + target]
        
        # Define train valid test
        num_train = df_raw[(df_raw['Date'] >= '2010-01-01') & (df_raw['Date'] <= '2016-12-31')].shape[0]
        num_vali =  df_raw[(df_raw['Date'] >= '2017-01-01') & (df_raw['Date'] <= '2019-12-31')].shape[0]
        num_test =  df_raw[(df_raw['Date'] >= '2020-01-01')].shape[0] 

        #num_train = df_raw[(df_raw['Date'] >= '2010-01-01') & (df_raw['Date'] <= '2017-12-31')].shape[0]
        #num_vali =  df_raw[(df_raw['Date'] >= '2018-01-01') & (df_raw['Date'] <= '2019-12-31')].shape[0]
        #num_test =  df_raw[(df_raw['Date'] >= '2020-01-01')].shape[0] 
        
        
        border1s = [0,
                    num_train  -self.seq_len,
                    len(df_raw) - num_test - self.seq_len]
        
        border2s = [num_train,
                    num_train  + num_vali,
                    len(df_raw)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Step 1. Preprocessing price (or return) data
        cols_data = df_raw.columns[6:] # Exclude ["Date", "ICSA", "UMCSENT", "HSN1F", "UNRATE", "HYBS"]
        df_data = df_raw[cols_data]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
            
        # Step 2. Preprocessing Date data
        df_stamp = df_raw[['Date']][border1:border2]
        df_stamp['Date'] = pd.to_datetime(df_stamp.Date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.Date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.Date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.Date.apply(lambda row: row.weekday(), 1)
            data_stamp = df_stamp.drop(['Date'], 1).values
            
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        # Step 3. Preproecssing macro data; Must need to scaling 
        df_macro = df_raw[["ICSA", "UMCSENT", "HSN1F", "UNRATE", "HYBS"]]
        train_macro = df_macro[border1s[0]:border2s[0]]
        self.scaler_macro.fit(train_macro.values)
        data_macro = self.scaler_macro.transform(df_macro.values)
        
        # Step 4. Get data.
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_macro = data_macro[border1:border2]

        self.data_stamp = data_stamp
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len # we use label len?
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x_mac = self.data_macro[s_begin:s_end]
        #seq_y_mac = self.data_macro[r_begin:r_end]
        
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_mac
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        # We do not need to considering invers transform the macro data. Cuz we do not predict the macro. 
        return self.scaler.inverse_transform(data)      




class Dataset_Pred(Dataset):
    def __init__(self, root_path = "./preprocessing",
                       flag = "train",
                       size = None,
                       data_path = "SNP100_data_0930.csv", 
                       scale=True,
                       timeenc=0,
                       freq='d',
                       train_last= "2022-12-31",
                       test_start= "2023-01-01",
                        ):
        
        if size == None:
            self.seq_len   = 5 * 4 * 12  # 1 yeras, 5 days * 4 weeks * 12 months
            self.label_len = 5 * 4 # a month
            self.pred_len  = 5 * 4 # a month 
        else:
            self.seq_len   = size[0]
            self.label_len = size[1]
            self.pred_len  = size[2]
        
        # Define Tran, valid, test. 
        assert flag in ['train',  "test"]       # add
        type_map = {'train': 0, 'test': 1}  # add
        self.set_type = type_map[flag]            
        self.set_type = type_map[flag]           
        
        self.train_last = train_last
        self.test_start = test_start
        
        
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()      
        
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def __read_data__(self):
        # Step 0: Get Price data with macro data. And then Define split range.
        # Note that we can change price data to return data.
        # Also, the macro data can change origin value data to percentage data.

        self.scaler = StandardScaler() 
        self.scaler_macro = StandardScaler() 
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = df_raw[df_raw["Date"] >= "2010-01-01"].reset_index(drop = True)
       # df_raw = df_raw[df_raw["Date"] <= self.data_last].reset_index(drop = True) #df_raw.iloc[:df_raw[df_raw["Date"] <= self.data_last].shape[0], :]

        target = list(df_raw.columns)
        for col in ["Date", "ICSA", "UMCSENT", "HSN1F", "UNRATE", "HYBS"]:
            if col in target:  
                target.remove(col)
        df_raw = df_raw[["Date", "ICSA", "UMCSENT", "HSN1F", "UNRATE", "HYBS"] + target]

        # Define train valid test
        num_train = df_raw[(df_raw['Date'] <= self.train_last)].shape[0]
        num_test  = df_raw[df_raw["Date"] >= self.test_start].shape[0] #.reset_index(drop = True) # add

        border1s = [0,
                    num_train - self.seq_len]

        border2s = [num_train,
                    len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Step 1. Preprocessing price (or return) data
        cols_data = df_raw.columns[6:] # Exclude ["Date", "ICSA", "UMCSENT", "HSN1F", "UNRATE", "HYBS"]
        df_data = df_raw[cols_data]

        train_data = df_data[border1s[0]:border2s[0]]
        
        if self.scale == True:
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        
        # Remove the following line to prevent overwriting scaled data
        #data = df_data.values

        # Step 2. Preprocessing Date data
        df_stamp = df_raw[['Date']][border1:border2]
        df_stamp['Date'] = pd.to_datetime(df_stamp.Date)

        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.Date.astype(object).apply(lambda row: row.year)
            df_stamp['month'] = df_stamp.Date.astype(object).apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.Date.astype(object).apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.Date.astype(object).apply(lambda row: row.weekday())
            data_stamp = df_stamp.drop(['Date'], axis=1).values

        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # Step 3. Preprocessing macro data; Must need to scaling 
        df_macro = df_raw[["ICSA", "UMCSENT", "HSN1F", "UNRATE", "HYBS"]]
        train_macro = df_macro[border1s[0]:border2s[0]]
        self.scaler_macro.fit(train_macro.values)
        data_macro = self.scaler_macro.transform(df_macro.values)

        # Step 4. Get data.
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_macro = data_macro
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len # we use label len?
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x_mac = self.data_macro[s_begin:s_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_mac
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        # We do not need to considering invers transform the macro data. Cuz we do not predict the macro. 
        return self.scaler.inverse_transform(data)      


def get_result_list(args, model, accelerator, test_input_loader, dfl_metric, lambda_v):
    import torch
    import torch.nn as nn
    #[0.0145, 0.2656, 0.9545, 3.4305, 5.4623]
    
        
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
        return 
    
    preds = []
    trues = []
    x_mark = []
    y_mark = [] 
    lambda_l = []
    history_ = []
    x_mark = []
    y_mark = [] 
    pre_list = []
    w_pred = []
    s_pred = []
    w_star_l = []
    s_star_l = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_x_mac) in enumerate(test_input_loader):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            seq_x_mac = seq_x_mac.float().to(accelerator.device)


            outputs, pre_, _, _,_ ,_= model(batch_x, batch_y, batch_x_mark, batch_y_mark, seq_x_mac, None, None)
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :]
            history = batch_x[:, -args.pred_len*3:, :]
            pre_ = pre_[:, -args.pred_len:, :]
            if args.is_price == True:
                pre_, batch_y, history = denormalize_fn(pre_, batch_y, history)
            else:
                pass
            
            _,w,s, w_star, s_star = dfl_metric(pre_.detach(), batch_y.detach(), history.detach(), lambda_v, "mae")

            pred = outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y.detach().cpu().numpy()  # .squeeze()
            history = history.detach().cpu().numpy()
            lambda_v = lambda_v
            pre_ = pre_.detach().cpu().numpy()
            w = w.detach().cpu().numpy()
            s = s.detach().cpu().numpy()
            w_star = w_star.detach().cpu().numpy()
            s_star = s_star.detach().cpu().numpy()
            
            
            x_mark_data = batch_x_mark.detach().cpu().numpy()  # .squeeze()
            y_mark_data = batch_y_mark.detach().cpu().numpy()  # .squeeze()

            history_.append(history)
            preds.append(pred)
            trues.append(true)
            lambda_l.append(lambda_v)
            x_mark.append(x_mark_data)
            y_mark.append(y_mark_data)
            pre_list.append(pre_)
            w_pred.append(w)
            s_pred.append(s)
            w_star_l.append(w_star)
            s_star_l.append(s_star)
            
    preds = np.array(preds)
    trues = np.array(trues)
    history_ = np.array(history_)
    lambda_l = np.array(lambda_l)
    x_mark = np.array(x_mark)
    y_mark = np.array(y_mark)
    pre_list = np.array(pre_list)
    w_pred = np.array(w_pred)
    s_pred = np.array(s_pred)
    w_star_l = np.array(w_star_l) 
    s_star_l = np.array(s_star_l)
    torch.cuda.empty_cache()
    gc.collect()

    
    return preds, trues, history_, pre_list, x_mark, y_mark,w_pred, s_pred, w_star_l, s_star_l





def get_result_list2(args, model, accelerator, test_input_loader, dfl_metric, lambda_v):
    import torch
    import torch.nn as nn
    #[0.0145, 0.2656, 0.9545, 3.4305, 5.4623]
    

    
    preds = []
    trues = []
    x_mark = []
    y_mark = [] 
    lambda_l = []
    history_ = []
    x_mark = []
    y_mark = [] 
    pre_list = []
    w_pred = []
    s_pred = []
    w_star_l = []
    s_star_l = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_x_mac) in enumerate(test_input_loader):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            seq_x_mac = seq_x_mac.float().to(accelerator.device)

            outputs, pre_, _, _,_ ,_= model(batch_x, batch_y, batch_x_mark, batch_y_mark, seq_x_mac)
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :]
            history = batch_x[:, -args.pred_len*3:, :]
            pre_ = pre_[:, -args.pred_len:, :]            
            _,w,_, _, _ = dfl_metric(pre_.detach(), batch_y.detach(), history.detach(), lambda_v, "mae")


            w = w.detach().cpu().numpy()

            w_pred.append(w)

            

    w_pred = np.array(w_pred)

    torch.cuda.empty_cache()
    gc.collect()

    return w_pred




def get_result_list_analysis(args, model, accelerator, test_input_loader, dfl_metric, lambda_v):
    import torch
    import torch.nn as nn
    #[0.0145, 0.2656, 0.9545, 3.4305, 5.4623]

    preds = []
    trues = []
    x_mark = []
    y_mark = [] 
    lambda_l = []
    history_ = []
    x_mark = []
    y_mark = [] 
    pre_list = []
    w_pred = []
    s_pred = []
    w_star_l = []
    s_star_l = []
    
    cov_x_list = []
    cov_y_list = []
    bx_l = []
    by_l = []
    attns_t_l = []
    attns_s_l = []
    attns_t_idx_l = []
    attns_s_idx_l = []
    
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_x_mac) in enumerate(test_input_loader):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            seq_x_mac = seq_x_mac.float().to(accelerator.device)


            outputs, pre_, _, attns_, attns_idx,_ = model(batch_x, batch_y, batch_x_mark, batch_y_mark, seq_x_mac)
            
            
            attn_t = attns_[0].float()
            attn_s = attns_[1].float()
            
            attn_idx_t = attns_idx[0].float()
            attn_idx_s = attns_idx[1].float()
            
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :]
            history = batch_x[:, -args.pred_len*3:, :]
            pre_ = pre_[:, -args.pred_len:, :]            
            
            
            w,s, w_star, s_star, cov_pred, cov_true, bx,by  = dfl_metric(pre_.detach(), batch_y.detach(), history.detach(), lambda_v, "mae")


            pred = outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y.detach().cpu().numpy()  # .squeeze()
            history = history.detach().cpu().numpy()
            lambda_v = lambda_v
            pre_ = pre_.detach().cpu().numpy()
            w = w.detach().cpu().numpy()
            s = s.detach().cpu().numpy()
            w_star = w_star.detach().cpu().numpy()
            s_star = s_star.detach().cpu().numpy()
            
            cov_pred = cov_pred.detach().cpu().numpy()
            cov_true = cov_true.detach().cpu().numpy()
            bx = bx.detach().cpu().numpy()
            by = by.detach().cpu().numpy()      
            
            attn_t = attn_t.squeeze().detach().cpu().numpy()
            attn_s = attn_s.squeeze().detach().cpu().numpy() 
            
            attn_idx_t = attn_idx_t.squeeze().detach().cpu().numpy()
            attn_idx_s = attn_idx_s.squeeze().detach().cpu().numpy() 
            
            
            x_mark_data = batch_x_mark.detach().cpu().numpy()  # .squeeze()
            y_mark_data = batch_y_mark.detach().cpu().numpy()  # .squeeze()

            history_.append(history)
            preds.append(pred)
            trues.append(true)
            lambda_l.append(lambda_v)
            x_mark.append(x_mark_data)
            y_mark.append(y_mark_data)
            pre_list.append(pre_)
            w_pred.append(w)
            s_pred.append(s)
            w_star_l.append(w_star)
            s_star_l.append(s_star)
            
            cov_x_list.append(cov_pred)
            cov_y_list.append(cov_true)
            bx_l.append(bx)
            by_l.append(by)
            attns_t_l.append(attn_t)
            attns_s_l.append(attn_s)

            attns_t_idx_l.append(attn_idx_t)
            attns_s_idx_l.append(attn_idx_s)
            
            
            
    preds = np.array(preds)
    trues = np.array(trues)
    history_ = np.array(history_)
    lambda_l = np.array(lambda_l)
    x_mark = np.array(x_mark)
    y_mark = np.array(y_mark)
    pre_list = np.array(pre_list)
    w_pred = np.array(w_pred)
    s_pred = np.array(s_pred)
    w_star_l = np.array(w_star_l) 
    s_star_l = np.array(s_star_l)
    
    cov_x_list = np.array(cov_x_list) 
    cov_y_list = np.array(cov_y_list)
    bx_l = np.array(bx_l) 
    by_l = np.array(by_l)
    #attns_t_l = np.array(attns_t_l) 
    #attns_s_l = np.array(attns_s_l)
    
    torch.cuda.empty_cache()
    gc.collect()

    
    return preds, trues, history_, pre_list, x_mark, y_mark,w_pred, s_pred, w_star_l, s_star_l, cov_x_list, cov_y_list, bx_l, by_l, attns_t_l, attns_s_l, attns_t_idx_l, attns_s_idx_l


def get_result_list3(args, model, accelerator, test_input_loader, dfl_metric, lambda_v):
    import torch
    import torch.nn as nn

    preds = []
    w_pred = []
    model.eval()
    with torch.no_grad():
        for batch in test_input_loader:
            # Unpack and move data to the appropriate device
            batch = [item.to(accelerator.device) for item in batch]
            batch_x, batch_y, batch_x_mark, batch_y_mark, seq_x_mac = batch

            # Forward pass
            outputs, pre_, _, _, _, _ = model(
                batch_x.float(),
                batch_y.float(),
                batch_x_mark.float(),
                batch_y_mark.float(),
                seq_x_mac.float(),
                None,
                None
            )

            # Process outputs
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :]
            history = batch_x[:, -args.pred_len * 3:, :]
            pre_ = pre_[:, -args.pred_len:, :]
            _, w, _, _, _ = dfl_metric(
                pre_.detach(),
                batch_y.detach(),
                history.detach(),
                lambda_v,
                "mae"
            )

            # Move weights to CPU and collect
            w = w.detach().cpu().numpy()
            w_pred.append(w)

    w_pred = np.array(w_pred)
    return w_pred

# ######

def get_result_analysis_list(args, model, accelerator, test_input_loader, dfl_metric, lambda_v):
    import torch
    import torch.nn as nn
    #[0.0145, 0.2656, 0.9545, 3.4305, 5.4623]
    
        
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
        return 
    
    preds = []
    trues = []
    x_mark = []
    y_mark = [] 
    lambda_l = []
    history_ = []
    x_mark = []
    y_mark = [] 
    pre_list = []
    w_pred = []
    s_pred = []
    w_star_l = []
    s_star_l = []    
    cov_x_list = []
    cov_y_list = []
    bx_l = []
    by_l = []
    attns_t_l = []
    attns_s_l = []
    attns_t_idx_l = []
    attns_s_idx_l = []

    
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_x_mac) in enumerate(test_input_loader):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            seq_x_mac = seq_x_mac.float().to(accelerator.device)


            outputs, pre_, _, attns_, attns_idx,_ = model(batch_x, batch_y, batch_x_mark, batch_y_mark, seq_x_mac)
            
            attn_t = attns_[0].float()
            attn_s = attns_[1].float()
            
            attn_idx_t = attns_idx[0].float()
            attn_idx_s = attns_idx[1].float()
            
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :]
            history = batch_x[:, -args.pred_len*3:, :]
            pre_ = pre_[:, -args.pred_len:, :]
            if args.is_price == True:
                pre_, batch_y, history = denormalize_fn(pre_, batch_y, history)
            else:
                pass
            
            w,s, w_star, s_star, cov_pred, cov_true, bx,by = dfl_metric(pre_.detach(), batch_y.detach(), history.detach(), lambda_v, "mae")

            pred = outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y.detach().cpu().numpy()  # .squeeze()
            history = history.detach().cpu().numpy()
            lambda_v = lambda_v
            pre_ = pre_.detach().cpu().numpy()
            w = w.detach().cpu().numpy()
            s = s.detach().cpu().numpy()
            w_star = w_star.detach().cpu().numpy()
            s_star = s_star.detach().cpu().numpy()
            
            cov_pred = cov_pred.detach().cpu().numpy()
            cov_true = cov_true.detach().cpu().numpy()
            bx = bx.detach().cpu().numpy()
            by = by.detach().cpu().numpy()      
            
            attn_t = attn_t.squeeze().detach().cpu().numpy()
            attn_s = attn_s.squeeze().detach().cpu().numpy() 
            
            attn_idx_t = attn_idx_t.squeeze().detach().cpu().numpy()
            attn_idx_s = attn_idx_s.squeeze().detach().cpu().numpy() 
            
            
            x_mark_data = batch_x_mark.detach().cpu().numpy()  # .squeeze()
            y_mark_data = batch_y_mark.detach().cpu().numpy()  # .squeeze()

            history_.append(history)
            preds.append(pred)
            trues.append(true)
            lambda_l.append(lambda_v)
            x_mark.append(x_mark_data)
            y_mark.append(y_mark_data)
            pre_list.append(pre_)
            w_pred.append(w)
            s_pred.append(s)
            w_star_l.append(w_star)
            s_star_l.append(s_star)
            
            cov_x_list.append(cov_pred)
            cov_y_list.append(cov_true)
            bx_l.append(bx)
            by_l.append(by)
            attns_t_l.append(attn_t)
            attns_s_l.append(attn_s)

            attns_t_idx_l.append(attn_idx_t)
            attns_s_idx_l.append(attn_idx_s)
            
            
            
    preds = np.array(preds)
    trues = np.array(trues)
    history_ = np.array(history_)
    lambda_l = np.array(lambda_l)
    x_mark = np.array(x_mark)
    y_mark = np.array(y_mark)
    pre_list = np.array(pre_list)
    w_pred = np.array(w_pred)
    s_pred = np.array(s_pred)
    w_star_l = np.array(w_star_l) 
    s_star_l = np.array(s_star_l)
    
    cov_x_list = np.array(cov_x_list) 
    cov_y_list = np.array(cov_y_list)
    bx_l = np.array(bx_l) 
    by_l = np.array(by_l)
    #attns_t_l = np.array(attns_t_l) 
    #attns_s_l = np.array(attns_s_l)
    
    torch.cuda.empty_cache()
    gc.collect()

    
    return preds, trues, history_, pre_list, x_mark, y_mark,w_pred, s_pred, w_star_l, s_star_l, cov_x_list, cov_y_list, bx_l, by_l, attns_t_l, attns_s_l, attns_t_idx_l, attns_s_idx_l


def get_result_analysis_emb_list(args, model, accelerator, test_input_loader, dfl_metric, lambda_v):
    import torch
    import torch.nn as nn
    #[0.0145, 0.2656, 0.9545, 3.4305, 5.4623]
    
        
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
        return 
    
    preds = []
    trues = []
    x_mark = []
    y_mark = [] 
    lambda_l = []
    history_ = []
    x_mark = []
    y_mark = [] 
    pre_list = []
    w_pred = []
    s_pred = []
    w_star_l = []
    s_star_l = []    
    cov_x_list = []
    cov_y_list = []
    bx_l = []
    by_l = []
    attns_t_l = []
    attns_s_l = []
    attns_t_idx_l = []
    attns_s_idx_l = []

    emb_t_l = []
    emb_s_l = []

    
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_x_mac) in enumerate(test_input_loader):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            seq_x_mac = seq_x_mac.float().to(accelerator.device)


            outputs, pre_, _, attns_, attns_idx, emb = model(batch_x, batch_y, batch_x_mark, batch_y_mark, seq_x_mac, None, None)
            
            attn_t = attns_[0].float()
            attn_s = attns_[1].float()
            
            attn_idx_t = attns_idx[0].float()
            attn_idx_s = attns_idx[1].float()
            
            emb_t = emb[0].float()
            emb_s = emb[1].float()
            
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :]
            history = batch_x[:, -args.pred_len*3:, :]
            pre_ = pre_[:, -args.pred_len:, :]
            if args.is_price == True:
                pre_, batch_y, history = denormalize_fn(pre_, batch_y, history)
            else:
                pass
            
            w,s, w_star, s_star, cov_pred, cov_true, bx,by = dfl_metric(pre_.detach(), batch_y.detach(), history.detach(), lambda_v, "mae")

            pred = outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y.detach().cpu().numpy()  # .squeeze()
            history = history.detach().cpu().numpy()
            lambda_v = lambda_v
            pre_ = pre_.detach().cpu().numpy()
            w = w.detach().cpu().numpy()
            s = s.detach().cpu().numpy()
            w_star = w_star.detach().cpu().numpy()
            s_star = s_star.detach().cpu().numpy()
            
            cov_pred = cov_pred.detach().cpu().numpy()
            cov_true = cov_true.detach().cpu().numpy()
            bx = bx.detach().cpu().numpy()
            by = by.detach().cpu().numpy()      
            
            attn_t = attn_t.squeeze().detach().cpu().numpy()
            attn_s = attn_s.squeeze().detach().cpu().numpy() 

            emb_t = emb_t.squeeze().detach().cpu().numpy()
            emb_s = emb_s.squeeze().detach().cpu().numpy() 
            
            attn_idx_t = attn_idx_t.squeeze().detach().cpu().numpy()
            attn_idx_s = attn_idx_s.squeeze().detach().cpu().numpy() 
            
            
            x_mark_data = batch_x_mark.detach().cpu().numpy()  # .squeeze()
            y_mark_data = batch_y_mark.detach().cpu().numpy()  # .squeeze()
            
            

            history_.append(history)
            preds.append(pred)
            trues.append(true)
            lambda_l.append(lambda_v)
            x_mark.append(x_mark_data)
            y_mark.append(y_mark_data)
            pre_list.append(pre_)
            w_pred.append(w)
            s_pred.append(s)
            w_star_l.append(w_star)
            s_star_l.append(s_star)
            
            cov_x_list.append(cov_pred)
            cov_y_list.append(cov_true)
            bx_l.append(bx)
            by_l.append(by)
            attns_t_l.append(attn_t)
            attns_s_l.append(attn_s)

            attns_t_idx_l.append(attn_idx_t)
            attns_s_idx_l.append(attn_idx_s)


            emb_t_l.append(emb_t)
            emb_s_l.append(emb_s)
            
            
            
    preds = np.array(preds)
    trues = np.array(trues)
    history_ = np.array(history_)
    lambda_l = np.array(lambda_l)
    x_mark = np.array(x_mark)
    y_mark = np.array(y_mark)
    pre_list = np.array(pre_list)
    w_pred = np.array(w_pred)
    s_pred = np.array(s_pred)
    w_star_l = np.array(w_star_l) 
    s_star_l = np.array(s_star_l)
    
    cov_x_list = np.array(cov_x_list) 
    cov_y_list = np.array(cov_y_list)
    bx_l = np.array(bx_l) 
    by_l = np.array(by_l)
    emb_t_l = np.array(emb_t_l) 
    emb_s_l = np.array(emb_s_l)
    
    #attns_t_l = np.array(attns_t_l) 
    #attns_s_l = np.array(attns_s_l)
    
    torch.cuda.empty_cache()
    gc.collect()

    
    return preds, trues, history_, pre_list, x_mark, y_mark,w_pred, s_pred, w_star_l, s_star_l, cov_x_list, cov_y_list, bx_l, by_l, attns_t_l, attns_s_l, attns_t_idx_l, attns_s_idx_l, emb_t_l, emb_s_l
