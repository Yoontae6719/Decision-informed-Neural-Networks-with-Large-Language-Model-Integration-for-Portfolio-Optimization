# Imports
import argparse
import os
import random
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from models import DBLM2
from data_provider.data_loader import Dataset_Pred, get_result_list2
from utils.losses import (
    cvx_layer, cpu_compute_dfl_loss
)
from utils.infer_tool import (
    group_data_by_month_new, load_data, calculate_metrics, plot_cumulative_return
)
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali
from utils.main_analysis import *
# Matplotlib settings
import matplotlib.pyplot as plt

def calculate_portfolio_values_new(df_w_r, dates, returns, transaction_cost=0.01, initial_value=1.0):
    # Filter only rows where rebalancing occurs
    df_w_r = df_w_r[df_w_r["reb"] == 1].reset_index(drop=True)

    portfolio_values = [initial_value]
    turnover_list = []

    for i in range(df_w_r.shape[0] - 1):
        # Extract current and next weights
        current_date = str(df_w_r.iloc[i, 0]).split(" ")[0]
        current_weights = np.array(df_w_r.iloc[i, 5:], dtype=float)
        current_weights = np.round(current_weights, 4)

        rebalance_date = str(df_w_r.iloc[i + 1, 0]).split(" ")[0]
        next_weights = np.array(df_w_r.iloc[i + 1, 5:], dtype=float)
        next_weights = np.round(next_weights, 4)

        # Find indices in the dates array
        # Ensure current_date and rebalance_date exist in 'dates'
        start_idx = np.where(dates == pd.Timestamp(current_date))[0][0]
        end_idx = np.where(dates == pd.Timestamp(rebalance_date))[0][0]

        # Compute returns for the period between rebalances
        # Adjust indices as necessary based on your return data alignment
        period_returns = returns[start_idx:end_idx]

        for j in range(len(period_returns)):
            portfolio_return = np.dot(current_weights, period_returns[j])
            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))

        # Calculate turnover
        turnover = np.sum(np.abs(next_weights - current_weights)) / 2
        turnover_list.append(turnover)

        # Apply transaction costs at the rebalance point
        transaction_costs = turnover * transaction_cost
        portfolio_values[-1] *= (1 - transaction_costs)

    # After the final rebalance, extend returns until the end
    final_date = str(df_w_r.iloc[-1, 0]).split(" ")[0]
    final_weights = np.array(df_w_r.iloc[-1, 5:], dtype=float)
    final_weights = np.round(final_weights, 4)

    # Ensure final_date exists in 'dates'
    final_idx = np.where(dates == pd.Timestamp(final_date))[0][0]
    final_returns = returns[final_idx + 1:]

    for k in range(len(final_returns)):
        portfolio_return = np.dot(final_weights, final_returns[k])
        portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))

    return portfolio_values, turnover_list
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def calculate_metrics(portfolio_values, value_name, rf_path="rf.csv"):
    # Read the daily risk-free data
    rf = pd.read_csv(rf_path)

    # Ensure portfolio_values is in a DataFrame with a DatetimeIndex
    df = pd.DataFrame({"Cumulative Value": portfolio_values})
    df["rf"] = rf["rf"]
    df["date"] = rf["date"]
    df = df[["date","Cumulative Value","rf"]]
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date') 

    # Calculate daily returns
    df['daily_return'] = df['Cumulative Value'].pct_change()
    df = df.dropna(subset=['daily_return'])

    # Compute daily means
    daily_mean_return = df['daily_return'].mean()
    daily_rf_mean = df['rf'].mean()

    # Annualized Return (No RF)
    annualized_return = (1 + daily_mean_return) ** 252 - 1

    # Annualized Std (No RF)
    annualized_std = df['daily_return'].std() * np.sqrt(252)

    # Annualized Risk-Free Rate
    risk_free_annual = (1 + daily_rf_mean) ** 252 - 1

    # Annualized Excess Return = returns after subtracting RF each day
    daily_excess_return = df['daily_return'] - df['rf']
    annualized_excess_return = (1 + daily_excess_return.mean()) ** 252 - 1

    # Annualized Excess Std = std of daily excess returns
    annualized_excess_std = daily_excess_return.std() * np.sqrt(252)

    # Sharpe Ratio
    sharpe_ratio = (annualized_return - risk_free_annual) / annualized_std if annualized_std != 0 else np.nan

    # Arithmetic Return (annualized)
    arithmetic_return = daily_mean_return * 252

    # Geometric Return (annualized)
    geometric_mean_return = (df['Cumulative Value'].iloc[-1] / df['Cumulative Value'].iloc[0]) ** (1 / (len(df))) - 1
    geometric_return = (1 + geometric_mean_return) ** 252 - 1

    # Maximum Drawdown
    rolling_max = df['Cumulative Value'].cummax()
    drawdown = df['Cumulative Value'] / rolling_max - 1
    max_drawdown = drawdown.min() * -1

    # Annualized Skewness
    daily_skewness = skew(df['daily_return'])
    annualized_skewness = daily_skewness 

    # Annualized Kurtosis
    daily_kurtosis = kurtosis(df['daily_return'], fisher=False)
    annualized_kurtosis = daily_kurtosis

    # Cumulative Return
    cumulative_return = (df['Cumulative Value'].iloc[-1] / df['Cumulative Value'].iloc[0]) - 1

    # Monthly 95% VaR
    monthly_returns = df['Cumulative Value'].resample('M').last().pct_change().dropna()
    monthly_rf = (1 + df['rf']).resample('M').prod() - 1

    monthly_var = -monthly_returns.quantile(0.05)

    #monthly_var = -df['daily_return'].quantile(0.05)

    # Wealth (final portfolio value)
    wealth = df['Cumulative Value'].iloc[-1]

    # Sortino Ratio
    downside_mask = df['daily_return'] < df['rf']
    downside_returns = df.loc[downside_mask, 'daily_return'] - df.loc[downside_mask, 'rf']
    if len(downside_returns) > 1:
        annualized_downside_dev = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - risk_free_annual) / annualized_downside_dev
    else:
        sortino_ratio = np.nan

    # Return Over VaR (ROV)
    monthly_mean_return = monthly_returns.mean()
    monthly_mean_rf = monthly_rf.mean() 
    rov_ratio = (monthly_mean_return - monthly_mean_rf) / monthly_var if monthly_var != 0 else np.nan

    performance_summary = pd.DataFrame({
        'Metric': [
            'Annualized Return (No RF)', 
            'Annualized Std (No RF)',
            'Annualized Excess Return', 
            'Annualized Excess Std', 
            'Sharpe Ratio', 
            'Sortino Ratio',
            'Arithmetic Return', 
            'Geometric Return', 
            'Maximum Drawdown', 
            'Annualized Skewness', 
            'Annualized Kurtosis', 
            'Cumulative Return', 
            'Monthly 95% VaR', 
            'Wealth',
            'Return Over VaR'
        ],
        value_name: [
            annualized_return, # 1
            annualized_std,    # 2
            annualized_excess_return, 
            annualized_excess_std, 
            sharpe_ratio,      # 3
            sortino_ratio,     # 4
            arithmetic_return, 
            geometric_return,  
            max_drawdown,      # 5
            annualized_skewness, 
            annualized_kurtosis, 
            cumulative_return, 
            monthly_var,       # 6
            wealth,
            rov_ratio          # 7
        ]
    })

    return performance_summary
def get_pv(HA, AA, BB, CC, HC):
    import copy
    uni = copy.deepcopy(AA)
    uni.iloc[:, 5:] = np.float32(0.02) 
    transaction_cost = 0.01
    initial_value = 1.0
    file_path = "./preprocessing/SNP100_ret_1129.csv"

    dates, returns = load_data(file_path)
    uniform_weight = uni

    portfolio_values_u, turnover_list_u = calculate_portfolio_values_new(uniform_weight, dates, returns, transaction_cost, initial_value)
    portfolio_values_HA, turnover_list_HA = calculate_portfolio_values_new(HA, dates, returns, transaction_cost, initial_value)
    portfolio_values_AA, turnover_list_AA = calculate_portfolio_values_new(AA, dates, returns, transaction_cost, initial_value)
    portfolio_values_BB, turnover_list_BB = calculate_portfolio_values_new(BB, dates, returns, transaction_cost, initial_value)
    portfolio_values_CC, turnover_list_CC = calculate_portfolio_values_new(CC, dates, returns, transaction_cost, initial_value)
    portfolio_values_HC, turnover_list_HC = calculate_portfolio_values_new(HC, dates, returns, transaction_cost, initial_value)
    
    return portfolio_values_u, portfolio_values_HA, portfolio_values_AA, portfolio_values_BB, portfolio_values_CC, portfolio_values_HC, dates, returns


def plot_cumulative_return(dates, *portfolio_values_list):
    plt.figure(figsize=(10, 6))

    alpha_values = ["High Aggressive", "Aggressive", "Balanced", "Conservative", "High Conservative" , "EWP"]
    
    # 각 portfolio_values에 대해 플랏팅
    for i, portfolio_values in enumerate(portfolio_values_list):
        label = f'{alpha_values[i]}' if alpha_values[i] != "EWP" else 'EWP'
        color = 'black' if alpha_values[i] == "EWP" else None  # EWP 포트폴리오일 경우 검정색으로 설정
        plt.plot(dates[:len(portfolio_values)], portfolio_values, label=label, color=color)
    
    plt.xlabel('Date', fontname='Times New Roman')
    plt.ylabel('Portfolio Value', fontname='Times New Roman')
    plt.title('Cumulative Return of Portfolio vs Benchmark', fontname='Times New Roman')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def get_measure(UU,HA, AA, BB, CC, HC):
    
    out = pd.concat([calculate_metrics(HA, "High aggressive"),
                   calculate_metrics(AA, "Aggressive").iloc[:, 1:],
                   calculate_metrics(BB, "Balanced").iloc[:, 1:],
                   calculate_metrics(CC, "Conservative").iloc[:, 1:],
                   calculate_metrics(HC, "High conservative").iloc[:, 1:],
                   calculate_metrics(UU,"EWP").iloc[:, 1:]]
           , axis = 1)
    return out

def comparison_model(args,weight, lambda_):
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
    model = DBLM2.Model(args).to(torch.bfloat16)
    model.to(accelerator.device)
        
    test_set = Dataset_Pred(
                    root_path=args.root_path,
                    data_path=args.data_path,
                    flag="test",
                    size=[args.seq_len, 0, args.pred_len],
                    scale = False,
                    timeenc=0,
                    freq=args.freq,
                    train_last = "2019-12-31",
                    test_start = "2020-01-01")

    test_input_loader  = DataLoader( test_set,
                                         batch_size=20, #35
                                         shuffle=False,
                                         num_workers=10,
                                         drop_last=True)
    
    path = os.path.join(args.checkpoints, weight)
    best_model_path = path + '/' + 'checkpoint'
    model.load_state_dict(torch.load(best_model_path), strict=False)

    w = get_result_list2(args, model, accelerator, test_input_loader, cpu_compute_dfl_loss, lambda_)
    w_shape = w.reshape(-1, 50).shape[0]
    w_date = pd.read_csv("w_infer_date.csv").iloc[:w_shape, :]
    w = pd.concat([w_date, pd.DataFrame(w.reshape(-1, 50))], axis = 1)
    return w 

def plot_cumulative_return(dates, name, *portfolio_values_list):
    plt.figure(figsize=(10, 6))

    alpha_values = ["High Aggressive", "Aggressive", "Balanced", "Conservative", "High Conservative", "EWP"]

    # 각 portfolio_values에 대해 플랏팅
    for i, portfolio_values in enumerate(portfolio_values_list):
        label = f'{alpha_values[i]}' if alpha_values[i] != "EWP" else 'EWP'
        color = 'black' if alpha_values[i] == "EWP" else None  # EWP 포트폴리오일 경우 검정색으로 설정
        plt.plot(dates[:len(portfolio_values)], portfolio_values, label=label, color=color)

    plt.xlabel('Date', fontname='Times New Roman')
    plt.ylabel('Portfolio Value', fontname='Times New Roman')
    plt.title(name, fontname='Times New Roman')
    plt.legend()
    plt.grid(True)

    # 이미지 저장
    image_path = f'./hyper/{name}.png'
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path)
    plt.close()

def get_measure(UU, HA, AA, BB, CC, HC, name):
    out = pd.concat([calculate_metrics(HA, "High aggressive"),
                     calculate_metrics(AA, "Aggressive").iloc[:, 1:],
                     calculate_metrics(BB, "Balanced").iloc[:, 1:],
                     calculate_metrics(CC, "Conservative").iloc[:, 1:],
                     calculate_metrics(HC, "High conservative").iloc[:, 1:],
                     calculate_metrics(UU, "EWP").iloc[:, 1:]],
                    axis=1)
    
    # CSV 저장
    csv_path = f'./hyper/{name}.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    out.to_csv(csv_path, index=False)
    
    return out

def process_results(weight_lists, label, loss, head, encoder, dff, hidden, risk, ii):
    name = f'loss_{loss}_head_{head}_encoder_{encoder}_dff_{dff}_hidden_{hidden}_risk_{risk}_ii_{ii}'
    
    UU, HA, AA, BB, CC, HC, dates, returns = get_pv(*weight_lists)
    plot_cumulative_return(dates, name, HA, AA, BB, CC, HC, UU)
    measure_df = get_measure(UU, HA, AA, BB, CC, HC, name)
    print(measure_df)
    print(f"Results processed for {label}, risk={risk}")
    
def run_experiment(args, loss, head, encoder, dff, hidden, noise_values, model_name_template, risk, ii):
    args.heads = head
    args.num_enc_l = encoder
    args.d_ff = dff
    args.num_hidden = hidden
    args.loss = loss
    args.lambda_value = risk
    args.ii = ii
    
    weight_lists = []
    predict_list = []
    for noise in noise_values:
        model_name = model_name_template.format(loss=args.loss,
                                                head=args.heads,
                                                num_enc_l=args.num_enc_l,
                                                dff=args.d_ff,
                                                num_hidden=args.num_hidden,
                                                risk=args.lambda_value,
                                                ii = args.ii)
        
        w_list = comparison_model(args, model_name, noise)
        weight_lists.append(w_list)

    return weight_lists
