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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


# ##

def load_data(file_path):
    ret_data = pd.read_csv(file_path).iloc[:, :51]
    ret_data['Date'] = pd.to_datetime(ret_data['Date'])
    ret_data = ret_data[ret_data["Date"] >= "2020-01-01"]
    dates = ret_data['Date'].values
    returns = ret_data.iloc[:, 1:].values
    return dates, returns


def calculate_portfolio_values(w_list, dates, returns, transaction_cost=0.01, initial_value=1.0):
    portfolio_values = [initial_value]
    turnover_list = []

    for i in range(len(w_list)-1):
        current_weights = w_list[i][0]
        current_weights = np.array(current_weights, dtype=float)
        current_weights = np.round(current_weights, 4)
        currendt_date_year = str(w_list[i][1][0])
        currendt_date_month = str(w_list[i][1][1])
        currendt_date_day = str(w_list[i][1][2])
        currendt_date = currendt_date_year +"-"+ currendt_date_month +"-"+currendt_date_day

        next_weights = w_list[i + 1][0]
        next_weights = np.array(next_weights, dtype=float)
        next_weights = np.round(next_weights, 4)

        rebalance_date_year = str(w_list[i + 1][1][0])
        rebalance_date_month = str(w_list[i + 1][1][1])
        rebalance_date_day = str(w_list[i + 1][1][2])
        rebalance_date = rebalance_date_year +"-"+ rebalance_date_month +"-"+rebalance_date_day

        #print("currendt_date:", currendt_date, "rebalance_date:", rebalance_date)

        start_idx = np.where(dates == pd.Timestamp(currendt_date))[0][0]
        end_idx   = np.where(dates == pd.Timestamp(rebalance_date))[0][0]
        period_returns = returns[start_idx:end_idx]

        for j in range(len(period_returns)):
            portfolio_return = np.dot(current_weights, period_returns[j])
            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))

        turnover = np.sum(np.abs(next_weights - current_weights)) / 2
        turnover_list.append(turnover)
        transaction_costs = turnover * transaction_cost
        portfolio_values[-1] *= (1 - transaction_costs)

    final_weights = w_list[-1][0]
    final_weights = np.array(final_weights, dtype=float)
    final_weights = np.round(final_weights, 4)

    final_date_year = str(w_list[-1][1][0])
    final_date_month = str(w_list[-1][1][1])
    final_date_day = str(w_list[-1][1][2])
    final_date = final_date_year +"-"+ final_date_month +"-"+final_date_day

    final_idx = np.where(dates == pd.Timestamp(final_date))[0][0]
    final_returns = returns[final_idx+1:]
    for k in range(len(final_returns)):
        portfolio_return = np.dot(final_weights, final_returns[k])
        portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
        
    return portfolio_values, turnover_list


def plot_cumulative_return(dates, portfolio_values):
    plt.figure(figsize=(10, 6))
    plt.plot(dates[:len(portfolio_values)], portfolio_values, label='Cumulative Return')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title('Cumulative Return of Portfolio vs Benchmark')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_metrics(portfolio_values, value_name):
    df = pd.DataFrame({"Cumulative Value" : portfolio_values})

    df['daily_return'] = df['Cumulative Value'].pct_change()
    df['daily_return'] = df['daily_return'].dropna()

    # Annualized Return
    annualized_return = (1 + df['daily_return'].mean())**252 - 1

    # Annualized Std
    annualized_std = df['daily_return'].std() * np.sqrt(252)

    # Sharpe Ratio
    sharpe_ratio = annualized_return / annualized_std


    # Arithmetic Return
    arithmetic_mean_return = df['daily_return'].mean()
    arithmetic_return = arithmetic_mean_return * 252

    # Geometric Return
    geometric_mean_return = (df['Cumulative Value'].iloc[-1] / df['Cumulative Value'].iloc[0]) ** (1 / (len(df) - 1)) - 1
    geometric_return = geometric_mean_return * 252

    # Maximum Drawdown
    max_drawdown = -(df['Cumulative Value'] / df['Cumulative Value'].cummax() - 1).min()

    # Annualized Skewness
    daily_skewness = skew(df['daily_return'].dropna())
    annualized_skewness = daily_skewness * np.sqrt(252)

    # Annualized Kurtosis
    daily_kurtosis = kurtosis(df['daily_return'].dropna(), fisher=False)
    annualized_kurtosis = daily_kurtosis # * np.sqrt(252)

    # Cumulative Return
    cumulative_return = (df['Cumulative Value'].iloc[-1] / df['Cumulative Value'].iloc[0]) - 1

    # Monthly 95% VaR
    monthly_var = -df['daily_return'].quantile(0.05)
    
    wealth = df['Cumulative Value'].iloc[-1]

    performance_summary = pd.DataFrame({
        'Metric': ['Annualized Return', 'Annualized Std', 'Sharpe Ratio', 'Arithmetic Return', 
                   'Geometric Return', 'Maximum Drawdown', 'Annualized Skewness', 
                   'Annualized Kurtosis', 'Cumulative Return', 'Monthly 95% VaR', 'Wealth'],
        value_name: [annualized_return, annualized_std, sharpe_ratio, arithmetic_return, 
                  geometric_return, max_drawdown, annualized_skewness, 
                  annualized_kurtosis, cumulative_return, monthly_var, wealth]
    })
    return performance_summary



########## New one
def group_data_by_month_new(preds, trues, w, s, y_mark, dfl_preds, w_star, s_star):
    preds_flat = preds.reshape(-1, preds.shape[-1])
    trues_flat = trues.reshape(-1, trues.shape[-1])
    dfl_preds_flat = dfl_preds.reshape(-1, dfl_preds.shape[-1])

    y_mark_flat = y_mark.reshape(-1, y_mark.shape[-1])
    y_mark_flat = y_mark_flat[:, :3].astype(int)

    dates = pd.to_datetime({
        'year': y_mark_flat[:, 0],
        'month': y_mark_flat[:, 1],
        'day': y_mark_flat[:, 2]
    })

    df = pd.DataFrame({
        'date': dates,
        'year': dates.dt.year,
        'month': dates.dt.month,
        'day': dates.dt.day
    })
    df_preds = pd.concat([df, pd.DataFrame(preds_flat)], axis=1)
    df_trues = pd.concat([df, pd.DataFrame(trues_flat)], axis=1)
    df_dfl_preds = pd.concat([df, pd.DataFrame(dfl_preds_flat)], axis=1)

    df_preds = df_preds.drop_duplicates(subset=['date'], keep='first').reset_index(drop=True)
    df_trues = df_trues.drop_duplicates(subset=['date'], keep='first').reset_index(drop=True)
    df_dfl_preds = df_dfl_preds.drop_duplicates(subset=['date'], keep='first').reset_index(drop=True)

    w_flat = w.reshape(-1, w.shape[-1])
    s_flat = s.reshape(-1, s.shape[-1])
    w_star_flat = w_star.reshape(-1, w_star.shape[-1])
    s_star_flat = s_star.reshape(-1, s_star.shape[-1])

    date_df = df_preds.iloc[:-preds.shape[2]+1, :4]
    date_df['reb'] = date_df.groupby(['year', 'month'])['date'].transform(lambda x: (x == x.min()).astype(int))

    df_w  = pd.concat( [date_df, pd.DataFrame(w_flat)], axis = 1)
    df_s  = pd.concat( [date_df, pd.DataFrame(s_flat)], axis = 1)

    df_w_star  = pd.concat( [date_df, pd.DataFrame(w_star_flat)], axis = 1)
    df_s_star  = pd.concat( [date_df, pd.DataFrame(s_star_flat)], axis = 1)

    #df_w_r = df_w[df_w["reb"] == 1].reset_index(drop = True)
    #df_s_r = df_s[df_s["reb"] == 1].reset_index(drop = True)
    #df_w_star_r = df_w_star[df_w_star["reb"] == 1].reset_index(drop = True)
    #df_s_star_r = df_s_star[df_s_star["reb"] == 1].reset_index(drop = True)
    
    return df_preds, df_trues, df_w, df_s, df_w_star, df_s_star



def calculate_portfolio_values_new(df_w_r, dates, returns, transaction_cost=0.01, initial_value=1.0):
    
    portfolio_values = [initial_value]
    turnover_list = []
    df_w_r = df_w_r[df_w_r["reb"] == 1].reset_index(drop = True)


    for i in range(df_w_r.shape[0]-1):
        current_weights = np.array(df_w_r.iloc[i, 5:], dtype = float)
        current_weights = np.round(current_weights, 4)
        currendt_date = str(df_w_r.iloc[i,0]).split(" ")[0]

        next_weights = np.array(df_w_r.iloc[i+1, 5:], dtype = float)
        next_weights = np.round(next_weights, 4)
        rebalance_date = str(df_w_r.iloc[i+1,0]).split(" ")[0]

        start_idx = np.where(dates== pd.Timestamp(currendt_date))[0][0]
        end_idx = np.where(dates== pd.Timestamp(rebalance_date))[0][0]
        period_returns = returns[start_idx:end_idx]
        
        for j in range(len(period_returns)):
            portfolio_return = np.dot(current_weights, period_returns[j])
            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))

        turnover = np.sum(np.abs(next_weights - current_weights)) / 2
        turnover_list.append(turnover)
        transaction_costs = turnover * transaction_cost
        portfolio_values[-1] *= (1 - transaction_costs)
    
    final_weights = np.array(df_w_r.iloc[-1, 5:], dtype = float)
    final_weights = np.round(final_weights, 4)
    final_date= str(df_w_r.iloc[-1,0]).split(" ")[0]
    
    final_idx = np.where(dates == pd.Timestamp(final_date))[0][0]
    final_returns = returns[final_idx+1:]
    
    for k in range(len(final_returns)):
        portfolio_return = np.dot(final_weights, final_returns[k])
        portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
        
    return portfolio_values, turnover_list
