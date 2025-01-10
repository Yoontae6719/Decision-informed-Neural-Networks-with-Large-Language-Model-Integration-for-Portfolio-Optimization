
import torch
import torch as t
import torch.nn as nn
import numpy as np
import pdb
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


def cvx_layer(mean_mat, cholesky_mat, lambda_):
    batch_size, n_assets = mean_mat.shape

    z = cp.Variable(n_assets)
    s = cp.Variable()
    mu = cp.Parameter((n_assets,))
    A = cp.Parameter((n_assets, n_assets))
    lambda_ = float(lambda_)

    objective = cp.Minimize(lambda_ * cp.square(s) - mu.T @ z)

    constraints = [
        cp.norm(A @ z) <= s,
        s >= 0,
        cp.sum(z) == 1,
        z >= 0,
        z <= 1
    ]

    problem = cp.Problem(objective, constraints)
    cvxpylayer = CvxpyLayer(problem, parameters=[mu, A], variables=[z, s])


    z_star, s_star = cvxpylayer(mean_mat, cholesky_mat, solver_args={'solve_method': 'SCS'})
    return z_star, s_star

def compute_mean_cholesky(returns_batch, history):
    batch_size, pred_len, n_assets = returns_batch.shape
    epsilon = 1e-6

    mean_list = []
    cholesky_list = []
    cov_list = [] 

    for i in range(batch_size):
        returns_sample = returns_batch[i]  
        history_sample = history[i]

        mean = returns_sample.mean(dim=0)
        mean_list.append(mean)

        history_d = torch.cat([history_sample, returns_sample], dim=0)
        returns_transposed = history_d.T
        cov_matrix = torch.cov(returns_transposed)
        cov_list.append(cov_matrix)
        cov_matrix += epsilon * torch.eye(n_assets, device=returns_batch.device)
        cholesky = torch.linalg.cholesky(cov_matrix)
        cholesky_list.append(cholesky)

    mean_mat = torch.stack(mean_list)
    cholesky_mat = torch.stack(cholesky_list)
    cov_mat = torch.stack(cov_list)
    
    return mean_mat, cholesky_mat, cov_mat

def get_w(batch_x, batch_y, history, lambda_, ):
    mean_mat_pred, cholesky_mat_pred, cov_mat_pred = compute_mean_cholesky(batch_x, history)
    mean_mat_true, cholesky_mat_true, cov_mat_true = compute_mean_cholesky(batch_y, history)

    w_star_true, s_star_true = cvx_layer(mean_mat_true, cholesky_mat_true, lambda_)
    w_star_pred, s_star_pred = cvx_layer(mean_mat_pred, cholesky_mat_pred, lambda_)
    return w_star_pred, w_star_true

def compute_dfl_loss(batch_x, batch_y, history, lambda_, loss_style='mae'):
    mean_mat_pred, cholesky_mat_pred, cov_mat_pred = compute_mean_cholesky(batch_x, history)
    mean_mat_true, cholesky_mat_true, cov_mat_true = compute_mean_cholesky(batch_y, history)

    z_star_true, s_star_true = cvx_layer(mean_mat_true, cholesky_mat_true, lambda_)
    z_star_pred, s_star_pred = cvx_layer(mean_mat_pred, cholesky_mat_pred, lambda_)

    true_risk = lambda_ * torch.norm(torch.bmm(cholesky_mat_true, z_star_true.unsqueeze(2)), dim=1).squeeze() ** 2
    true_return = (mean_mat_true * z_star_true).sum(dim=1)

    pred_risk = lambda_ * torch.norm(torch.bmm(cholesky_mat_true, z_star_pred.unsqueeze(2)), dim=1).squeeze() ** 2
    pred_return = (mean_mat_true * z_star_pred).sum(dim=1)

    first_term  = true_risk - true_return
    second_term = pred_risk - pred_return

    # 손실 계산
    if loss_style == 'difference':
        loss = (first_term - second_term).mean()
    elif loss_style == 'mse':
        loss = ((first_term - second_term)**2).mean()
    elif loss_style == 'mae':
        loss = torch.abs(first_term - second_term).mean()
    else:
        raise ValueError(f"Invalid loss_style '{loss_style}'. Expected 'difference', 'mse', or 'mae'.")

    return loss









def cpu_cvx_layer(mean_mat, cholesky_mat, lambda_):
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer

    original_device = mean_mat.device

    mean_mat_cpu = mean_mat.detach().cpu()
    cholesky_mat_cpu = cholesky_mat.detach().cpu()
    
    batch_size, n_assets = mean_mat_cpu.shape

    z = cp.Variable(n_assets)
    s = cp.Variable()
    mu = cp.Parameter(n_assets)
    A = cp.Parameter((n_assets, n_assets))
    lambda_ = float(lambda_)

    objective = cp.Minimize(lambda_ * cp.square(s) - mu.T @ z)

    constraints = [
        cp.norm(A @ z) <= s,
        s >= 0,
        cp.sum(z) == 1,
        z >= 0,
        z <= 1
    ]
    
    problem = cp.Problem(objective, constraints)
    cvxpylayer = CvxpyLayer(problem, parameters=[mu, A], variables=[z, s])

    z_star_list = []
    s_star_list = []
    
    for i in range(batch_size):
        mu_i = mean_mat_cpu[i]
        A_i = cholesky_mat_cpu[i]
        z_star_i, s_i = cvxpylayer(mu_i, A_i, solver_args={'solve_method': 'SCS'})
        z_star_list.append(z_star_i)
        s_star_list.append(s_i)

    z_star = torch.stack(z_star_list)
    s_star = torch.stack(s_star_list)

    z_star = z_star.to(original_device)
    s_star = s_star.to(original_device)

    return z_star, s_star

def cpu_compute_mean_cholesky(returns_batch, history):
    original_device = returns_batch.device

    returns_batch_cpu = returns_batch.detach().cpu()
    history_cpu = history.detach().cpu()

    batch_size, pred_len, n_assets = returns_batch_cpu.shape
    epsilon = 1e-6  

    mean_list = []
    cholesky_list = []
    cov_list = [] 

    for i in range(batch_size):
        returns_sample = returns_batch_cpu[i]
        history_sample = history_cpu[i]
        mean = returns_sample.mean(dim=0)
        mean_list.append(mean)
        
        history_d = torch.cat([history_sample, returns_sample], dim=0)
        returns_transposed = history_d.T
        cov_matrix = torch.cov(returns_transposed)
        cov_matrix += epsilon * torch.eye(n_assets)
        #cov_list.append(cov_matrix)
        
        cholesky = torch.linalg.cholesky(cov_matrix)
        cholesky_list.append(cholesky)

    mean_mat = torch.stack(mean_list)
    cholesky_mat = torch.stack(cholesky_list)

    mean_mat = mean_mat.to(original_device)
    cholesky_mat = cholesky_mat.to(original_device)
    
    cov_mat = torch.stack(cov_list)
    cov_mat = cov_mat.to(original_device)

    return mean_mat, cholesky_mat, cov_matrix

def cpu_compute_dfl_loss(batch_x, batch_y, history, lambda_, loss_style='mae'):
    original_device = batch_x.device

    batch_x = batch_x.to(original_device)
    batch_y = batch_y.to(original_device)
    history = history.to(original_device)
    
    mean_mat_pred, cholesky_mat_pred, cov_mat_pred = compute_mean_cholesky(batch_x, history)
    mean_mat_true, cholesky_mat_true, cov_mat_true = compute_mean_cholesky(batch_y, history)

    z_star_true, s_star_true = cvx_layer(mean_mat_true, cholesky_mat_true, lambda_)
    z_star_pred, s_star_pred = cvx_layer(mean_mat_pred, cholesky_mat_pred, lambda_)
    # Compute risks using z^T Σ z
    # Reshape z_star to match matrix multiplication dimensions
    z_star_true_unsqueezed = z_star_true.unsqueeze(1)  # [batch_size, 1, n_assets]
    z_star_pred_unsqueezed = z_star_pred.unsqueeze(1)  # [batch_size, 1, n_assets]
    
    true_risk = lambda_ * torch.bmm(
        torch.bmm(z_star_true_unsqueezed, cov_mat_true),
        z_star_true_unsqueezed.transpose(1, 2)
    ).squeeze()

    pred_risk = lambda_ * torch.bmm(
        torch.bmm(z_star_pred_unsqueezed, cov_mat_true),
        z_star_pred_unsqueezed.transpose(1, 2)
    ).squeeze()
    
    
    true_return = (mean_mat_true * z_star_true).sum(dim=1).unsqueeze(1)
    pred_return = (mean_mat_true * z_star_pred).sum(dim=1).unsqueeze(1)
    
    first_term  = true_risk - true_return
    second_term = pred_risk - pred_return
    
    
    if loss_style == 'difference':
        loss = first_term - second_term
        loss = loss.mean()
    elif loss_style == 'mse':
        loss = (first_term - second_term) ** 2
        loss = loss.mean()
    elif loss_style == 'mae':
        loss = torch.abs(first_term - second_term)
        loss = loss.mean()
    else:
        raise ValueError(f"Invalid loss_style '{loss_style}'. Expected 'difference', 'mse', or 'mae'.")
        
    return loss, z_star_pred, s_star_pred, z_star_true, s_star_true



def cpu_compute_analysis_cholesky(returns_batch, history):
    original_device = returns_batch.device

    returns_batch_cpu = returns_batch.detach().cpu()
    history_cpu = history.detach().cpu()

    batch_size, pred_len, n_assets = returns_batch_cpu.shape
    epsilon = 1e-6  

    mean_list = []
    cholesky_list = []
    cov_list = [] 

    for i in range(batch_size):
        returns_sample = returns_batch_cpu[i]
        history_sample = history_cpu[i]
        mean = returns_sample.mean(dim=0)
        mean_list.append(mean)
        
        history_d = torch.cat([history_sample, returns_sample], dim=0)
        returns_transposed = history_d.T
        cov_matrix = torch.cov(returns_transposed)
        cov_matrix += epsilon * torch.eye(n_assets)
        cov_list.append(cov_matrix)
        
        cholesky = torch.linalg.cholesky(cov_matrix)
        cholesky_list.append(cholesky)

    mean_mat = torch.stack(mean_list)
    cholesky_mat = torch.stack(cholesky_list)

    mean_mat = mean_mat.to(original_device)
    cholesky_mat = cholesky_mat.to(original_device)
    
    cov_mat = torch.stack(cov_list)
    cov_mat = cov_mat.to(original_device)

    return mean_mat, cholesky_mat, cov_matrix

def cpu_compute_dfl_analysis(batch_x, batch_y, history, lambda_, loss_style='mae'):
    original_device = batch_x.device

    batch_x = batch_x.to(original_device)
    batch_y = batch_y.to(original_device)
    history = history.to(original_device)
    
    mean_mat_pred, cholesky_mat_pred, cov_matrix_pred = cpu_compute_analysis_cholesky(batch_x, history)
    mean_mat_true, cholesky_mat_true, cov_matrix_true = cpu_compute_analysis_cholesky(batch_y, history)

    z_star_true, s_star_true = cvx_layer(mean_mat_true, cholesky_mat_true, lambda_)
    z_star_pred, s_star_pred = cvx_layer(mean_mat_pred, cholesky_mat_pred, lambda_)    

    return z_star_pred, s_star_pred, z_star_true, s_star_true, cov_matrix_pred, cov_matrix_true, batch_x, batch_y


