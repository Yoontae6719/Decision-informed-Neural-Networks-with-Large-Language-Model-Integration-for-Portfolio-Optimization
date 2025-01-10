import numpy as np
import torch

def compute_optimal_weights(Sigma: np.ndarray, mu: np.ndarray):
    inv_Sigma = np.linalg.inv(Sigma)
    ones = np.ones_like(mu)
    denom = ones @ inv_Sigma @ ones    
    numerator = (ones @ inv_Sigma @ mu) - 1.0
    w = inv_Sigma @ mu - (numerator / denom) * (inv_Sigma @ ones)
    return w

def gradient_wrt_mu(Sigma: np.ndarray):
    """
    Theorem 1 
    """
    inv_Sigma = np.linalg.inv(Sigma)
    ones = np.ones((Sigma.shape[0], 1))
    
    denom = ones.T @ inv_Sigma @ ones  # scalar
    grad_matrix = inv_Sigma - (inv_Sigma @ ones @ ones.T @ inv_Sigma) / denom
    return grad_matrix



def measure_sensitivity_both_ends(grad_w_mu: np.ndarray, top_k: int = 5, bottom_k: int = 5):
    abs_matrix = np.abs(grad_w_mu)
    scores = abs_matrix.sum(axis=1)  # shape: (N,)

    idx_asc = np.argsort(scores)  # 작은 순부터 정렬
    bottom_indices = idx_asc[:bottom_k]

    idx_desc = np.argsort(-scores)  # 큰 순부터 정렬
    top_indices = idx_desc[:top_k]

    return scores, top_indices, bottom_indices

def partial_p_wrt_Sigma(mu: torch.Tensor, Sigma: torch.Tensor):
    r"""
    Theorem 2 : Partial derivation -> p = (g - 1)/z.
      g = 1^T Σ^{-1} μ
      z = 1^T Σ^{-1} 1

    d p / d Σ = [ -Σ^{-1} 1 μ^T Σ^{-1} * z + (g - 1) Σ^{-1} 1 1^T Σ^{-1} ] / z^2
      (shape: (N,N))
    """
    N = mu.shape[0]
    device = mu.device
    ones = torch.ones(N, dtype=mu.dtype, device=mu.device)
    Sigma_inv = torch.inverse(Sigma)
    g = ones @ Sigma_inv @ mu
    z = ones @ Sigma_inv @ ones

    # - Σ^{-1} 1 μ^T Σ^{-1} * z
    minus_1_muT = - (Sigma_inv @ ones).unsqueeze(-1) @ (Sigma_inv @ mu).unsqueeze(0)
    part_a = minus_1_muT * z

    # + (g - 1) Σ^{-1} 1 1^T Σ^{-1}
    part_b = (g - 1.0) * torch.ger(Sigma_inv @ ones, Sigma_inv @ ones)

    dp_dSigma = (part_a + part_b) / (z**2)  # shape (N,N)

    return dp_dSigma



def dw_dSigma_closed_form(mu: torch.Tensor, Sigma: torch.Tensor):
    r"""
    Theorem 2 : closed form solution  1
      d w / d Σ =  -Σ^{-1}(μ - p·1)Σ^{-1} - Σ^{-1} ( (dp/dΣ) * 1 ).
    """
    N = mu.shape[0]
    Sigma_inv = torch.inverse(Sigma)
    ones = torch.ones(N, dtype=mu.dtype, device=mu.device)
    g = ones @ Sigma_inv @ mu
    z = ones @ Sigma_inv @ ones
    p = (g - 1.0) / z

    dp_dSigma = partial_p_wrt_Sigma(mu, Sigma)  # (N,N)

    base_vec = Sigma_inv @ (mu - p*ones)   # shape (N,)
    base_mat = -torch.outer(base_vec, Sigma_inv @ ones)  # shape (N,N)

    dwdSigma = torch.zeros(N, N, N, device=mu.device, dtype=mu.dtype)

    for a in range(N):
        for b in range(N):
            # 이부분 좀더 효율적으로 짜기위해서 deltaSigma_{a,b}에서  각 w_k 변화를 얻는식으로 진행함.
            
            Ndim = mu.shape[0]
            deltaSigma = torch.zeros(Ndim, Ndim, device=mu.device, dtype=mu.dtype)
            deltaSigma[a,b] = 1.0  # (a,b) 위치만 1
                        
            dw_vec = apply_dw_dSigma(mu, Sigma, deltaSigma)
            # dw_vec: shape(N,) = [dw_0, dw_1, ..., dw_{N-1}].
            
            dwdSigma[:, a, b] = dw_vec

    return dwdSigma



def apply_dw_dSigma(mu, Sigma, dSigma):
    r"""
    Theorem 2 Final
      dw/dΣ = - Σ^{-1} (μ - p·1) Σ^{-1}  - Σ^{-1} ( (dp/dΣ) 1 ),
    에서, "어떤 소량 dΣ"가 주어졌을 때의 w 변화량을 편의상  1차 근사로 계산. -> 결국 솔루션은 논문수식이랑 같음    
    """
    N = mu.shape[0]
    ones = torch.ones(N, dtype=mu.dtype, device=mu.device)
    Sigma_inv = torch.inverse(Sigma)

    # p
    g = ones @ Sigma_inv @ mu
    z = ones @ Sigma_inv @ ones
    p = (g - 1.0) / z

    # dp/dSigma (N,N)
    dp_dSigma = partial_p_wrt_Sigma(mu, Sigma)

    mu_minus_p1 = mu - p*ones   # shape(N,)
    tmp_vec = Sigma_inv @ mu_minus_p1  # shape(N,)
    first_term = - Sigma_inv @ ( dSigma @ tmp_vec )    
    dp_scalar = torch.sum(dp_dSigma * dSigma)  # shape: scalar
    
    second_term = - Sigma_inv @ ( dp_scalar * ones )   # shape(N,)
    
    dw_vec = first_term + second_term
    return dw_vec


def grad_wrt_cholesky_full(mu: torch.Tensor, L: torch.Tensor):

    N = mu.shape[0]
    Sigma = L @ L.T  # (N,N)

    dwdSigma = dw_dSigma_closed_form(mu, Sigma)  # (N, N, N)

    dwdL = torch.zeros(N, N, N, dtype=mu.dtype, device=mu.device)

    for i in range(N):
        for j in range(N):
            dL = torch.zeros_like(L)
            dL[i,j] = 1.0

            dSigma_ij = dL @ L.T + L @ dL.T  # (N,N)

            
            for k in range(N):
                dwdL[k,i,j] = torch.sum(dwdSigma[k] * dSigma_ij)
            
    return dwdL

def find_top_bottom_assets(dwdL_3d, top_k_list=[5, 10, 20]):


    N = dwdL_3d.shape[0]

    #각 자산별로 그라디언트 norm 계산
    grad_norm = dwdL_3d.abs().sum(dim=(1,2))  # (N,)

    sorted_values_desc, indices_desc = torch.sort(grad_norm, descending=True)
    sorted_values_asc, indices_asc = torch.sort(grad_norm, descending=False)

    result = {
        'top_k': {},
        'bottom_k': {}
    }
    for k_val in top_k_list:
        top_indices = indices_desc[:k_val].tolist()   
        bottom_indices = indices_asc[:k_val].tolist()

        top_indices_sorted = sorted(top_indices)
        bottom_indices_sorted = sorted(bottom_indices)

        result['top_k'][k_val] = top_indices_sorted
        result['bottom_k'][k_val] = bottom_indices_sorted

    return result

def measure_sensitivity_both_ends(grad_w_mu: np.ndarray, top_k_list=[5, 10, 20]):
    abs_matrix = np.abs(grad_w_mu)
    scores = abs_matrix.sum(axis=1)  # shape: (N,)

    idx_asc = np.argsort(scores)     # 작은 순
    idx_desc = np.argsort(-scores)   # 큰 순

    result = {
        'scores': scores,
        'top_k': {},
        'bottom_k': {}
    }

    for k_val in top_k_list:
        top_indices = idx_desc[:k_val]
        bottom_indices = idx_asc[:k_val]

        top_indices_sorted = np.sort(top_indices)
        bottom_indices_sorted = np.sort(bottom_indices)

        result['top_k'][k_val] = top_indices_sorted.tolist()
        result['bottom_k'][k_val] = bottom_indices_sorted.tolist()

    return result
