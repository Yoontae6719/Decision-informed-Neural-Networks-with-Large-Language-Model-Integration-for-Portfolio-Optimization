import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask

class ProbAttention(nn.Module): 
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Find the Top_k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            context = V.cumsum(dim=-2)
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            # Return attention scores
            return (context_in, attn)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # Add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # Get the context
        context = self._get_initial_context(values, L_Q)
        # Update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn, index

class CrossAttention_Prob(nn.Module):
    def __init__(self, hidden_dim, num_heads, factor=5, attention_dropout=0.1, output_attention=True):
        super(CrossAttention_Prob, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.factor = factor
        self.prob_attention = ProbAttention(mask_flag=False, factor=factor, attention_dropout=attention_dropout, output_attention=output_attention)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.output_attention = output_attention

    def forward(self, q, k, v, attn_mask=None):
        B, L_Q, _ = q.shape
        B, L_K, _ = k.shape
        H = self.num_heads
        D = self.head_dim

        q = self.W_q(q).view(B, L_Q, H, D)
        k = self.W_k(k).view(B, L_K, H, D)
        v = self.W_v(v).view(B, L_K, H, D)

        # Apply ProbAttention
        context, attn,index_data = self.prob_attention(q, k, v, attn_mask)  # context: [B, L_Q, H, D]
        context = context.contiguous().view(B, L_Q, H * D)  # [B, L_Q, hidden_dim]

        out = self.fc_out(context)  # [B, L_Q, hidden_dim]

        if self.output_attention:
            return out, attn, index_data  # attn: Attention scores
        else:
            return out

class CrossAttns(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=768, num_heads=16, num_encoder_layers=1, d_llm=None):
        super(CrossAttns, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(seq_len, hidden_dim)  # seq_len을 hidden_dim으로 매핑

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.cross_attention = CrossAttention_Prob(hidden_dim=hidden_dim, num_heads=num_heads, output_attention=True)

        if d_llm is not None and d_llm != hidden_dim:
            self.word_projection = nn.Linear(d_llm, hidden_dim)
        else:
            self.word_projection = nn.Identity()

        self.output_projection = nn.Linear(hidden_dim, d_llm)  # hidden_dim을 d_llm으로 매핑

    def forward(self, x, word_embedding):
        B = x.shape[0]
        # x: [B, input_dim, seq_len]

        if word_embedding.ndim == 2:
            word_embedding = word_embedding.unsqueeze(0).repeat(B, 1, 1)
        elif word_embedding.shape[0] != B:
            word_embedding = word_embedding[0].unsqueeze(0).repeat(B, 1, 1)

        # 필요하면 word_embedding을 프로젝션
        word_embedding = self.word_projection(word_embedding)  # [B, num_token, hidden_dim]

        # x의 seq_len 차원을 hidden_dim으로 매핑
        x = self.linear(x)  # x: [B, input_dim, hidden_dim]

        # Transformer Encoder에 입력하기 위해 차원 변경
        x = x.transpose(0, 1)  # x: [input_dim, B, hidden_dim]

        x = self.transformer_encoder(x)  # x: [input_dim, B, hidden_dim]

        x = x.transpose(0, 1)  # x: [B, input_dim, hidden_dim]

        x_time = x  # x_time: [B, input_dim, hidden_dim]

        q = x  # [B, input_dim, hidden_dim]

        k = v = word_embedding  # [B, num_token, hidden_dim]

        x_cross, attn_scores, index_data = self.cross_attention(q, k, v)  # x_cross: [B, input_dim, hidden_dim]

        # 출력 차원을 d_llm으로 매핑
        x_cross = self.output_projection(x_cross)  # x_cross: [B, input_dim, d_llm]

        return x_time, x_cross, attn_scores, index_data  # x_cross의 shape은 [B, input_dim, d_llm]

    
