from math import sqrt
import itertools

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
from layers.Embed import DataEmbedding_wo_pos
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi
from layers.cross import CrossAttns

import scipy.stats as stats 
from layers.View import user_view_, user_negative_view_
from einops import rearrange
transformers.logging.set_verbosity_error()
import random
import json
from itertools import combinations




class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_llm = configs.llm_dim
        self.enc_in = configs.enc_in
        self.lambda_value = configs.lambda_value

        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False
            
        self.price = configs.is_price
        self.enc_in = configs.enc_in
        
        # Step 1.1 Revin and Decomposition
        kernel_size = configs.moving_avg # add argument
        self.normalize_layers = Normalize(configs.enc_in, affine=False) # True
        self.after_normalize_layers = Normalize(configs.enc_in, affine=False) #True
        
        kernel_sizes = [3, 5, 15, 21, 65]
        self.decomp = series_decomp_multi(kernel_sizes) #series_decomp_multi(kernel_size)
         
        
        
        # Ticker infomation
        with open(configs.ticker_dict_path, 'r') as file:
            self.ticker_dict = json.load(file)
            
        #self.ticker_dict = {int(k): v for k, v in self.ticker_dict.items()}
        
        with open(configs.sector_dict_path, 'r') as file:
            self.sector_dict = json.load(file)
            
        #self.sector_dict = {k: v for k, v in self.sector_dict.items()}
        
        self.d_ff = configs.d_ff


        self.linear_trend_out = nn.Linear(self.d_ff, self.pred_len )
        self.linear_resid_out = nn.Linear(self.d_ff, self.pred_len )
        self.linear_trend_out.weight = nn.Parameter((1/self.d_ff)*torch.ones([self.pred_len,self.d_ff]))
        self.linear_resid_out.weight = nn.Parameter((1/self.d_ff)*torch.ones([self.pred_len,self.d_ff]))     

        self.linear_full = nn.Linear(self.d_ff, self.pred_len )
        self.linear_full.weight = nn.Parameter((1/self.d_ff)*torch.ones([self.pred_len,self.d_ff]))
        
        self.cross_market = CrossAttns(input_dim=self.enc_in,
                                       seq_len = self.seq_len,
                                        hidden_dim=configs.num_hidden,
                                        num_heads=configs.num_heads,
                                        num_encoder_layers=configs.num_enc_l,
                                        d_llm=self.d_llm)

        
        self.cross_stocks = CrossAttns(input_dim=self.enc_in,
                                       seq_len = self.seq_len,
                                        hidden_dim=configs.num_hidden,
                                        num_heads=configs.num_heads,
                                        num_encoder_layers=configs.num_enc_l,
                                        d_llm=self.d_llm)
        
        self.sp = nn.Softplus()

        
        
    def forward(self, x_enc, seq_y, seq_x_mark, seq_y_mark, seq_x_mac,mask=None):
        dec_out, pre_pred_dfl_out, lambda_, attns,index_attns, emb = self.forecast(x_enc, seq_y, seq_x_mark, seq_y_mark, seq_x_mac)
        return dec_out[:, -self.pred_len:, :], pre_pred_dfl_out[:, -self.pred_len:, :] ,lambda_, attns, index_attns, emb
    
    def forecast(self, x_enc, seq_y, seq_x_mark, seq_y_mark, seq_x_mac):
        # Step 1.1. Revin and Decomposition
        B, S, D = x_enc.shape
        _, _, N_m = seq_x_mac.size()
        
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        
        # Step 1.2. get decomposed data
        resid_term, trend_term = self.decomp(x_enc) # resid_term, trend term : [Batch, Input length, Channel of stocks]  
        
        # Step 2.1.1. Stocks embeddings
        B, T, N = x_enc.size()
        stocks_prompt = self.get_stock_prompt(x_enc)
        market_prompt = self.get_market_prompt(seq_x_mac)
        
        trend_term = rearrange(trend_term, 'b l m -> b m l').to(torch.bfloat16)
        resid_term = rearrange(resid_term, 'b l m -> b m l').to(torch.bfloat16)
        
        _, dec_trend_in, attn_trend, index_trend = self.cross_market(trend_term, market_prompt) #output_trend_time
        _, dec_resid_in, attn_stock, index_stock = self.cross_stocks(resid_term, stocks_prompt) #output_resid_time
        

        # Step 3.2 Make dec
        dec_trend_out1 = self.llm_model(inputs_embeds= dec_trend_in).last_hidden_state
        dec_resid_out1 = self.llm_model(inputs_embeds= dec_resid_in).last_hidden_state
        
        dec_trend_out2 = dec_trend_out1[:, :, :self.d_ff]
        dec_resid_out2 = dec_resid_out1[:, :, :self.d_ff]
        
        pre_pred_out     = self.linear_full( dec_trend_out2 +  dec_resid_out2 )
        pre_pred_out     = rearrange(pre_pred_out, 'b m l -> b l m')        

        #pred_out = pre_pred_out * stdev + means
        pre_pred_out = pre_pred_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        pre_pred_out = pre_pred_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            
        lambda_ = self.preference_view(pre_pred_out) # ([1, 1])

        return pre_pred_out, pre_pred_out, lambda_, (attn_trend, attn_stock), (index_trend, index_stock), (dec_trend_out1, dec_resid_out1)
    
       
    def preference_view(self, out):
        user_preferences = ["HA", "A", "B", "C", "HC"] ##"high aggressive", "aggressive", "balanced", "conservative", "high 
        preference_values = [0.0145, 0.2656, 0.9545, 2.4305, 3.4623]

        if hasattr(self, 'lambda') and self.lambda_value in user_preferences:
            lambda_value = preference_values[user_preferences.index(self.lambda_value)]
        else:
            self.lambda_value = 'B'
            lambda_value = preference_values[user_preferences.index('B')]
        
        return torch.tensor(lambda_value, requires_grad=True).to(out.device)
    
    def compute_average_return(self, returns):
        average_return = returns.mean().item()
        return average_return

    def get_stock_prompt(self, x_enc):
        prompt = []
        all_pairs = list(combinations(range(self.enc_in), 2))
        if self.price == True:
            x_enc = (x_enc[:, 1:, :] - x_enc[:, :-1, :]) / (x_enc[:, :-1, :] )
        
        for b in range(x_enc.shape[0]):
            random_pairs = random.sample(all_pairs, 20)
            indices_in_pairs = set()
            comparison_str_list = []
            for i, j in random_pairs:
                indices_in_pairs.update([i, j])

                num_times_greater = (x_enc[b, :, i] > x_enc[b, :, j]).sum().item()
                num_times_less  = (x_enc[b, :, i] < x_enc[b, :, j]).sum().item()
                num_times_equal = self.seq_len - num_times_greater - num_times_less  

                stock_i = self.ticker_dict[str(i)]
                stock_j = self.ticker_dict[str(j)]
                sector_i = self.sector_dict[stock_i]
                sector_j = self.sector_dict[stock_j]

                if num_times_greater >= 127:
                    comparison_str = f"The returns of {stock_i} ({sector_i}) were greater than {stock_j} ({sector_j}) in {num_times_greater} out of {self.seq_len} time steps"
                elif num_times_greater <= 125:
                    comparison_str = f"The returns of {stock_i} ({sector_i}) were less than {stock_j} ({sector_j}) in {num_times_less} out of {self.seq_len} time steps"
                else:
                    comparison_str = f"The relative difference in returns between {stock_i} ({sector_i}) and {stock_j} ({sector_j}) was the same over {self.seq_len} time steps"
                comparison_str_list.append(comparison_str)

            column_mapping_list = []
            for idx in indices_in_pairs:
                stock = self.ticker_dict[str(idx)]
                sector = self.sector_dict[stock]
                column_mapping_list.append(f"Column {idx} corresponds to {stock} ({sector})")
            column_mapping_str = "; ".join(column_mapping_list)

            # Map sectors to their corresponding indices
            sector_to_indices = {}
            for idx in range(self.enc_in):
                stock = self.ticker_dict[str(idx)]
                sector = self.sector_dict[stock]
                sector_to_indices.setdefault(sector, []).append(idx)
                
            # Calculate average returns for each sector
            sector_returns = {}
            for sector, indices in sector_to_indices.items():
                returns = x_enc[b, :, indices]  # Shape: (seq_len, num_stocks_in_sector)
                mean_returns = returns.mean(dim=1)  # Shape: (seq_len,)
                average_return = self.compute_average_return(mean_returns)
                sector_returns[sector] = average_return
                
            # Format the sector returns
            sector_returns_list = [f"- {sector}: {sector_returns[sector] * 100:.2f}%" for sector in sector_returns]
            sector_returns_str = "\n".join(sector_returns_list)

            # Prepare formatted strings to avoid backslashes in f-string expressions
            comparison_str_formatted = "\n".join(comparison_str_list)


            prompt_ = ( "<|start_prompt|>\n"
                        "**Dataset Description:**\n"
                        f"The dataset consists of return data for {self.enc_in} stocks listed in the DOW 30. Each column corresponds to a specific stock, as detailed in the column mapping below.\n\n"
                        "**Column Mapping:**\n"
                        f"{column_mapping_str}\n\n"
                       
                       "**Task Description:**\n"
                        f"The task is to learn the relationships between assets over a period of {str(self.seq_len)} days to help predict future returns over the next {str(self.pred_len)} days.\n\n"
                        f"**Input Comparisons Over {str(self.seq_len)} Days:**\n"
                        f"{comparison_str_formatted}\n\n"
                        f"**Sector Average Returns Over {str(self.seq_len)} Days:**\n"
                        f"{sector_returns_str}\n"
                        "<|end_prompt|>"
                    )

            prompt.append(prompt_)

        stock_embeddings = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        #tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)
        stocks_prompt_embeddings = self.llm_model.get_input_embeddings()(stock_embeddings.input_ids.to(x_enc.device))
          
        #stocks_prompt_embeddings = stocks_prompt_embeddings.reshape(-1,self.d_llm)

        #stocks_prompt_embeddings = stocks_prompt_embeddings.mean(dim=(1))  # shape: [batch_size,d_llm]
        stocks_prompt_embeddings = stocks_prompt_embeddings#.reshape(-1,self.d_llm)

        return stocks_prompt_embeddings#.permute(0,2,1)
        
    def get_market_prompt(self, seq_x_mac):
        macro_variables = ['ICSA', 'UMCSENT', 'HSN1F', 'UNRATE', 'HYBS']
        positive_when_falling = {'ICSA', 'UNRATE', 'HYBS'}
        positive_when_rising = {'UMCSENT', 'HSN1F'}

        macro_prompts = []
        for b in range(seq_x_mac.shape[0]):
            macro_stats_list = []

            for m in range(seq_x_mac.shape[2]):
                macro_name = macro_variables[m]
                data = seq_x_mac[b, :, m]  

                min_value = data.min().item()
                max_value = data.max().item()
                median_value = data.median().item()
                mean_value = data.mean().item()
                std_dev = data.std(unbiased=False).item()   

                if data.numel() > 1:
                    diffs = data.diff()
                    trends = diffs.sum().item()
                    trend_direction = 'upward' if trends > 0 else 'downward'

                    top_k = 5
                    lags, lag_values = self.calculate_lags(data, top_k=top_k)
                    lags_values_str = ', '.join([f"{int(lags[i].item())}" for i in range(len(lags))])
                else:
                    trends = float('nan')
                    trend_direction = 'no trend'
                    lags_values_str = 'N/A'

                if macro_name in positive_when_falling:
                    direction = "more positive as it falls"
                elif macro_name in positive_when_rising:
                    direction = "more positive as it rises"
                else:
                    direction = "directional impact unspecified"

                stats_str = (
                    f"**{macro_name} ({direction}):**\n"
                    f"- Min: {min_value:.2f}\n"
                    f"- Max: {max_value:.2f}\n"
                    f"- Median: {median_value:.2f}\n"
                    f"- Mean: {mean_value:.2f}\n"
                    f"- Std Dev: {std_dev:.2f}\n"
                    f"- Trend: {trend_direction}\n"
                    f"- Top 5 Lags: {lags_values_str}"
                )

                macro_stats_list.append(stats_str)

            macro_stats_formatted = "\n\n".join(macro_stats_list)
            macro_prompt = (
                "<|start_prompt|>\n"
                "**Dataset Description:**\n"
                "The dataset includes macroeconomic variables that influence the stock market. These variables are:\n"
                f"- {', '.join(macro_variables)}\n\n"
                f"**Task Description:**\n"
                f"Analyze the characteristics of each macroeconomic variable over a period of {self.seq_len} days to understand their impact on the market.\n\n"
                "**Macro Variable Statistics:**\n"
                f"{macro_stats_formatted}\n"
                "<|end_prompt|>"
            )
            macro_prompts.append(macro_prompt)

        macro_prompts = self.tokenizer(
            macro_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ) 
        macro_prompt_embeddings = self.llm_model.get_input_embeddings()(
            macro_prompts.input_ids.to(seq_x_mac.device)
        )
        
        #macro_prompt_embeddings = macro_prompt_embeddings.reshape(-1, self.d_llm)
        #macro_prompt_embeddings = macro_prompt_embeddings.mean(dim=(1))  
        macro_prompt_embeddings = macro_prompt_embeddings#.reshape(-1, self.d_llm)

        return macro_prompt_embeddings#.permute(0,2,1)

        
    def calculate_lags(self, data, top_k=5):
        data = data - data.mean()
        q_fft = torch.fft.rfft(data)
        res = q_fft * torch.conj(q_fft)
        corr = torch.fft.irfft(res)
        corr = corr / corr[0]  
        mean_value = corr
        lags = torch.arange(1, len(mean_value),device=mean_value.device)
        mean_value = mean_value[1:]
        topk_values, topk_indices = torch.topk(mean_value, k=top_k)
        topk_lags = lags[topk_indices]
        return topk_lags, topk_values
