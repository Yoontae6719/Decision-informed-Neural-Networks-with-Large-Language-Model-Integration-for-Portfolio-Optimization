# DINN (Decision-informed-Neural-Networks-with-Large-Language-Model-Integration-for-Portfolio-Optimization)

This is the origin Pytorch implementation of DINN in the following paper: Decision-informed Neural Networks with Large Language Model Integration for Portfolio Optimization.
 
🚩**News**(Jan 11, 2025)  We have released DINN.

### Installation

```bash
# Simple installation packages
pip install -r requirements.txt
```

### How to run?
```
Note that : Depending on the batch size, the optimal performance requires a 96GB GPU based on the current version. 


git clone https://github.com/Yoontae6719/Decision-informed-Neural-Networks-with-Large-Language-Model-Integration-for-Portfolio-Optimization.git
bash ./runfile/SNP.sh # Get model weight (Balanced settings)
bash ./runfile/DOW.sh # Get model weight (Balanced settings)

bash ./runfile/SNP_ALL.sh # Get model weight (highly aggressive, aggressive, Balanced, conservative and highly conservative settings)
bash ./runfile/DOW_ALL.sh # Get model weight (highly aggressive, aggressive, Balanced, conservative and highly conservative settings)

python anaysis.py     # Senstivity analysis 
python anaysis_dow.py # Senstivity analysis
```

### Dataset
- You can download via WRDS. (Check 0_Get_dataset_using_WRDS.ipynb)
    - You need to WRDS account for download dataset !!
- We added the macro variable (foucsd on Composite Leading Indicators Index) such as 
    - ICSA : https://fred.stlouisfed.org/series/ICSA
    - UMCSENT : https://fred.stlouisfed.org/series/UMCSENT
    - HSN1F : https://fred.stlouisfed.org/series/HSN1F
    - UNRATE : https://fred.stlouisfed.org/series/UNRATE
    - HIgh-Yield Bond Spread : https://fred.stlouisfed.org/series/BAMLH0A0HYM2
 - Note that : If you have any questions, please send them to the email address below.
    - Will be write  
   
### What is DINN?
This paper tackles the persistent disconnect between predictive modeling and actual decision quality in portfolio optimization. Although recent developments in deep learning and large language models (LLMs) have shown promise in forecasting financial time series, traditional approaches still treat prediction and optimization as separate tasks, often leading to suboptimal allocations when small forecast errors are magnified by downstream decision-making. To address this, we propose a \textit{decision-informed neural network} (DINN) framework that integrates LLM-derived embeddings directly into the portfolio optimization process via a differentiable, convex optimization layer. Our approach attends selectively to cross-sectional relationships, temporal trends, and macroeconomic factors, yielding a richer feature space without overwhelming the model with noise. Crucially, DINN employs a hybrid loss function—balancing prediction accuracy with a decision-focused objective—that enables end-to-end training. Extensive experiments on equity datasets (S\&P 100 and DOW 30) reveal that DINN significantly improves both return generation and risk-adjusted performance compared to conventional deep learning baselines. Prob-sparse attention mechanisms identify a subset of “high-impact” assets, enabling robust allocations under diverse market regimes. Also, gradient-based sensitivity analyses further show that DINN assigns greater learning capacity to assets most critical for decision making, thus mitigating the impact of prediction errors on portfolio performance. These findings emphasize the importance of embedding financial decision-making objectives within model training rather than merely optimizing for statistical accuracy, and represent the potential of decision-focused learning frameworks in advancing robust, context-aware portfolio management.


### 📚 Citation

```bibtex
will be write
```


### Acknowledgement
- [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://github.com/KimMeen/Time-LLM/tree/main)
- [thuml_LTS](https://github.com/thuml/Large-Time-Series-Model)
- [thuml_TS](https://github.com/thuml/Time-Series-Library)
- [Chronos: Pretrained Models for Probabilistic Time Series Forecasting](https://github.com/amazon-science/chronos-forecasting)

Built by [Anonymous](https://yoontae6719.github.io/) - Copyright (c) 2025 Anonymous
