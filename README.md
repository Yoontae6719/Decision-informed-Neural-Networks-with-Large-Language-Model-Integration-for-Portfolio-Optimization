# DINN (Decision-informed-Neural-Networks-with-Large-Language-Model-Integration-for-Portfolio-Optimization)

This is the origin Pytorch implementation of DINN in the following paper: Decision-informed Neural Networks with Large Language Model Integration for Portfolio Optimization.
 
🚩**News**(Jan 11, 2025)  We have released DINN.

### Installation

```bash
# Simple installation packages
pip install -r requirements.txt
```

### How to run?
```git clone https://github.com/Yoontae6719/Decision-informed-Neural-Networks-with-Large-Language-Model-Integration-for-Portfolio-Optimization.git
bash ./runfile/SNP.sh # Get model weight (Balanced settings)
bash ./runfile/DOW.sh # Get model weight (Balanced settings)

bash ./runfile/SNP_ALL.sh # Get model weight (highly aggressive, aggressive, Balanced, conservative and highly conservative settings)
bash ./runfile/DOW_ALL.sh # Get model weight (highly aggressive, aggressive, Balanced, conservative and highly conservative settings)

python anaysis.py     # Senstivity analysis 
python anaysis_dow.py # Senstivity analysis
```

### Dataset
- You can download via WRDS. (Check 0_Get_dataset_using_WRDS.ipynb)
- We added the macro variable (foucsd on Composite Leading Indicators Index) such as 
    - ICSA : https://fred.stlouisfed.org/series/ICSA
    - UMCSENT : https://fred.stlouisfed.org/series/UMCSENT
    - HSN1F : https://fred.stlouisfed.org/series/HSN1F
    - UNRATE : https://fred.stlouisfed.org/series/UNRATE
    - HIgh-Yield Bond Spread : https://fred.stlouisfed.org/series/BAMLH0A0HYM2
 

### What is DINN?
will be written


### 📚 Citation

```bibtex
will be write
```
### Acknowledgement
- [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://github.com/KimMeen/Time-LLM/tree/main)
- [thuml_LTS](https://github.com/thuml/Large-Time-Series-Model)
- [thuml_TS](https://github.com/thuml/Time-Series-Library)
- [Chronos: Pretrained Models for Probabilistic Time Series Forecasting](https://github.com/amazon-science/chronos-forecasting)

Built by [Yoontae Hwang](https://yoontae6719.github.io/) - Copyright (c) 2025 Yoontae Hwang
