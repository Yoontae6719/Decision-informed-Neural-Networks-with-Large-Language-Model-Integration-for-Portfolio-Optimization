# DINN (Decision-informed-Neural-Networks-with-Large-Language-Model-Integration-for-Portfolio-Optimization)

This is the origin Pytorch implementation of DINN in the following paper: Decision-informed Neural Networks with Large Language Model Integration for Portfolio Optimization.
 
🚩**News**(Jan 01, 2025)  We will release DINN.


### Installation

```bash
# Simple installation packages
pip install -r requirements.txt
```

### How to run?
```git clone https://github.com/Yoontae6719/Decision-informed-Neural-Networks-with-Large-Language-Model-Integration-for-Portfolio-Optimization.git
bash ./runfile/SNP.sh # Get model weight (Balanced settings)
bash ./runfile/DOW.sh # Get model weight (Balanced settings)

bash ./runfile/SNP_ALL.sh # Get model weight (highly aggressive, aggressive, conservative and highly conservative settings)
bash ./runfile/DOW_ALL.sh # Get model weight (highly aggressive, aggressive, conservative and highly conservative settings)

bash ./runfile/SNP_Lambda.sh # Get model weight (loss function weight)
bash ./runfile/DOW_Lambda.sh # Get model weight (loss function weight)

python anaysis.py     # Senstivity analysis 
python anaysis_dow.py # Senstivity analysis
```

### Dataset
if you request the dataset, we can provide. or you can download via WRDS.


### What is DINN?
will be written






### 📚 Citation

```bibtex
will be write
```

Built by [Yoontae Hwang](https://yoontae6719.github.io/) - Copyright (c) 2025 Yoontae Hwang
