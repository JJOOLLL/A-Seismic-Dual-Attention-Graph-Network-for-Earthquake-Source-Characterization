# A-Seismic-Dual-Attention-Graph-Network-for-Earthquake-Source-Characterization

This repository contains the code and data processing scripts for the paper **“A Seismic Dual-Attention Graph Network for Earthquake Source Characterization.”**  
The study proposes a dual-attention graph neural network that integrates spatial and temporal dependencies in seismic data to improve earthquake source localization and parameter estimation.

## Files

- `download_data.ipynb` — notebook for downloading and preprocessing seismic data  
- `SAGE_model.py` — implementation of the Seismic Dual-Attention Graph Network (SAGE) and dataloader  
- `train_model.ipynb` — notebook for training and evaluation  

## Quick Start

1. Run `download_data.ipynb` to download and preprocess the raw seismic data.  
2. After the data is prepared, open `train_model.ipynb` to train and evaluate the model.

## Requirements

Tested on **Python 3.9+** with the following packages:

```bash
torch >= 2.0.0
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
tqdm
scikit-learn >= 1.0
