# A-Seismic-Dual-Attention-Graph-Network-for-Earthquake-Source-Characterization

This repository contains the code and data processing scripts for the paper **“A Seismic Dual-Attention Graph Network for Earthquake Source Characterization.”**  
The study proposes a dual-attention graph neural network that integrates spatial and temporal dependencies in seismic data to improve earthquake source localization and parameter estimation.

## Files

- `download_data.ipynb` — notebook for downloading and preprocessing seismic data  
- `data_DA.zip` — preprocessed dataset equivalent to the output of `download_data.ipynb`  
  > You can **either** run `download_data.ipynb` to generate the dataset manually, **or** directly use `data_DA.zip` to skip the data download and preprocessing step.
- `SAGE_model.py` — implementation of the Seismic Dual-Attention Graph Network (SAGE) and dataloader  
- `train_model.ipynb` — notebook for training and evaluation  

## Quick Start

### Option 1: Use the preprocessed data
1. Unzip `data_DA.zip` in the project directory.  
2. Open `train_model.ipynb` and start training directly.

### Option 2: Generate the data manually
1. Run `download_data.ipynb` to download and preprocess the raw seismic data.  
2. Proceed to `train_model.ipynb` for model training and evaluation.

## Requirements

Tested on **Python 3.9+** with the following packages:

```bash
torch >= 2.0.0
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
tqdm
scikit-learn >= 1.0
