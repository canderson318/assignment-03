# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: cu-cpbs-7602
#     language: python
#     name: python3
# ---

# %%

import pandas as pd
import numpy as np  
import os 
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
# from joblib import Parallel, delayed
from typing import Optional, Sequence, Dict, Any


# %%
os.chdir((Path.home()/"wdpath.txt").read_text().strip())

# %% [markdown]
# ### Load Data

# %%
with open("processed-data/003-dat-dict.pkl",'rb') as f:
    dat_dict = pickle.load(f)


# %% [markdown]
# ### Clean Data

# %%
# filter data_dict for these obs

def filter_se(
    SE: Dict[str, Any],
    margin: Optional[int] = None,
    indices: Optional[Sequence[int]] = None
) -> Dict[str, Any]:
    """
    Filter a SingleCellExperiment-style dictionary.

    Parameters
    ----------
    SE : dict
        A data dict with 'counts', 'rowData', 'colData'
    margin : int
        0 = filter rows (genes); 
        1 = filter columns (samples)
    indices : int or bool sequence
        Indices to keep.

    Returns
    -------
    dict
        Filtered SE dictionary.
    """
    ...

    if margin == 0:
        # filter genes (rows)
        return {
            "logcounts": SE['logcounts'].iloc[indices, :],
            "counts": SE['counts'].iloc[indices, :],
            "rowData": SE['rowData'].iloc[indices, :],
            "colData": SE['colData']
        }

    elif margin == 1:
        # filter samples (columns)
        return {
            "logcounts": SE['logcounts'].iloc[:, indices],
            "counts": SE['counts'].iloc[:, indices],
            "rowData": SE['rowData'],  # unchanged
            "colData": SE['colData'].iloc[indices, :]
        }

    else:
        raise ValueError("margin must be 0 (rows/genes) or 1 (columns/samples)")
        


# %%
# make AGE two columns
dat_dict['colData'] = (
    dat_dict['colData']
        .assign(
            AGE_lwr = dat_dict['colData']['AGE'].str.split('-', expand=True)[0].astype(float),
            AGE_upr = dat_dict['colData']['AGE'].str.split('-', expand=True)[1].astype(float)
        )
)

# make SEX female binary column
dat_dict['colData'] = (
    dat_dict['colData']
    .assign(
        SEXF = (dat_dict['colData'].SEX == 2).astype(int)
    )
)


# %% [markdown]
# ### Tasks:
# __Ensemble-Based Models to Predict Tissue__
#
# Using projected samples from the latent spaces:
#
# 1. Use **at least two ensemble approaches** to predict tissue of origin:
#    - One approach must use an estimator that **tends to overfit**.
#    - Another must use a **weak learner** that slightly outperforms random guessing.
# 2. Use an appropriate **model evaluation strategy** to tune hyperparameters and assess:
#    - Prediction quality across tissues  
#    - Generalization on unseen data  
#    - Consider using `classification_report` from sklearn.
#
# __Ensemble-Based Models to Predict Age__
#
# Using **blood samples** and their projections into latent spaces:
#
# 1. Use at least two ensemble approaches to predict **age as a continuous variable**.  
# 2. Use a model evaluation strategy to:
#    - Tune hyperparameters  
#    - Assess prediction quality  
#    - Evaluate generalization on unseen data  
#
#

# %% [markdown]
# ## Predict Tissue From Latent Dimensions

# %% [markdown]
# ### Overfitting Learner

# %%

# %% [markdown]
# ### Weak Learner

# %%

# %% [markdown]
# ## Predict Age 
