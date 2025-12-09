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
from joblib import Parallel, delayed
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
# ## Tasks
# __Model 1:__
#
# gene_i ~ tissue {brain, blood}  + sex + age
#
# __Model 2:__
#
# sex ~ gene_i

# %% [markdown]
# ### Linear Model 
# gene_i ~ tissue {brain, blood}  + sex + age
#

# %%

# sample metadata
samp_dat = dat_dict['colData'][['SMTS', 'AGE_lwr','AGE_upr', 'SEXF']]



# %%
samp_dat["SEXF"]

# %%

# model gene i expression given tisue{brain,blood} correcting for sex and age
results = {}
for gene in dat_dict['logcounts'].index:
    mod_d = pd.concat([
        samp_dat,
        dat_dict['logcounts'].loc[gene].rename("expr")
    ], axis=1)

    fit = ols("expr ~ SMTS + SEXF + AGE_lwr", data=mod_d).fit()
    results[gene] = fit


# %%
# make results df
rows = []

for gene, res in results.items():
    for coef, pval in res.pvalues.items():
        rows.append({
            "gene": gene,
            "coef": coef,
            "estimate": res.params[coef],
            "std_err": res.bse[coef],
            "t": res.tvalues[coef],
            "p": pval
        })

res_df = pd.DataFrame(rows)
 

# %%
res_df.head()

# %%
top_sig = (
     res_df
        .loc[lambda d: d['coef'].isin(["SMTS[T.Brain]"])]
        .assign(abs_eff = res_df['estimate'].abs())
        .sort_values(['p'], ascending=[True])
        .head(5)
)
top_eff = (
     res_df
        .loc[lambda d: d['coef'].isin(["SMTS[T.Brain]"])]
        .assign(abs_eff = res_df['estimate'].abs())
        .sort_values(['abs_eff'], ascending=[False])
        .head(5)
)

# %%
pd.set_option("display.width", 200)
print("\n\\Sorted by decreasing significance\\\n",top_sig)
print("\n\\Sorted by decreasing effect\\\n",top_eff)


# %% [markdown]
# #### Linear Modeling Results
# The genes shown above all have a negative relationship to tissue being brain. That means that there is a significantly positive relationship between each of these gene's expression levels and tissue being blood. These genes are most likely involved in blood specific processes that are not present in the brain. One reason for this is that the brain does not have blood so it makes sense that we would see the super blood specific genes causing the main differences. 

# %% [markdown]
# ### Logistic model 
# sex ~ gene_i

# %%

# Filter for blood only samples
where_blood = (dat_dict['colData'].SMTS == "Blood").values
where_blood


# %%

dat_dict_blood = filter_se(SE = dat_dict, margin = 1, indices = where_blood)
print(dat_dict_blood['colData'].SMTS.unique())
print(dat_dict_blood.keys())

# %%
# make modeling DF
## sample metadata
samp_dat = dat_dict_blood['colData'][['SMTS', 'AGE_lwr','AGE_upr', 'SEXF']]


# %%

# model sex given gene expression 
results = {}

for gene in dat_dict_blood['logcounts'].index:
    mod_d = (
        samp_dat
        .assign(expr = dat_dict_blood['logcounts'].loc[gene])
    )

    fit = logit("SEXF ~ expr", data=mod_d).fit(disp = False)
    results[gene] = fit


# %%
# make results df
rows = []

for gene, res in results.items():
    for coef, pval in res.pvalues.items():
        rows.append({
            "gene": gene,
            "coef": coef,
            "estimate": res.params[coef],
            "std_err": res.bse[coef],
            "t": res.tvalues[coef],
            "p": pval
        })

res_df = pd.DataFrame(rows)
 

# %%
res_df

# %%
top_sig = (
     res_df
        .loc[lambda d: d['coef'].isin(["expr"])]
        .assign(abs_eff = res_df['estimate'].abs())
        .sort_values(['p'], ascending=[True])
        .head(5)
)
top_eff = (
     res_df
        .loc[lambda d: d['coef'].isin(["expr"])]
        .assign(abs_eff = res_df['estimate'].abs())
        .sort_values(['abs_eff'], ascending=[False])
        .head(5)
)


# %%

pd.set_option("display.width", 200)
print("\n\\Sorted by decreasing significance\\\n",top_sig)
print("\n\\Sorted by decreasing effect\\\n",top_eff)


# %% [markdown]
# #### Logistic model Results

# %% [markdown]
# Here I am seeing what genes are most significantly associated with SEX being female. Genes with positive estimates mean that for every one unit increase in gene expression, there is an _estimate_ increase in the log-odds of being that subject's sex being female. When negative, the relationship is reveresed and those genes are more associated with male subjects. 
#
# For the results ordered by decreasing significance, we see effect sizes in a positive direction, meaning the top 5 genes that distinguish between male and female are mostly in females. Looking at the results ordered by effect size, all of the effects are negative, suggesting that these genes must not be expressed at all in females so they could potentially be Y chromosome related. 
