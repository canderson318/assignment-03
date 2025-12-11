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
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from typing import Optional, Sequence, Any, Dict


# %%
os.chdir((Path.home() / "wdpath.txt").read_text().strip())
os.getcwd()

# %%
with open("processed-data/001-dat-dict.pkl",'rb') as f:
    dat_dict = pickle.load(f)

# %%
type(dat_dict)
len(dat_dict)
dat_dict.keys()
for key, value in dat_dict.items():
    print(f"{key}: {value.shape}")

# %%
with open("processed-data/001-dat-dict.pkl",'rb') as f:
    dat_dict = pickle.load(f)

# %%
type(dat_dict)
len(dat_dict)
dat_dict.keys()
for key, value in dat_dict.items():
    print(f"{key}: {value.shape}")

# %%
for col in dat_dict["colData"]:
    print(col)
    print(dat_dict["colData"][col].unique()[:10])


# %% [markdown]
# ## Filter Data

# %%

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
            # "logcounts": SE['logcounts'].iloc[indices, :],
            "counts": SE['counts'].iloc[indices, :],
            "rowData": SE['rowData'].iloc[indices, :],
            "colData": SE['colData']
        }

    elif margin == 1:
        # filter samples (columns)
        return {
            # "logcounts": SE['logcounts'].iloc[:, indices],
            "counts": SE['counts'].iloc[:, indices],
            "rowData": SE['rowData'],  # unchanged
            "colData": SE['colData'].iloc[indices, :]
        }

    else:
        raise ValueError("margin must be 0 (rows/genes) or 1 (columns/samples)")

# %% [markdown]
# ### Select top tissues

# %%
top_tiss = pd.DataFrame(dat_dict["colData"]['SMTS'].value_counts()).sort_values(by = "count", ascending = False).head(10).index
print(top_tiss)

# %%

tiss_inds = dat_dict["colData"]['SMTS'].isin(top_tiss).values
dat_dict = filter_se(dat_dict, 1, tiss_inds)

# %%
print(dat_dict['colData'].shape)
print(dat_dict['counts'].shape)

# %% [markdown]
#
# ### Select top 5000 most variable genes

# %%
# first exclude missy genes
## gene missing proportions
zero_sum = (dat_dict['counts']==0).sum(axis=1).values
zero_prop = zero_sum / (dat_dict['counts'].shape[1])
pd.Series(zero_prop).describe()

# %%
x, bins = np.histogram(zero_prop)

# %%
plt.figure()
plt.hist(bins[:-1],weights =  x)
plt.show()

# %%
# select for genes with < 60% 0s
non_missy_genes = zero_prop<.6
dat_dict = filter_se(dat_dict, margin = 0, indices = non_missy_genes)

# %%
# Compute variance for each gene 
variance = dat_dict["counts"].var(axis=1)

# %%
x, y = range(len(variance.values)), sorted(np.log1p(variance))

# %%
top_y = y[-5000:]
len(top_y)
is_top = np.zeros(len(y), dtype = bool)
is_top[-5000:]= True

# %%
plt.figure()
p1 = plt.scatter(
    x, y, c=is_top, s=10, cmap=plt.cm.Paired
)
plt.xlabel("Ind")
plt.ylabel("Variance")
plt.show()

# %%
# select top genes
dat_dict = filter_se(dat_dict, 0, is_top)

# %% [markdown]
# ### Standardize data

# %%
# lognormalize
## colSums
libsize = dat_dict['counts'].sum(axis=0) # sum across rows= colsums
# sum normalize columns and multiply by scaling factor so each sample sum is same
norm = dat_dict['counts'].div(libsize, axis=1) * 10_000
# log normalize
lognorm = np.log1p(norm)

# %%
# standardize 
## gene rowmeans
mean = lognorm.mean(axis=1) # axis 1 = across columns
## rowSds
std = lognorm.std(axis=1,  ddof=1)   
## centered and scaled
Z = lognorm.sub(mean, axis=0).div(std, axis=0)

# %%
dat_dict["logcounts"]= Z


# %%
print("Saving...")
with(open("processed-data/002-dat-dict.pkl", 'wb') as f ):
    pickle.dump(dat_dict, f)
print("Done")

# %%
