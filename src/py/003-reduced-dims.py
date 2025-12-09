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
from sklearn.decomposition import PCA
import seaborn as sns

# %%
os.chdir((Path.home()/"wdpath.txt").read_text().strip())

# %%
with open("processed-data/002-dat-dict.pkl",'rb') as f:
    dat_dict = pickle.load(f)


# %% [markdown]
# ### Fix colData index

# %%
dat_dict['colData'].index = dat_dict['colData'].SAMPID

# %% [markdown]
# ### Remove sparse genes 

# %%
na_sum = dat_dict['logcounts'].isna().sum(axis=1).values
na_prop = na_sum / (dat_dict['logcounts'].shape[1])
print(na_prop)
# ^^ not necessary

# %% [markdown]
# ### Calculate reduced dimensions

# %%
# fit PCA to capture 95% of variance (change n_components as needed)
print("Running PCA…")
pca = PCA(n_components=50, svd_solver='auto', random_state=0)
scores = pca.fit_transform(np.transpose(dat_dict['logcounts']))
scores = np.transpose(scores)
print("PCA finished")

# %%
print(scores.shape)
print(pca.explained_variance_ratio_.sum())

# %%
df = pd.DataFrame(np.transpose(scores)).iloc[:, :2].copy()
df["tissue"] = dat_dict["colData"]["SMTS"].values
sns.pairplot(df,
             hue="tissue",
             plot_kws={'s': 10})
plt.show()
plt.close()

# %%
# add to dat_dict
dat_dict['rDims'] = {}
dat_dict['rDims']['PCA']= {"scores": scores.T, "attributes": pca}


# %% [markdown]
# #### Other dimension reduction technique

# %% [markdown]
# NMF

# %%
# NMF needs the data to be non-negative
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_non_negative = scaler.fit_transform(dat_dict['logcounts'].T)

# %%
X_non_negative.shape

# %%
from sklearn.decomposition import NMF

nmf = NMF(n_components=50, max_iter=1000)

np.random.seed(1293)
nmf.fit(X_non_negative)
X_nmf_reduced = nmf.transform(X_non_negative)
#^^ 30s

# %%
nmf.components_

# %%
plot_d = (
    pd.DataFrame(X_nmf_reduced[:, :5])
    .assign(tissue = dat_dict['colData'].SMTS.values)
)

plot_d.head

# %%

sns.pairplot(
    plot_d,
    hue='tissue',
    palette="deep"
)

# %%

# add to dat_dict
dat_dict['rDims']['NMF'] = {"scores": X_nmf_reduced, "attributes": nmf}

# %% [markdown]
# ### Save

# %%
print("Saving...")
with(open("processed-data/003-dat-dict.pkl", 'wb') as f ):
    pickle.dump(dat_dict, f)
print("Done")

# %%
print(dat_dict['rDims']['NMF']['scores'].shape)
print(dat_dict['rDims']['PCA']['scores'].shape)

# %%
