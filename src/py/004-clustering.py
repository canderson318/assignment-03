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
import pickle
from sklearn.mixture import GaussianMixture
from pathlib import Path
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# set wd
os.chdir((Path.home()/"wdpath.txt").read_text().strip())

# %%
# load data
with open("processed-data/003-dat-dict.pkl",'rb') as f:
    dat_dict = pickle.load(f)

# %% [markdown]
# ### Cluster PCs and PCAs with GMM

# %%
print(dat_dict['rDims']["PCA"]['scores'].shape)
print(dat_dict['rDims']["PCA"]['scores'].shape)

# %%
PC_df = (
    pd.DataFrame(dat_dict['rDims']["PCA"]['scores'][:,:3])
    .assign(tissue = dat_dict['colData'].SMTS.values)
)

NMF_df = (
    pd.DataFrame(dat_dict['rDims']["NMF"]['scores'][:,:3])
    .assign(tissue = dat_dict['colData'].SMTS.values)
)


# %%
# --- PCA pairplot ---
p1 = sns.PairGrid(PC_df, hue="tissue", corner=False, 
              vars=PC_df.columns[:3], diag_sharey=False)
p1.map_lower(sns.scatterplot)
p1.map_diag(sns.histplot)
p1.map_upper(sns.scatterplot)
p1.add_legend()
p1.figure.suptitle("PCA (first 3 components)", y=1.02)

# --- PCA pairplot ---
p2 = sns.PairGrid(NMF_df, hue="tissue", corner=False,
               vars=NMF_df.columns[:3], diag_sharey=False)
p2.map_lower(sns.scatterplot)
p2.map_diag(sns.histplot)
p2.map_upper(sns.scatterplot)
p2.add_legend()
p2.figure.suptitle("NMF (first 3 components)", y=1.02)


plt.tight_layout()


# %%
# find best component number
dims = ['PCA','NMF']
for dim in dims:
    res = []
    for i in range(1, 5):
        gmm = GaussianMixture(n_components=i, random_state=0)
        gmm.fit(dat_dict['rDims'][dim]['scores'])
        BIC = gmm.bic(dat_dict['rDims'][dim]['scores'])
        res.append({"i": i, "BIC": BIC})
        
    res = pd.DataFrame(res)
    res['BIC'] = res['BIC'] - res['BIC'].min()
    
    # res['BIC'] = res['BIC'] / res['BIC'].sum()
    
    plt.scatter(res.iloc[:,0], res.iloc[:,1], label = dim)
    # connect points with a line
    plt.plot(res['i'], res['BIC'])

plt.legend(title="dims")
plt.xlabel("Number of mixture components")
plt.ylabel("Normalized BIC")
plt.title("Model selection via BIC")
plt.show()


# %% [markdown]
#
# NMF components for two components have best BIC relative to PCA

# %% [markdown]
# #### Cluster on NMF Components

# %%

# make model object
np.random.seed(1293)
NMF_gmm= GaussianMixture(n_components= 2,random_state=0) 
PCA_gmm= GaussianMixture(n_components= 2,random_state=0) 

# fit model
NMF_gmm.fit(dat_dict['rDims']['NMF']['scores'])
PCA_gmm.fit(dat_dict['rDims']['PCA']['scores'])

# predict clusters from data
NMF_clusters = NMF_gmm.predict(dat_dict['rDims']['NMF']['scores'])
PCA_clusters = PCA_gmm.predict(dat_dict['rDims']['PCA']['scores'])

nmf = pd.DataFrame(
    dat_dict['rDims']["NMF"]["scores"],
    columns=[f"NMF{i+1}" for i in range(dat_dict['rDims']['NMF']['scores'].shape[1])]
)

pca = pd.DataFrame(
    dat_dict['rDims']["PCA"]["scores"],
    columns=[f"PCA{i+1}" for i in range(dat_dict['rDims']['PCA']['scores'].shape[1])]
)

plot_df = pd.concat([nmf, pca], axis=1).assign(
    nmf_cluster=NMF_clusters,
    pca_cluster=PCA_clusters,
    tissue = (dat_dict['colData'].SMTS.values=="Brain").astype(int)
)


plt.figure(figsize=(6,5))
sns.scatterplot(
    data=plot_df,
    x="NMF1", y="NMF2",
    hue="nmf_cluster",
    # palette={0: "red", 1: "blue"},
    s=10
)
plt.show()

plt.figure(figsize=(6,5))
sns.scatterplot(
    data=plot_df,
    x="NMF1", y="NMF2",
    hue="tissue",
    # palette={0: "red", 1: "blue"},
    s=10
)
plt.show()

plt.figure(figsize=(6,5))
sns.scatterplot(
    data=plot_df,
    x="PCA1", y="PCA2",
    hue="pca_cluster",
    # palette={0: "red", 1: "blue"},
    s=10
)
plt.show()


plt.figure(figsize=(6,5))
sns.scatterplot(
    data=plot_df,
    x="PCA1", y="PCA2",
    hue="tissue",
    # palette={0: "red", 1: "blue"},
    s=10
)
plt.show()



# %%
from sklearn.metrics import classification_report

y_true = (dat_dict['colData']['SMTS'].values == "Brain").astype(int)

print("\\\\ NMF GMM classification Report \\\\\n",classification_report(y_true, NMF_clusters, zero_division=0))
print("\\\\ PCA GMM classification Report \\\\\n",classification_report(y_true, PCA_clusters, zero_division=0))

# %% [markdown]
# Classification on NMF performs pefectly

# %%
