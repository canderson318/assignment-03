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
import os 
import requests as rq
import gzip
import polars as pl
import shutil
import pickle
import pandas as pd
from pathlib import Path



# %% [markdown]
# ### Set up

# %% [markdown]
# make directory file to be used in following notebooks. 
#
# _Change `working_dir` to correct location_

# %%
working_dir = "/Users/canderson/Documents/school/CPBS7602-class/assignment-03/"

# store working directory txt in home: ~/.wdpath.txt
filepath = Path.home() / "wdpath.txt"

filepath.write_text(working_dir.strip() + "\n")


# %% [markdown]
# #### Set working directory 

# %%

wdpath = Path.home() / "wdpath.txt"
wd = wdpath.read_text().strip()
os.chdir(wd)
print(f"Working directory changed to: {os.getcwd()}")

# %% [markdown]
# ### Download Data from URLs

# %% [markdown]
# Make data directory

# %%
# make raw-data directory
Path('raw-data').mkdir(exist_ok=True)


# %%
# download
def download(url, dest, out=None):
    dest = Path(dest)
    if out is not None:
        out = Path(out)

    print("Getting content from URL...")
    r = rq.get(url)
    r.raise_for_status()
    print("...Content recieved")

    print("Writing content...")
    # Write get to file
    with open(dest, "wb") as f:
        f.write(r.content)
    print("...Content written")

    # Unzip if needed
    if dest.suffix == ".gz" and out is not None:
        print("G-unzipping...")
        with gzip.open(dest, 'rb') as f_in:
            with open(out, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(dest)   # remove .gz
        print(f"Saved unzipped file to: {out}")
    else:
        print(f"Saved file to: {dest}")

    print("Done.\n")

# %%
# gtex 
download("https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz","raw-data/gene-tpm.gct.gz", "raw-data/gene-tpm.gct")
#^ ~4 min

# %%
# sample metadata
download("https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDD.xlsx", "raw-data/sample-attributes.xlsx")

# %%
# sample phenotypes
download("https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDD.xlsx", "raw-data/sample-phenotypes.xlsx")

# %%
# write sample-phenotypes.xlsx to CSV (same dir, same base name)
def writeCsv(pth):
    xlsx_path = Path(pth)
    csv_path = xlsx_path.with_suffix(".csv")
    df = pd.read_excel(xlsx_path)
    df.to_csv(csv_path, index=False)

# %%
for pth in ["raw-data/sample-attributes.xlsx", "raw-data/sample-phenotypes.xlsx"]:
    writeCsv(pth)

# %%
# dbGaP attributes
with(open("raw-data/sample-attributes-dbGaP.csv", "wb") as f):
    f.write(rq.get("https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt").content)

# %%
# dbGaP phenotypes
with(open("raw-data/sample-phenotypes-dbGaP.csv", "wb") as f):
    f.write(rq.get("https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt").content)

# %% [markdown]
# ### Mung Data

# %%
# read gene matrix
gene_dat = pl.read_csv(
    "raw-data/gene-tpm.gct",
    separator="\t",
    skip_rows=2
).to_pandas()
# ^ gene x sample

# %%
# # read metadata
samp_attr = pd.read_csv("raw-data/sample-attributes-dbGaP.csv", delimiter = "\t") # < sample level metadata
samp_pheno = pd.read_csv("raw-data/sample-phenotypes-dbGaP.csv", delimiter = "\t") # < subject level metadata 

# %%
# select columns to counts rowDat and colDat
print(f"Gene data shape: {gene_dat.shape}")

# %%
counts = gene_dat.drop(["Name", "Description"], axis=1).copy()
rowDat = gene_dat[["Name", "Description"]].copy()

# %%
# change rownames
counts.index = rowDat['Name']

# %%
# merge samp attr and pheno dfs
samp_attr["SAMPID"].unique()[1]
samp_pheno["SUBJID"].unique()[1]

# %%
# make subjid as first two chunks of sampid
samp_attr["SUBJID"]= ["-".join(str(val).split("-")[:2]) for val in samp_attr["SAMPID"]]
# samp_attr[["SUBJID","SAMPID"]]

# %%
# merge pheno into attr
colDat = samp_attr.merge(samp_pheno, "left", "SUBJID")

# %%
# check shape
colDat.shape[0]==samp_attr.shape[0]
# ^^ columns of gene_dat are rows in colDat

# %%
# filter for samples in gene counts
colDat= colDat[colDat["SAMPID"].isin(counts.columns)]

# %%
# save to dictionary
dat_dict = {'counts': counts, 'rowData': rowDat, 'colData': colDat} 

# %%
for key, value in dat_dict.items():
    print(f"{key} shape: {value.shape}")

# %% [markdown]
# ### Save 

# %%
print("Saving...")
with open("processed-data/001-dat-dict.pkl", 'wb') as f:
    pickle.dump(dat_dict, f)
print("Done")

# %%
