# Assignment 3

[Analyisis Report](REPORT.md)

This assignment ...  
Look at [assignment-description.md](assignment-description.md) for more detail.

## Set up 
Download dependencies with `conda env create -f environment.yml`. 
(this file was created with `conda env export --from-history > environment.yml`)

## Analysis
Run these [scripts](src/py/) in numerical order to reproduce results. 

#### `001-download-data`
- Download gene counts matrix and sample/subject metadata and format in summarized experiment format. 
- Save as pandas DataFrames in a dictionary. 

#### `002-process-data`
- Filter gene counts for top 5000 most variable genes and top 10 most prevelant tissues excluding sparse genes. 
- Log normalize and standardize to mean 0 and sd of 1. 

#### `003-reduced-dims`
- Calculate 50 principal components on gene counts matrix.

## AI use
I used Chat-GPT code generation to help me learn python by translating R functions into python. All of the analysis ideas were mine and my prompts were for syntax, etc., only. 