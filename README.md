# Assignment 3

*__Find the Analyisis Report [here](REPORT.pdf).__*

In this assignment I first explored two different dimension reduction techniques on the gtex data and got a feel for their strengths by evaluating tissue prediction with them. I also tried several ensemble methods for predicting tissue and age. 

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
- Calculate 50 principal components from the gene counts matrix.
- Calculate 50 non-negative matrix factorization components from the gene counts matrix.

#### `004-clustering`
- Use Gaussian Mixture Modeling to predict tissue from NMF and PCA components. 
- Evaluate prediction accuracy using confusion matrices

#### `005-ensemble-modeling`
- Train two ensemble models to predict __tissue__ from PCA components
    - An overfitting Random Forest classifier
    - An underfitting AdaBoost classifier
    - Evaluate with accuracies
- Train two ensemble models to predict __age__ from PCA components
    - Random Forest regressor
    - Gradient Boosting regressor
    - (Ridge regressor as benchmark)
    - Evaluate with RMSE

## AI use
I used Chat-GPT code generation to help me learn python by translating R functions into python. All of the analysis ideas were mine and my prompts were for syntax, etc., only. 