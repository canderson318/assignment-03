# Assignment 3: Comparison of Ensemble Approaches on Gene Expression Data
**CPBS 7602: Introduction to Big Data in the Biomedical Sciences**

**By:** Milton Pividori  
**Department of Biomedical Informatics**  
**University of Colorado Anschutz Medical Campus**

---

## Assignment Overview

In this assignment, students will build upon work from Assignments 1 and 2, exploring dimensionality reduction and ensemble techniques, and comparing their performance. The task uses the same GTEx gene expression data but focuses on ensemble-based supervised learning. The assignment emphasizes model evaluation, overfitting/underfitting, and appropriate evaluation metrics.

**Due:** *December 13, 2025 at 5 pm MT*

---

## Reading Material (Optional)

See lecture slides with links to scikit-learn documentation on:
- Dimensionality reduction  
  - Decomposition methods  
  - Manifold learning  
- Ensembling  
- Individual estimators  

---

## Assignment Tasks

### **Data source**
Continue using the GTEx dataset from previous assignments. Use the same subset of genes (top 5,000 most variable genes, or a manageable subset) and the same tissues and samples.

---

## 1. Dimensionality Reduction

We are interested in how different dimensionality reduction techniques impact cluster analysis.

1. Apply **PCA** and (if resources allow) another dimensionality reduction technique to visualize the data.  
2. Adjust parameters—especially the **number of components**—to generate different latent spaces.  
   - Use the best clustering algorithm you identified in Assignment 1.  
3. Compare each latent space in terms of its ability to group samples by **tissue of origin**.

---

## 2. Ensemble-Based Models to Predict Tissue

Using projected samples from the latent spaces:

1. Use **at least two ensemble approaches** to predict tissue of origin:
   - One approach must use an estimator that **tends to overfit**.
   - Another must use a **weak learner** that slightly outperforms random guessing.
2. Use an appropriate **model evaluation strategy** to tune hyperparameters and assess:
   - Prediction quality across tissues  
   - Generalization on unseen data  
   - Consider using `classification_report` from sklearn.

---

## 3. Ensemble-Based Models to Predict Age

Using **blood samples** and their projections into latent spaces:

1. Use at least two ensemble approaches to predict **age as a continuous variable**.  
2. Use a model evaluation strategy to:
   - Tune hyperparameters  
   - Assess prediction quality  
   - Evaluate generalization on unseen data  

---

# Assignment Deliverables

## 1. Code

Submit your work as a GitHub repository containing a folder named `assignment03`. Include:

- A **README.md** explaining:
  - How to reproduce results  
  - How to create the conda environment  
- Well-documented **Jupyter notebooks** for each step  
- Properly set **random seeds** (e.g., `numpy.random.seed()`)

---

## 2. Analysis Report

Write a short report (1–2 pages) summarizing your main findings. Additional detail may be included in your notebooks.

---

# Grading Rubric

| Category | Points | Description |
|---------|--------|-------------|
| **Dimensionality Reduction** | 30% | Correct use of methods and evaluation of clustering results |
| **Ensemble Models: Predict Tissue** | 30% | Correct selection and evaluation of ensemble methods |
| **Ensemble Models: Predict Age** | 30% | Correct selection and evaluation of ensemble methods |
| **Interpretation & Reporting** | 10% | Well-structured, clear, and accurate reporting |

---
