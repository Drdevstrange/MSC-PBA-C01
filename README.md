# MSC-PBA-C01 — COMM053 Practical Business Analytics (Group Coursework)

This repository contains the group coursework deliverables for the **Practical Business Analytics (COMM053)** module. The project focuses on applying supervised machine learning to classify **Chronic Kidney Disease (CKD)** using a structured analytics workflow (CRISP-DM style). :contentReference[oaicite:2]{index=2}

---

## Project Overview

We implement:
- A **shared baseline Logistic Regression model** (common comparison point), and
- **One unique model per group member** (algorithm diversity), including KNN, MLP, Decision Tree, Random Forest, XGBoost, and SVM (RBF). :contentReference[oaicite:3]{index=3}

The goal is to compare predictive performance and discuss trade-offs such as **recall vs interpretability**, robustness, and suitability for potential clinical decision-support use.

---

## Repository Contents

### Key folders
- `Dataset/` — dataset files used for modelling (raw/processed). :contentReference[oaicite:4]{index=4}

### Preprocessing (R)
- `Preprocessing_steps_completed.r` — completed preprocessing and dataset preparation steps. :contentReference[oaicite:5]{index=5}

### Modelling notebooks (Jupyter)
- `6903269_UtsavSonkar_Baseline Logistic Regression+Tuned Decision Tree.ipynb` :contentReference[oaicite:6]{index=6}  
- `Baselinelog+MLP.ipynb` :contentReference[oaicite:7]{index=7}  
- `MLP_Kidneydataset_PBA.ipynb` :contentReference[oaicite:8]{index=8}  
- `PBA_COMM053_6905368_XGBoost.ipynb` :contentReference[oaicite:9]{index=9}  
- `PBA_COMM053_6905368_logistic_regression_XGBoost.ipynb` :contentReference[oaicite:10]{index=10}  
- `PBA_COMM053_6905368_test.ipynb` :contentReference[oaicite:11]{index=11}  
- `log+svm.ipynb` — Logistic Regression + tuned SVM (RBF). :contentReference[oaicite:12]{index=12}  

### Scripts / outputs
- `log+svm.py` — Python script version of Logistic Regression + tuned SVM workflow. :contentReference[oaicite:13]{index=13}  
- `model_comparison.csv` — summary comparison table of model metrics. :contentReference[oaicite:14]{index=14}  
- `base log+tuned dt results .csv` — exported results for baseline LR and tuned decision tree. :contentReference[oaicite:15]{index=15}  
- `output_6905368_COMM053.pdf` — exported PDF output for one contributor’s work. :contentReference[oaicite:16]{index=16}  

---

## How to Run (Recommended Workflow)

### 1) Preprocessing (R)
Run the R script first to clean and prepare the dataset:
- Open `Preprocessing_steps_completed.r`
- Execute end-to-end to generate the processed dataset splits (if your script outputs train/test CSVs). :contentReference[oaicite:17]{index=17}

### 2) Modelling (Python / Jupyter)
Open any notebook in Jupyter and run all cells:
- Baseline Logistic Regression is used for comparison across all models.
- Each notebook implements a unique algorithm and evaluation (confusion matrix, ROC-AUC, etc.). :contentReference[oaicite:18]{index=18}

---

## Environment Setup

### Python (recommended)
Create a virtual environment and install common dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
