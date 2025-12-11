"""
CKD Project – Python Modelling (Member X)

- Baseline model  : Logistic Regression
- My main model   : SVM with RBF kernel

Input files:
- kidney_train.csv
- kidney_test.csv
"""

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    make_scorer
)

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


# =========================================================
# 1. LOAD DATA
# =========================================================

train_df = pd.read_csv("kidney_train.csv")
test_df  = pd.read_csv("kidney_test.csv")

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

TARGET_COL = "classification"

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]

X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

# Binary version of target (ckd = 1, notckd = 0) for ROC–AUC
y_train_binary = (y_train == "ckd").astype(int)
y_test_binary  = (y_test == "ckd").astype(int)


# =========================================================
# 2. DEFINE NUMERIC & CATEGORICAL FEATURES (MATCH R)
# =========================================================

numeric_features = [
    "age", "bp", "sg", "al", "su",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wbcc", "rbcc"
]

categorical_features = [
    "rbc", "pc", "pcc", "ba",
    "htn", "dm", "cad",
    "appet", "pe", "ane"
]


# =========================================================
# 3. PREPROCESSOR (SCALER + ONE-HOT ENCODER)
# =========================================================

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# =========================================================
# 4. HELPER: EVALUATION FUNCTION
# =========================================================

def evaluate_model(name, model, X_train, y_train, X_test, y_test, y_test_binary):
    """
    Fit model, print metrics, and plot ROC curve.
    """
    print("\n" + "=" * 70)
    print(f"MODEL: {name}")
    print("=" * 70)

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities (needed for ROC–AUC / ROC curve)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        classes = model.classes_
        # column index for "ckd" (positive class)
        pos_index = list(classes).index("ckd")
        y_proba = proba[:, pos_index]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="ckd")
    rec = recall_score(y_test, y_pred, pos_label="ckd")
    f1 = f1_score(y_test, y_pred, pos_label="ckd")

    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")

    if y_proba is not None:
        auc = roc_auc_score(y_test_binary, y_proba)
        print(f"ROC–AUC  : {auc:.3f}")
    else:
        auc = None
        print("ROC–AUC  : not available (no probabilities)")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # ROC curve
    if y_proba is not None:
        RocCurveDisplay.from_predictions(
            y_test_binary, y_proba, name=name
        )
        plt.title(f"ROC Curve – {name}")
        plt.show()

    return {
        "fitted_model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc
    }


# =========================================================
# 5. BASELINE MODEL – LOGISTIC REGRESSION
# =========================================================

log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",   # handle CKD class imbalance
    solver="liblinear"
)

log_reg_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", log_reg)
    ]
)

log_results = evaluate_model(
    "Baseline – Logistic Regression",
    log_reg_pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    y_test_binary
)


# =========================================================
# 6. FEATURE IMPORTANCE (LOGISTIC REGRESSION)
# =========================================================

def get_log_reg_feature_importance(pipeline):
    """
    Extract feature names and coefficients from a fitted
    LogisticRegression pipeline (with ColumnTransformer).
    """
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    num_feats = numeric_features

    ohe = pre.named_transformers_["cat"]
    cat_feats = ohe.get_feature_names_out(categorical_features)

    all_features = np.concatenate([num_feats, cat_feats])
    coefs = model.coef_[0]

    importance_df = pd.DataFrame({
        "feature": all_features,
        "coef": coefs,
        "abs_coef": np.abs(coefs)
    }).sort_values("abs_coef", ascending=False)

    return importance_df

print("\nTop 15 features by |coefficient| in the baseline Logistic Regression:")
log_importance = get_log_reg_feature_importance(log_results["fitted_model"])
print(log_importance.head(15))


# =========================================================
# 7. MY MODEL – SVM (RBF) WITH GRID SEARCH
# =========================================================

svm_base = SVC(
    kernel="rbf",
    probability=True,      # so we can compute ROC–AUC
    class_weight="balanced"
)

svm_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", svm_base)
    ]
)

param_grid = {
    "model__C": [0.1, 1, 10],
    "model__gamma": ["scale", "auto"]
}

# Custom F1 scorer where positive class = "ckd"
f1_ckd_scorer = make_scorer(f1_score, pos_label="ckd")

svm_grid = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=param_grid,
    scoring=f1_ckd_scorer,    # use F1 for CKD as positive class
    cv=5,
    n_jobs=-1
)

print("\nRunning GridSearchCV for my SVM model...")
svm_grid.fit(X_train, y_train)

print("\nBest SVM parameters:", svm_grid.best_params_)
best_svm = svm_grid.best_estimator_

svm_results = evaluate_model(
    "My model – SVM (RBF, tuned)",
    best_svm,
    X_train,
    y_train,
    X_test,
    y_test,
    y_test_binary
)


# =========================================================
# 8. MODEL COMPARISON SUMMARY
# =========================================================

summary_df = pd.DataFrame([
    {
        "Model": "Baseline – Logistic Regression",
        "Accuracy": log_results["accuracy"],
        "Precision": log_results["precision"],
        "Recall": log_results["recall"],
        "F1": log_results["f1"],
        "AUC": log_results["auc"],
    },
    {
        "Model": "My model – SVM (RBF, tuned)",
        "Accuracy": svm_results["accuracy"],
        "Precision": svm_results["precision"],
        "Recall": svm_results["recall"],
        "F1": svm_results["f1"],
        "AUC": svm_results["auc"],
    },
])

print("\n\n===== MODEL COMPARISON SUMMARY (BASELINE VS MY MODEL) =====")
print(summary_df.to_string(index=False))

# Optionally save to CSV for your report
summary_df.to_csv("model_comparison_memberX.csv", index=False)
print("\nSaved 'model_comparison_memberX.csv' with results.")
