"""Helper functions for Notebook 03 (CIND 820 Milestone 3).

Kept in a separate file so the notebook stays short and the functions can be reused in Milestone 4.
Everything here works on one household table with the columns named below.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

TARGET = "food_insecure"     # 1 = any food insecurity (marginal, moderate or severe), 0 = food secure
WEIGHT = "FWEIGHT"           # the major income earner's person weight, used as a working household weight


def make_X(df, cols):
    """Turn each predictor into text labels. NaN becomes the label 'Missing'."""
    X = df[cols].copy()
    for c in cols:
        X[c] = X[c].astype("Int64").astype("string").fillna("Missing")
    return X.astype(object)


def fit_model(df, use_weights, cols, c_value=1.0):
    """Fit encoder + logistic regression on df only. If use_weights, each household counts by FWEIGHT / mean(FWEIGHT).
    Because the encoder is inside the pipeline, it learns its categories from df only (this is the leakage rule)."""
    model = Pipeline([
        ("encode", OneHotEncoder(handle_unknown="ignore")),     # a category never seen in df gets all zeros
        ("logit", LogisticRegression(C=c_value, max_iter=1000)),
    ])
    w = df[WEIGHT] / df[WEIGHT].mean() if use_weights else None
    model.fit(make_X(df, cols), df[TARGET], logit__sample_weight=w)
    return model


def predict_prob(model, df, cols):
    """Probability that each household is food insecure (column 1 = class 1)."""
    return model.predict_proba(make_X(df, cols))[:, 1]


def confusion_scores(y, prob, threshold, w=None):
    """Confusion matrix and the classification metrics at one cut-off. w = optional weights (FWEIGHT)."""
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1], sample_weight=w).ravel()
    div = lambda a, b: a / b if b > 0 else np.nan
    recall, specificity = div(tp, tp + fn), div(tn, tn + fp)
    return {"threshold": round(float(threshold), 2), "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": div(tn + tp, tn + fp + fn + tp), "precision": div(tp, tp + fp),
            "recall": recall, "specificity": specificity, "f1": div(2 * tp, 2 * tp + fp + fn),
            "f2": div(5 * tp, 5 * tp + 4 * fn + fp),       # like F1 but recall counts twice as much as precision
            "balanced_accuracy": (recall + specificity) / 2}


def all_scores(y, prob, threshold, w=None):
    """Confusion-based metrics plus ROC-AUC and PR-AUC (these two do not depend on the cut-off)."""
    out = confusion_scores(y, prob, threshold, w)
    out["roc_auc"] = roc_auc_score(y, prob, sample_weight=w)
    out["pr_auc"] = average_precision_score(y, prob, sample_weight=w)
    return out


def best_threshold(y, prob, w, metric):
    """Try cut-offs 0.05, 0.06 ... 0.95 and return the one with the highest value of the chosen metric."""
    best_t, best_value = 0.5, -1
    for t in np.arange(0.05, 0.951, 0.01):
        value = confusion_scores(y, prob, t, w)[metric]
        if value > best_value:
            best_t, best_value = t, value
    return round(float(best_t), 2)
