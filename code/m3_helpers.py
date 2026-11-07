"""Helper functions for Notebook 03 (CIND 820 Milestone 3).

This file contains functions that prepare the predictor variables for modelling, fit a logistic
regression, make predictions, and calculate validation and test metrics. They are called from the
Python notebooks and can be reused in the Milestone 4 analysis.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

# food_insecure variable was created as a binary variable from the original FSCHHLDM variable that has 4 categories (0-3)
# the original variable had food insecurity responses representing (not food insecure = 0, marginal =1, moderate =2 or severe =3)
# in this note, we are assigning this food_insecure variable to TARGET variable for ML language's sake
# likewise, the original variable FWEIGHT is just being called WEIGHT for ML language purpose 

TARGET = "food_insecure"     # Following the food_insecure variable, the TARGET variable will contain 1 = any food insecurity and  0 = food secure
WEIGHT = "FWEIGHT"           # the major income earner's person weight, used as a working household weight

def make_X(df, cols):
    """Transform specified predictor columns into string-type text labels.
    Converts the data to integer and string formats, replacing any missing 
    or NaN values with the explicit label 'Missing'."""
    X = df[cols].copy()
    for c in cols:
        X[c] = X[c].astype("Int64").astype("string").fillna("Missing")
    return X.astype(object)


def fit_model(df, use_weights, cols, c_value=1.0):
    """Fit an encoder and a logistic regression model using only the provided dataframe (df). If use_weights is enabled, 
    individual household counts are scaled by FWEIGHT divided by its mean (FWEIGHT / mean(FWEIGHT)). 
    Because the encoder is integrated within the pipeline, it learns its categories strictly from df to adhere to data leakage rules."""
    model = Pipeline([
        ("encode", OneHotEncoder(handle_unknown="ignore")),     # a category never seen in df gets all zeros
        ("logit", LogisticRegression(C=c_value, max_iter=1000)),
    ])
    w = df[WEIGHT] / df[WEIGHT].mean() if use_weights else None
    model.fit(make_X(df, cols), df[TARGET], logit__sample_weight=w)
    return model


def predict_prob(model, df, cols):
    """Compute the estimated probability for each household being food insecure, derived from the 
    model's predictions where column 1 corresponds to class 1."""
    return model.predict_proba(make_X(df, cols))[:, 1]


def confusion_scores(y, prob, threshold, w=None):
    """Compute the confusion matrix and key classification metrics evaluated at a single classification cut-off threshold. 
    Accepts an optional weight parameter (w) using FWEIGHT for weighted evaluations."""
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
    """Compute standard confusion-based classification metrics alongside ROC-AUC and PR-AUC scores. 
    While confusion-based evaluations rely on a specific decision threshold, ROC-AUC and PR-AUC 
    remain independent of any cut-off."""
    out = confusion_scores(y, prob, threshold, w)
    out["roc_auc"] = roc_auc_score(y, prob, sample_weight=w)
    out["pr_auc"] = average_precision_score(y, prob, sample_weight=w)
    return out


def best_threshold(y, prob, w, metric):
    """Search across a sequence of decision cut-offs ranging from 0.05 to 0.95 to evaluate 
    the target performance metric, and return the specific cut-off 
    threshold that achieves the highest value."""
    best_t, best_value = 0.5, -1
    for t in np.arange(0.05, 0.951, 0.01):
        value = confusion_scores(y, prob, t, w)[metric]
        if value > best_value:
            best_t, best_value = t, value
    return round(float(best_t), 2)
