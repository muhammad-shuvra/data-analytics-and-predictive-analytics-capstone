"""Helper functions for Notebook 003 (CIND 820 Milestone 3): baseline logistic regression for food insecurity.

PURPOSE
    The notebook does the analysis and shows the results. This file holds the reusable steps so the
    notebook stays short and the same code can be used again in Milestone 4.

INPUT
    One household table (one row per household) with:
      - the outcome column TARGET ("food_insecure", 1 = food insecure, 0 = food secure)
      - the weight column WEIGHT ("FWEIGHT")
      - the nine predictor columns, passed in as a list called `cols`

HOW THE NOTEBOOK USES THESE FUNCTIONS (in order)
    1. fit_model()        fit the encoder and logistic regression on the training years only
    2. predict_prob()     get each household's predicted probability of food insecurity
    3. best_threshold()   on the 2019 validation data only, find the cut-off with the highest F1
    4. all_scores()       report accuracy, precision, recall, specificity, F1, ROC-AUC and PR-AUC at that cut-off
                          (it calls confusion_scores() for the cut-off-based metrics)
    make_X() is used inside fit_model() and predict_prob() so training and testing prepare the data in the same way.

LEAKAGE RULE
    Nothing is learned from validation or test years. The encoder sits inside the model pipeline, so it
    learns its categories from the training data only. The cut-off is chosen on 2019 and then frozen.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

TARGET = "food_insecure"     # 1 = any food insecurity (marginal, moderate or severe), 0 = food secure
WEIGHT = "FWEIGHT"           # the major income earner's person weight, used as a working household weight


def make_X(df, cols):
    """Prepare the predictors for the encoder.

    Every predictor is treated as a category (a text label). A missing value (NaN) becomes its own
    label, 'Missing', so those households stay in the analysis instead of being dropped.
    Returns a table of text labels with the same rows as df and only the columns in `cols`.
    """
    X = df[cols].copy()
    for c in cols:
        X[c] = X[c].astype("Int64").astype("string").fillna("Missing")
    return X.astype(object)


def fit_model(df, use_weights, cols, c_value=1.0):
    """Fit one-hot encoding + logistic regression on df only and return the fitted model.

    use_weights: False = every household counts equally (unweighted track);
                 True = each household counts by FWEIGHT / mean(FWEIGHT) (weighted track).
                 Dividing by the mean keeps the total weight equal to the number of households.
    c_value:     regularization strength (scikit-learn default 1.0).
    The encoder is inside the pipeline, so it learns its categories from df only (this is the leakage rule).
    """
    model = Pipeline([
        ("encode", OneHotEncoder(handle_unknown="ignore")),     # a category never seen in df gets all zeros
        ("logit", LogisticRegression(C=c_value, max_iter=1000)),
    ])
    w = df[WEIGHT] / df[WEIGHT].mean() if use_weights else None
    model.fit(make_X(df, cols), df[TARGET], logit__sample_weight=w)
    return model


def predict_prob(model, df, cols):
    """Return the predicted probability that each household is food insecure (class 1).
    These are probabilities, not yes/no labels; a cut-off turns them into labels later."""
    return model.predict_proba(make_X(df, cols))[:, 1]


def confusion_scores(y, prob, threshold, w=None):
    """Turn probabilities into yes/no labels at one cut-off and return the confusion matrix counts and metrics.

    y: true labels (0/1).  prob: predicted probabilities.  threshold: households at or above it are labelled food insecure.
    w: optional weights (FWEIGHT); if given, the counts are weighted counts.
    A metric whose denominator is zero is returned as NaN instead of causing an error.
    """
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
    """Return the confusion-based metrics plus ROC-AUC and PR-AUC.
    ROC-AUC and PR-AUC use all the probabilities, so they do not depend on the cut-off."""
    out = confusion_scores(y, prob, threshold, w)
    out["roc_auc"] = roc_auc_score(y, prob, sample_weight=w)
    out["pr_auc"] = average_precision_score(y, prob, sample_weight=w)
    return out


def best_threshold(y, prob, w, metric):
    """Try cut-offs 0.05, 0.06 ... 0.95 and return the one with the highest value of the chosen metric.
    Used on the 2019 validation data only; the chosen cut-off is then frozen for the test years."""
    best_t, best_value = 0.5, -1
    for t in np.arange(0.05, 0.951, 0.01):
        value = confusion_scores(y, prob, t, w)[metric]
        if value > best_value:
            best_t, best_value = t, value
    return round(float(best_t), 2)
