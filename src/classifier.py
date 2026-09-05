"""Trains logistic regression on cached embeddings"""

import json
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_PATH = "data/processed/difraud_resplit.parquet"
EMBEDDINGS_PATH = "data/processed/embeddings.npy"
METRICS_PATH = "outputs/metrics.json"


def compute_metrics(y_true, y_pred, y_prob):
    """Returns accuracy, precision, recall, F1, ROC-AUC, and PR-AUC."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def main():
    df = pd.read_parquet(DATA_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"

    X_train, y_train = embeddings[train_mask], df.loc[train_mask, "label"]
    X_test, y_test = embeddings[test_mask], df.loc[test_mask, "label"]

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    joblib.dump(model, "outputs/model.joblib")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {"overall": compute_metrics(y_test, y_pred, y_prob)}

    test_df = df.loc[test_mask].reset_index(drop=True)
    for domain in test_df["domain"].unique():
        domain_mask = (test_df["domain"] == domain).values
        results[domain] = compute_metrics(
            y_test.values[domain_mask],
            y_pred[domain_mask],
            y_prob[domain_mask],
        )

    print(json.dumps(results, indent=2))
    print("\nFull classification report (overall):")
    print(classification_report(y_test, y_pred))

    os.makedirs("outputs", exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {METRICS_PATH}")


if __name__ == "__main__":
    main()