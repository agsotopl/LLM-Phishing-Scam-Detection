"""Checks for overfitting: compares train vs test performance, runs 5-fold cross-validation on the training set, and sweeps regularization strength to see how sensitive results are to model complexity."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA_PATH = "data/processed/difraud_resplit.parquet"
EMBEDDINGS_PATH = "data/processed/embeddings.npy"


def load_data():
    df = pd.read_parquet(DATA_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    train_mask = (df["split"] == "train").values
    test_mask = (df["split"] == "test").values
    return (
        embeddings[train_mask], df.loc[train_mask, "label"].values,
        embeddings[test_mask], df.loc[test_mask, "label"].values,
    )


def train_vs_test_gap(X_train, y_train, X_test, y_test):
    """Fits on train, reports train and test AUC/F1 side by side."""
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print("train vs test gap:")
    print(f"  train  roc_auc={roc_auc_score(y_train, train_prob):.4f}  "
          f"f1={f1_score(y_train, train_pred):.4f}")
    print(f"  test   roc_auc={roc_auc_score(y_test, test_prob):.4f}  "
          f"f1={f1_score(y_test, test_pred):.4f}")


def cross_validate(X_train, y_train):
    """5-fold CV on training data to check variance across folds."""
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\n5-fold CV roc_auc: mean={scores.mean():.4f}  std={scores.std():.4f}")
    print(f"  per fold: {np.round(scores, 4)}")


def regularization_sweep(X_train, y_train, X_test, y_test):
    """Tests how test performance shifts across regularization strengths."""
    print("\nregularization sweep (C = inverse strength, lower = more regularized):")
    for c in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", C=c)
        model.fit(X_train, y_train)
        test_prob = model.predict_proba(X_test)[:, 1]
        print(f"  C={c:<6} test_roc_auc={roc_auc_score(y_test, test_prob):.4f}")


def main():
    X_train, y_train, X_test, y_test = load_data()
    train_vs_test_gap(X_train, y_train, X_test, y_test)
    cross_validate(X_train, y_train)
    regularization_sweep(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()