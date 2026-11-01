"""
train.py
--------
Trains five classification models on the UCI Breast Cancer Wisconsin
(Diagnostic) dataset and persists them, together with the fitted scaler,
into the ./model directory.

Also writes a held-out ``test_data.csv`` that the Streamlit app consumes.

Run:
    python train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 2026
TEST_FRACTION = 0.25
TARGET_COLUMN = "diagnosis"

ARTEFACT_DIR = Path(__file__).parent / "model"
TEST_CSV_PATH = Path(__file__).parent / "test_data.csv"

# Models that operate on standardised features.
SCALE_SENSITIVE = {"Logistic Regression", "K-Nearest Neighbour", "Support Vector Machine"}


def build_dataframe() -> pd.DataFrame:
    """Load the WDBC data and return a tidy dataframe with a named target."""
    bundle = load_breast_cancer(as_frame=True)
    frame = bundle.frame.copy()
    frame = frame.rename(columns={"target": TARGET_COLUMN})
    # Tidy the column names so they survive a CSV round-trip cleanly.
    frame.columns = [c.strip().replace(" ", "_") for c in frame.columns]
    return frame


def candidate_models() -> dict:
    """Return the five classifiers required by the assignment brief."""
    return {
        "Logistic Regression": LogisticRegression(
            C=0.8, max_iter=5000, solver="liblinear", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy",
            max_depth=5,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbour": KNeighborsClassifier(
            n_neighbors=9, weights="distance", metric="minkowski", p=2
        ),
        "Naive Bayes": GaussianNB(var_smoothing=1e-9),
        # decision_function is used for AUC, so probability calibration is not needed.
        "Support Vector Machine": SVC(
            C=2.0, kernel="rbf", gamma="scale", random_state=RANDOM_STATE
        ),
    }


def score_model(y_true, y_pred, y_score) -> dict:
    """Compute every metric demanded by Step 2 of the brief."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main() -> None:
    ARTEFACT_DIR.mkdir(exist_ok=True)

    frame = build_dataframe()
    features = frame.drop(columns=[TARGET_COLUMN])
    labels = frame[TARGET_COLUMN]

    print(f"Dataset shape      : {frame.shape[0]} rows x {features.shape[1]} features")
    print(f"Class distribution : {labels.value_counts().to_dict()}")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_FRACTION,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    joblib.dump(scaler, ARTEFACT_DIR / "scaler.joblib")
    joblib.dump(list(features.columns), ARTEFACT_DIR / "feature_names.joblib")

    leaderboard = {}

    for name, estimator in candidate_models().items():
        needs_scaling = name in SCALE_SENSITIVE
        fit_x = x_train_scaled if needs_scaling else x_train.to_numpy()
        eval_x = x_test_scaled if needs_scaling else x_test.to_numpy()

        estimator.fit(fit_x, y_train)
        predictions = estimator.predict(eval_x)

        if hasattr(estimator, "predict_proba"):
            confidence = estimator.predict_proba(eval_x)[:, 1]
        else:  # SVC without probability=True exposes a decision function instead
            confidence = estimator.decision_function(eval_x)

        leaderboard[name] = score_model(y_test, predictions, confidence)

        slug = name.lower().replace(" ", "_").replace("-", "_")
        joblib.dump(
            {"estimator": estimator, "requires_scaling": needs_scaling},
            ARTEFACT_DIR / f"{slug}.joblib",
        )
        print(f"  saved -> model/{slug}.joblib")

    # Persist the comparison table so the README and app can reuse it.
    board = pd.DataFrame(leaderboard).T.round(4)
    board.index.name = "ML Model Name"
    board.to_csv(ARTEFACT_DIR / "benchmark.csv")

    with open(ARTEFACT_DIR / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "target_column": TARGET_COLUMN,
                "positive_label": 1,
                "class_labels": {"0": "malignant", "1": "benign"},
                "n_features": int(features.shape[1]),
                "n_instances": int(frame.shape[0]),
                "random_state": RANDOM_STATE,
            },
            handle,
            indent=2,
        )

    # Export the untouched held-out split for the Streamlit upload widget.
    holdout = x_test.copy()
    holdout[TARGET_COLUMN] = y_test.to_numpy()
    holdout.to_csv(TEST_CSV_PATH, index=False)

    print("\nHold-out benchmark\n" + "-" * 70)
    print(board.to_string())
    print(f"\ntest_data.csv written with {len(holdout)} rows -> {TEST_CSV_PATH.name}")


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
