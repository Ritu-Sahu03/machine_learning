"""
app.py — Breast Cancer Diagnosis Classifier Studio
==================================================

Streamlit front-end for ML Assignment 2 (NSP1).

Implements the four graded UI requirements:
  a. CSV upload widget for the held-out test data
  b. Model selection dropdown across the five trained classifiers
  c. Display of Accuracy / AUC / Precision / Recall / F1 / MCC
  d. Confusion matrix heat-map plus full classification report

Trained estimators are loaded from ./model and are *never* refitted on the
uploaded data — the app only scores and reports.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).parent
ARTEFACT_DIR = PROJECT_ROOT / "model"
BUNDLED_TEST_CSV = PROJECT_ROOT / "test_data.csv"
TARGET_COLUMN = "diagnosis"

# Display label -> artefact filename stem produced by train.py
CLASSIFIER_CATALOGUE = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbour": "k_nearest_neighbour",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "support_vector_machine",
}

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(
    page_title="Breast Cancer Classifier Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Artefact loading (cached so Streamlit Cloud stays responsive)
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner="Loading trained estimators…")
def load_artefacts():
    """Return (models, scaler, feature_names, metadata) from the model folder."""
    if not ARTEFACT_DIR.exists():
        return None, None, None, None

    scaler = joblib.load(ARTEFACT_DIR / "scaler.joblib")
    feature_names = joblib.load(ARTEFACT_DIR / "feature_names.joblib")

    metadata = {}
    meta_path = ARTEFACT_DIR / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    models = {}
    for label, stem in CLASSIFIER_CATALOGUE.items():
        path = ARTEFACT_DIR / f"{stem}.joblib"
        if path.exists():
            models[label] = joblib.load(path)

    return models, scaler, feature_names, metadata


@st.cache_data(show_spinner=False)
def read_uploaded_csv(raw_bytes: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV into a dataframe."""
    return pd.read_csv(BytesIO(raw_bytes))


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #


def prepare_matrix(frame: pd.DataFrame, feature_names, scaler, requires_scaling: bool):
    """Align columns to training order and scale when the model expects it."""
    ordered = frame[feature_names]
    return scaler.transform(ordered) if requires_scaling else ordered.to_numpy()


def confidence_scores(estimator, matrix) -> np.ndarray:
    """Positive-class probability, falling back to the decision function."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(matrix)[:, 1]
    return estimator.decision_function(matrix)


def evaluate(estimator, matrix, y_true):
    """Score one estimator and return (metrics, predictions, confidences)."""
    y_pred = estimator.predict(matrix)
    y_score = confidence_scores(estimator, matrix)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics, y_pred, y_score


def validate_frame(frame: pd.DataFrame, feature_names) -> list:
    """Return a list of human-readable problems with the uploaded file."""
    problems = []
    if TARGET_COLUMN not in frame.columns:
        problems.append(
            f"Target column `{TARGET_COLUMN}` is missing from the uploaded file."
        )
    missing = [c for c in feature_names if c not in frame.columns]
    if missing:
        preview = ", ".join(f"`{c}`" for c in missing[:6])
        suffix = f" …and {len(missing) - 6} more" if len(missing) > 6 else ""
        problems.append(f"{len(missing)} feature column(s) missing: {preview}{suffix}")
    return problems


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def draw_confusion_matrix(y_true, y_pred, class_labels):
    matrix = confusion_matrix(y_true, y_pred)
    figure, axis = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="rocket_r",
        cbar=False,
        square=True,
        linewidths=1.2,
        linecolor="white",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=axis,
    )
    axis.set_xlabel("Predicted diagnosis")
    axis.set_ylabel("Actual diagnosis")
    axis.set_title("Confusion Matrix", pad=12, fontsize=11)
    figure.tight_layout()
    return figure


def draw_roc(curves: dict):
    figure, axis = plt.subplots(figsize=(5.4, 4.0))
    palette = sns.color_palette("husl", n_colors=max(len(curves), 1))

    for colour, (label, (y_true, y_score, auc)) in zip(palette, curves.items()):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        axis.plot(fpr, tpr, lw=2, color=colour, label=f"{label} (AUC {auc:.3f})")

    axis.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC curves on uploaded test data", fontsize=11)
    axis.legend(loc="lower right", fontsize=8, frameon=True)
    axis.grid(alpha=0.25, linestyle=":")
    figure.tight_layout()
    return figure


def draw_metric_bars(board: pd.DataFrame):
    melted = (
        board.reset_index()
        .melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
    )
    figure, axis = plt.subplots(figsize=(9.5, 4.0))
    sns.barplot(
        data=melted,
        x="Metric",
        y="Score",
        hue="ML Model Name",
        palette="crest",
        ax=axis,
    )
    axis.set_ylim(0, 1.05)
    axis.set_title("Metric comparison across all five classifiers", fontsize=11)
    axis.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)
    axis.grid(axis="y", alpha=0.25, linestyle=":")
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #


def main() -> None:
    st.title("🔬 Breast Cancer Diagnosis — Classifier Studio")
    st.caption(
        "Upload a held-out test CSV, pick a trained classifier, and inspect its "
        "diagnostic performance. Models are pre-trained and are **never refitted** here."
    )

    models, scaler, feature_names, metadata = load_artefacts()

    if not models:
        st.error(
            "No trained artefacts found in `./model`. "
            "Run `python train.py` first, then restart the app."
        )
        st.stop()

    class_labels = list(
        metadata.get("class_labels", {"0": "Class 0", "1": "Class 1"}).values()
    )

    # ---------------- Sidebar: data source + model dropdown ---------------- #
    with st.sidebar:
        st.header("⚙️ Controls")

        st.subheader("1 · Test dataset")
        upload = st.file_uploader(
            "Upload test CSV",
            type="csv",
            help="Must contain every training feature plus the "
            f"`{TARGET_COLUMN}` target column.",
        )

        use_bundled = False
        if upload is None and BUNDLED_TEST_CSV.exists():
            use_bundled = st.checkbox("Use the bundled `test_data.csv`", value=True)

        st.subheader("2 · Model")
        chosen_model = st.selectbox(
            "Classifier to analyse in depth", options=list(models.keys()), index=0
        )

        st.subheader("3 · Display")
        show_all = st.toggle("Benchmark all five models", value=True)
        show_roc = st.toggle("Overlay ROC curves", value=True)

        st.divider()
        st.caption(
            f"Features: **{metadata.get('n_features', len(feature_names))}** · "
            f"Training instances: **{metadata.get('n_instances', 'n/a')}**"
        )

    # ------------------------------- Load data ----------------------------- #
    if upload is not None:
        data = read_uploaded_csv(upload.getvalue())
        source_note = f"uploaded file `{upload.name}`"
    elif use_bundled:
        data = pd.read_csv(BUNDLED_TEST_CSV)
        source_note = "bundled `test_data.csv`"
    else:
        st.info("⬅️ Upload a test CSV from the sidebar to begin.")
        with st.expander("Expected CSV schema"):
            st.write(
                f"The file needs the `{TARGET_COLUMN}` column plus these "
                f"{len(feature_names)} feature columns:"
            )
            st.code(", ".join(feature_names), language="text")
        st.stop()

    problems = validate_frame(data, feature_names)
    if problems:
        st.error("The dataset cannot be scored:")
        for item in problems:
            st.markdown(f"- {item}")
        st.stop()

    data = data.dropna(subset=list(feature_names) + [TARGET_COLUMN])
    y_true = data[TARGET_COLUMN].to_numpy()

    st.success(
        f"Scoring **{len(data):,}** rows from {source_note} "
        f"across **{len(feature_names)}** features."
    )

    # ------------------------- Score every classifier ---------------------- #
    leaderboard, predictions, roc_inputs = {}, {}, {}

    for label, bundle in models.items():
        estimator = bundle["estimator"]
        matrix = prepare_matrix(data, feature_names, scaler, bundle["requires_scaling"])
        metrics, y_pred, y_score = evaluate(estimator, matrix, y_true)

        leaderboard[label] = metrics
        predictions[label] = y_pred
        roc_inputs[label] = (y_true, y_score, metrics["AUC"])

    board = pd.DataFrame(leaderboard).T[METRIC_ORDER]
    board.index.name = "ML Model Name"

    # -------------------------------- Tabs --------------------------------- #
    tab_focus, tab_compare, tab_preview = st.tabs(
        ["🎯 Selected model", "📊 Model comparison", "🗂️ Data preview"]
    )

    # --- Tab 1: the model chosen in the dropdown --- #
    with tab_focus:
        st.subheader(f"{chosen_model} — performance on this test set")

        focus_metrics = leaderboard[chosen_model]
        columns = st.columns(len(METRIC_ORDER))
        best_per_metric = board.idxmax()

        for column, metric in zip(columns, METRIC_ORDER):
            is_best = best_per_metric[metric] == chosen_model
            column.metric(
                label=metric,
                value=f"{focus_metrics[metric]:.4f}",
                delta="best" if is_best else None,
            )

        st.divider()

        left, right = st.columns([1, 1.15])

        with left:
            st.markdown("##### Confusion matrix")
            st.pyplot(
                draw_confusion_matrix(y_true, predictions[chosen_model], class_labels),
                clear_figure=True,
            )

        with right:
            st.markdown("##### Classification report")
            report = classification_report(
                y_true,
                predictions[chosen_model],
                target_names=class_labels,
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

        with st.expander("Row-level predictions"):
            detail = data[[TARGET_COLUMN]].copy()
            detail["predicted"] = predictions[chosen_model]
            detail["confidence"] = np.round(roc_inputs[chosen_model][1], 4)
            detail["correct"] = detail[TARGET_COLUMN] == detail["predicted"]
            st.dataframe(detail, use_container_width=True, height=320)
            st.download_button(
                "⬇️ Download predictions as CSV",
                detail.to_csv(index=False).encode("utf-8"),
                file_name=f"predictions_{CLASSIFIER_CATALOGUE[chosen_model]}.csv",
                mime="text/csv",
            )

    # --- Tab 2: all five side by side --- #
    with tab_compare:
        if not show_all:
            st.info("Enable **Benchmark all five models** in the sidebar.")
        else:
            st.subheader("Comparison table — all five classifiers")
            st.dataframe(
                board.round(4).style.format("{:.4f}").highlight_max(
                    axis=0, color="#1f6f3f"
                ),
                use_container_width=True,
            )

            champion = board["MCC"].idxmax()
            st.markdown(
                f"🏆 **Strongest overall on this test set: `{champion}`** "
                f"(MCC {board.loc[champion, 'MCC']:.4f}, "
                f"AUC {board.loc[champion, 'AUC']:.4f})"
            )

            st.pyplot(draw_metric_bars(board.round(4)), clear_figure=True)

            if show_roc:
                st.pyplot(draw_roc(roc_inputs), clear_figure=True)

            st.download_button(
                "⬇️ Download comparison table",
                board.round(4).to_csv().encode("utf-8"),
                file_name="model_comparison.csv",
                mime="text/csv",
            )

    # --- Tab 3: sanity check on the uploaded file --- #
    with tab_preview:
        st.subheader("Uploaded test data")
        st.dataframe(data.head(50), use_container_width=True)

        st.markdown("##### Target distribution")
        counts = (
            data[TARGET_COLUMN]
            .map({0: class_labels[0], 1: class_labels[1]})
            .value_counts()
            .rename_axis("Diagnosis")
            .reset_index(name="Count")
        )
        st.dataframe(counts, use_container_width=False, hide_index=True)

        st.markdown("##### Feature summary statistics")
        st.dataframe(data[feature_names].describe().T.round(3), use_container_width=True)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
