# Breast Cancer Diagnosis - Classifier App

ML Assignment2. Five classification models trained on a UCI dataset and served
through a Streamlit app.

## a. Problem statement

Given 30 measurements taken from a fine-needle aspirate biopsy image, predict whether a
tumour is **malignant** or **benign**. Since missing a malignant case is worse than a
false alarm, I report AUC, Recall and MCC alongside accuracy instead of relying on
accuracy alone.

## b. Dataset description

| Property | Value |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) - WDBC |
| Source | [UCI Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), loaded via `sklearn.datasets.load_breast_cancer` |
| Task | Binary classification |
| Instances | 569 (requirement: >= 500) |
| Features | 30 numeric (requirement: >= 12) |
| Target | `diagnosis` - 0 = malignant, 1 = benign |
| Class balance | 357 benign / 212 malignant |
| Missing values | None |

Ten base measurements of the cell nuclei (radius, texture, perimeter, area, smoothness,
compactness, concavity, concave points, symmetry, fractal dimension) are each recorded as
a mean, a standard error and a "worst" value, giving 10 x 3 = 30 features.

**Preprocessing:** stratified 75/25 train-test split (`random_state = 2026`). A
`StandardScaler` is fitted on the training split only and applied to Logistic Regression,
KNN and SVM. The Decision Tree and Naive Bayes use raw features since scaling does not
affect them. The 143-row test split is saved as `test_data.csv`.

## c. GitHub repository link

```
https://github.com/Ritu-Sahu03/machine_learning
```

Live app: `https://machinelearning-classification-model.streamlit.app`

```
project-folder/
|-- app.py              # Streamlit front-end
|-- train.py            # Trains and saves all five models
|-- requirements.txt
|-- README.md
|-- test_data.csv       # 143-row held-out test split
|-- model/              # 5 .joblib models + scaler + feature names + metadata
```

## d. Models used

| Model | Configuration | Scaled |
|---|---|---|
| Logistic Regression | `C=0.8`, `solver='liblinear'` | Yes |
| Decision Tree | `criterion='entropy'`, `max_depth=5`, `min_samples_leaf=4` | No |
| KNN | `n_neighbors=9`, `weights='distance'` | Yes |
| Naive Bayes | `GaussianNB` | No |
| SVM | `C=2.0`, RBF kernel | Yes |

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9790 | 0.9958 | 0.9780 | 0.9889 | 0.9834 | 0.9550 |
| Decision Tree | 0.9371 | 0.9457 | 0.9765 | 0.9222 | 0.9486 | 0.8700 |
| KNN | 0.9720 | 0.9823 | 0.9674 | 0.9889 | 0.9780 | 0.9400 |
| Naive Bayes | 0.9161 | 0.9820 | 0.9432 | 0.9222 | 0.9326 | 0.8219 |
| Support Vector Machine | 0.9790 | 0.9966 | 0.9888 | 0.9778 | 0.9832 | 0.9553 |

Positive class is `1 = benign`.

## How to run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## App features

- **CSV upload** in the sidebar, with the bundled `test_data.csv` as a fallback. Missing
  columns are reported by name.
- **Model dropdown** listing all five classifiers.
- **Evaluation metrics** shown as cards for the selected model, plus a full comparison
  table, a bar chart and ROC curves for all five.
- **Confusion matrix** heat-map and classification report for the selected model.
- Extras: row-level predictions with confidence scores and CSV downloads.

The app never retrains. `train.py` is the only file that calls `.fit()`; `app.py` just
loads the saved models and predicts.