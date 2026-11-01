# Breast Cancer Diagnosis — Classifier Studio

> ML Assignment 2 (NSP1) — five classification models trained on a UCI dataset and
> served through an interactive Streamlit application.

---

## a. Problem statement

Breast cancer diagnosis from a fine-needle aspirate (FNA) biopsy is traditionally a
manual, cytologist-dependent judgement. The goal of this project is to build a
**supervised binary classifier** that decides whether a tumour mass described by
quantitative cell-nucleus measurements is **malignant** or **benign**.

Because a missed malignancy is far more costly than a false alarm, the evaluation
deliberately looks beyond raw accuracy and reports **AUC, Recall and the Matthews
Correlation Coefficient (MCC)**, which stay honest under the mild class imbalance
present in the data.

Five classifiers are trained on identical splits, benchmarked against one another,
persisted to disk, and exposed through a Streamlit UI that scores freshly uploaded
test data **without ever retraining**.

---

## b. Dataset description

| Property | Value |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) — **WDBC** |
| Source | UCI Machine Learning Repository ([link](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)) |
| Access used | `sklearn.datasets.load_breast_cancer` (ships the identical UCI data) |
| Task | Binary classification |
| Instances | **569** (requirement: ≥ 500 ✅) |
| Features | **30** numeric, real-valued (requirement: ≥ 12 ✅) |
| Target column | `diagnosis` — `0 = malignant`, `1 = benign` |
| Class balance | 357 benign (62.7 %) / 212 malignant (37.3 %) |
| Missing values | None |

### Feature construction

Each image of an FNA slide yields **ten base morphological measurements** of the cell
nuclei:

`radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`,
`concave points`, `symmetry`, `fractal dimension`

For every base measurement three statistics are recorded across the nuclei in the
image — the **mean**, the **standard error**, and the **"worst"** (mean of the three
largest values) — giving `10 × 3 = 30` predictors.

### Preprocessing

* **Stratified split** — 75 % train / 25 % test (`random_state = 2026`), preserving the
  class ratio in both halves. The test half (143 rows) is exported as `test_data.csv`.
* **Standardisation** — a `StandardScaler` is fitted **on the training split only** and
  persisted to `model/scaler.joblib`. It is applied to the distance- and
  margin-sensitive learners (Logistic Regression, KNN, SVM). The Decision Tree and
  Gaussian Naive Bayes consume raw features, since scaling is irrelevant to them.
* **No leakage** — the scaler never sees test data during fitting, and the Streamlit app
  only ever calls `transform`.

---

## c. GitHub repository link

```
https://github.com/<your-username>/<your-repo-name>
```

> ⚠️ Replace the placeholder above with your actual repository URL before submitting.

**Live Streamlit app:** `https://<your-app-name>.streamlit.app`

### Repository contents

```
project-folder/
│-- app.py                 # Streamlit front-end
│-- train.py               # Trains + persists all five classifiers
│-- assignment_ml.ipynb    # Exploratory notebook
│-- requirements.txt       # Deployment dependencies
│-- README.md              # This file
│-- test_data.csv          # 143-row held-out test split
│-- .gitignore
└-- model/
    │-- logistic_regression.joblib
    │-- decision_tree.joblib
    │-- k_nearest_neighbour.joblib
    │-- naive_bayes.joblib
    │-- support_vector_machine.joblib
    │-- scaler.joblib        # Fitted StandardScaler
    │-- feature_names.joblib # Column order contract
    │-- metadata.json        # Class labels + dataset facts
    └-- benchmark.csv        # Saved comparison table
```

---

## d. Models used

All five models were trained on the **same** 426-row training split and evaluated on the
**same** 143-row held-out test split.

### Hyperparameters

| Model | Configuration | Scaled input |
|---|---|:---:|
| Logistic Regression | `C=0.8`, `solver='liblinear'`, `max_iter=5000` | ✅ |
| Decision Tree | `criterion='entropy'`, `max_depth=5`, `min_samples_leaf=4` | ❌ |
| K-Nearest Neighbour | `n_neighbors=9`, `weights='distance'`, Euclidean | ✅ |
| Naive Bayes | `GaussianNB(var_smoothing=1e-9)` | ❌ |
| Support Vector Machine | `C=2.0`, RBF kernel, `gamma='scale'` | ✅ |

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9790 | 0.9958 | 0.9780 | **0.9889** | **0.9834** | 0.9550 |
| Decision Tree | 0.9371 | 0.9457 | 0.9765 | 0.9222 | 0.9486 | 0.8700 |
| KNN | 0.9720 | 0.9823 | 0.9674 | **0.9889** | 0.9780 | 0.9400 |
| Naive Bayes | 0.9161 | 0.9820 | 0.9432 | 0.9222 | 0.9326 | 0.8219 |
| Support Vector Machine | **0.9790** | **0.9966** | **0.9888** | 0.9778 | 0.9832 | **0.9553** |

*The positive class is `1 = benign`. AUC uses predicted probabilities, except for the
SVM where the signed distance to the decision boundary is used.*

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Remarkably strong for a linear model — 97.90 % accuracy and 0.9958 AUC. This confirms that after standardisation the two diagnoses are very nearly **linearly separable** in the 30-dimensional space. It achieves the joint-best recall (0.9889), missing only one benign case. Its mild `C=0.8` regularisation keeps the 30 correlated predictors from inflating any single coefficient. The flat decision surface is also the easiest to explain clinically, which matters in a diagnostic setting. |
| **Decision Tree** | The clear weakest performer (93.71 % accuracy, 0.9457 AUC). A single axis-aligned tree must approximate a smooth, oblique boundary with staircase splits, so it discards the correlation structure that the other models exploit. Its AUC lags far behind the rest because a depth-5 tree produces only a handful of distinct leaf probabilities — the ROC curve is coarse and step-like rather than smooth. Its precision (0.9765) is respectable, but recall collapses to 0.9222, meaning it over-predicts malignancy. Useful only for its interpretable rule set. |
| **KNN** | Very competitive at 97.20 % accuracy, trailing the leaders by just one misclassification. Distance weighting lets nearby neighbours dominate the vote, and with `k=9` the model is smooth enough to resist noise. It ties for best recall (0.9889). Its lower AUC (0.9823) reflects the fact that probabilities are quantised into at most nine levels, limiting ranking resolution. Note that KNN is entirely dependent on the scaler — without standardisation, `area` (range ~2500) would swamp `smoothness` (range ~0.1). |
| **Naive Bayes** | Lowest accuracy (91.61 %) and MCC (0.8219), yet a surprisingly high AUC of 0.9820. This gap is the signature failure mode of Naive Bayes: the **conditional-independence assumption is badly violated here** (radius, perimeter and area are near-perfectly correlated by construction), so the product of likelihoods becomes grossly overconfident and the 0.5 threshold lands in the wrong place. The *ranking* of samples stays good (hence the high AUC) even though the *calibration* is poor. Threshold tuning would recover much of the lost accuracy. |
| **Support Vector Machine** | The overall winner on nearly every measure: best AUC (0.9966), best precision (0.9888) and best MCC (0.9553). The RBF kernel carves out a smooth non-linear margin that handles the slight curvature the linear model cannot, while the large-margin objective generalises well despite only 426 training rows. Its one trade-off is marginally lower recall than Logistic Regression — it is slightly more conservative about declaring a mass benign, which is arguably the **safer error direction** for cancer screening. |
| **Overall, winner for your dataset?** | **Support Vector Machine (RBF).** It tops the two most reliable headline metrics on an imbalanced problem — **MCC (0.9553)** and **AUC (0.9966)** — and has the highest precision. Logistic Regression is a very close second with identical accuracy and better recall; it is the pragmatic choice if **interpretability** or **minimising missed benign cases** is the priority. Given that malignant cases are the costly miss, the SVM's conservatism makes it the better clinical default. |

---

## How to reproduce

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all five models (writes model/ and test_data.csv)
python train.py

# 4. Launch the app
streamlit run app.py
```

---

## Streamlit app features

| Requirement | Where it lives in the UI |
|---|---|
| **a. Dataset upload (CSV)** | Sidebar → *1 · Test dataset* → file uploader. A bundled `test_data.csv` fallback is offered so the app is never blank on first load. Uploads are schema-validated and every missing column is reported by name. |
| **b. Model selection dropdown** | Sidebar → *2 · Model* → selectbox listing all five classifiers. |
| **c. Evaluation metrics** | Tab *🎯 Selected model* shows all six metrics as metric cards, with a **"best"** badge when the chosen model leads that metric. Tab *📊 Model comparison* shows the full table for all five models, a grouped bar chart, and overlaid ROC curves. |
| **d. Confusion matrix / report** | Tab *🎯 Selected model* renders an annotated confusion-matrix heat-map beside the full per-class classification report (precision / recall / F1 / support). |
| *Extras* | Row-level prediction inspector with confidence scores, CSV download of predictions and of the comparison table, target-distribution summary, and feature descriptive statistics. |

**No retraining occurs in the app.** `train.py` is the only place `.fit()` is called;
`app.py` loads the persisted estimators and calls `predict` / `predict_proba` only.

---

## Notes on reproducibility

* A single `RANDOM_STATE = 2026` seeds the split and every stochastic estimator, so
  `python train.py` reproduces the table above exactly.
* `model/feature_names.joblib` stores the training column order; the app reindexes any
  uploaded CSV to that order before scoring, so column shuffling in the upload is safe.
