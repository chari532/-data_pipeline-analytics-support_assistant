# Analytics Module (`/analytics`)

## What this does

Loads the classic Titanic dataset (via Seaborn's built-in loader), profiles and
cleans it, tells a visual data story about survival, and then builds and
rigorously evaluates a full predictive-modeling pipeline: three classifiers
(Logistic Regression, Decision Tree, Random Forest), an imbalance-handling
comparison, hyperparameter tuning, and a regression side-task predicting fare.

This is one connected pipeline, not two separate exercises: `eda.py` loads the
raw dataset exactly once, cleans it, and saves it as `titanic.csv`. `modeling.py`
reads that same committed CSV and continues straight into modeling -- it never
calls `sns.load_dataset()` again.

## Install

```bash
pip install -r requirements.txt
```

Requires: `pandas`, `seaborn`, `matplotlib`, `scikit-learn`, `imbalanced-learn`, `joblib`

## How to run (in order)

```bash
python eda.py        # Part A: load, profile, clean, EDA charts -> titanic.csv + plots/
python modeling.py    # Part B: train/test split, preprocessing, models, tuning, regression, joblib save
```

`sns.load_dataset('titanic')` requires internet access the first time it runs
(it fetches and caches the data). If internet is unavailable at grading time,
the committed `titanic.csv` (produced by `eda.py`) can be read directly with
`pd.read_csv("titanic.csv")` as an offline fallback.

## Part A -- Cleaning decisions

Missing-value percentages measured on load:
- `deck`: ~77% missing -> **dropped the column entirely**, since imputing
  roughly 3 in 4 values would be unreliable and would dominate the feature
  with guessed values.
- `age`: ~20% missing (5-30% band) -> **imputed with the median**, robust to
  the right-skew confirmed in the univariate analysis, rather than dropping
  ~1 in 5 passengers.
- `embarked`, `embark_town`: ~0.2% missing (under 5%) -> **dropped those few
  rows**, since so few are affected that dropping barely changes the dataset.

No duplicate rows were removed. The Seaborn Titanic dataset has no unique
passenger identifier (no name/ID column), so a number of rows are exact
duplicates across all recorded features. These very likely represent distinct
passengers who happen to share identical values (same class, sex, age, fare,
etc.), not data-entry errors, so they were retained.

## Part A -- Key EDA findings

- Overall survival rate: ~38%.
- Survival by sex: female ~74%, male ~19% -- the single strongest predictor.
- Survival by class: 1st ~63%, 2nd ~47%, 3rd ~24%.
- Fare is right-skewed (mean > median > mode): a long tail of expensive 1st
  class tickets pulls the mean above the median.
- Strongest correlations: `pclass` <-> `fare` (~-0.55, fare is essentially a
  proxy for class) and `sibsp` <-> `parch` (~0.41, family-size-related).
- IQR outliers: age and fare both show a meaningful number of outliers on the
  high end (older passengers, and expensive tickets respectively).

## Part B -- Modeling approach

- **Split**: stratified 80/20 train/test split on `survived`, to preserve the
  ~62/38 class balance in both splits (a plain random split could skew this
  by chance).
- **Features used**: `pclass, sex, age, sibsp, parch, fare, embarked`.
  Deliberately excluded `alive` (identical to the target -- would leak),
  `class`/`who`/`adult_male`/`alone`/`embark_town` (redundant/derived from
  the features already used).
- **Preprocessing**: a `ColumnTransformer` (median-impute + `StandardScaler`
  for numeric features; most-frequent-impute + one-hot encoding for `sex`/
  `embarked`), wrapped in a `Pipeline` with each classifier. This is fit only
  on the training split and applied in transform-only mode to the test split.
- **Models**: Logistic Regression, Decision Tree (`max_depth=4`, visualized
  with `plot_tree`), Random Forest -- see `plots/decision_tree.png`.
- **Evaluation**: confusion matrix, accuracy, precision, recall, F1, ROC-AUC
  for all three, compared side by side (see `plots/roc_curves.png`).
- **Imbalance handling**: compared baseline vs `class_weight='balanced'` vs
  SMOTE (applied only to the training fold via an `imblearn` Pipeline, to
  avoid leakage). SMOTE gave the best F1 balance in this run.
- **Tuning**: `GridSearchCV` over Random Forest's `n_estimators`, `max_depth`,
  `max_features`, with `oob_score=True` reported for the best model.
- **Regression side-task**: predicts `fare` from the remaining features with
  a multivariate Linear Regression; reports MAE, RMSE, R², Adjusted R², and a
  residual plot (`plots/regression_residuals.png`) that shows heteroscedasticity
  -- prediction error grows for higher predicted fares.
- **Final recommendation**: the classifier with the highest AUC is recommended
  for deployment (see the printed final comparison table and written
  recommendation in `modeling.py`'s output for exact metric values).
- **Saved artifact**: the best-performing full pipeline (preprocessing +
  model together) is saved with `joblib.dump()` to `best_titanic_pipeline.pkl`,
  and reloaded with `joblib.load()` to confirm it predicts correctly on raw,
  unprocessed input.

## Files in this module

| File | Purpose |
|---|---|
| `eda.py` | Part A: load, profile, clean, EDA charts |
| `modeling.py` | Part B: full modeling pipeline |
| `titanic.csv` | Cleaned dataset (committed offline fallback) |
| `plots/` | All EDA and modeling charts |
| `best_titanic_pipeline.pkl` | Saved best full pipeline (joblib) |

## Status
Module 2 (analytics) complete and tested end-to-end.