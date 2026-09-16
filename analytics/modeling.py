import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.makedirs("plots", exist_ok=True)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix,mean_absolute_error, mean_squared_error, r2_score
import joblib

df = pd.read_csv("titanic.csv")

df["survived"].value_counts(normalize=True).round(3)

feature = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = "survived"

data = df[feature + [target]].copy()

X = data.drop(target, axis=1)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size = 0.2, random_state = 42, stratify = y
)

numeric_features = ["pclass","age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ]), categorical_features)

])


lr = Pipeline([("prep", preprocessor), ("clf", LogisticRegression())])
dt = Pipeline([("prep", preprocessor), ("clf", DecisionTreeClassifier(max_depth=4, random_state=42))])
rf = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

feature_names_out = dt.named_steps["prep"].get_feature_names_out()

plt.figure(figsize=(20,10))
plot_tree(dt.named_steps["clf"], feature_names=feature_names_out,
          class_names=["Did not survive", "Survived"], filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree")
plt.savefig("plots/decision_tree.png")
plt.close()
models = {"Logistic Regression": lr, "Decision Tree": dt, "Random Forest": rf}
results = []

plt.figure(figsize=(7, 6))
for name, m in models.items():
  y_pred = m.predict(X_test)
  y_prob = m.predict_proba(X_test)[:, 1]

  cm = confusion_matrix(y_test, y_pred)
  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred)
  rec = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  auc = roc_auc_score(y_test, y_prob)

  print(f"\n--- {name} ---")
  print("Confusion Matrix:")
  print(cm)
  print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

  fpr, tpr, _ = roc_curve(y_test, y_prob)
  plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

  results.append({
        "Model": name, "Accuracy": round(acc, 3), "Precision": round(prec, 3),
        "Recall": round(rec, 3), "F1": round(f1, 3), "AUC": round(auc, 3)
    })


plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - All Models")
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_curves.png", dpi=100)
plt.close()
print("\nSaved plots/roc_curves.png")

comparison_df = pd.DataFrame(results)
print("\n Model Comparison Table")
print(comparison_df.to_string(index=False))

imbalance_results = []

pipe_baseline = Pipeline([("prep", preprocessor), ("clf", LogisticRegression()) ])
pipe_baseline.fit(X_train, y_train)
pred_base = pipe_baseline.predict(X_test)
imbalance_results.append({
    "Strategy" : "Baseline",
    "Precision" : round(precision_score(y_test, pred_base),3),
    "Recall" : round(recall_score(y_test, pred_base),3),
    "F1" : round(f1_score(y_test, pred_base),3)

})

pipe_balanced = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(class_weight = "balanced"))])
pipe_balanced.fit(X_train, y_train)
pred_bal = pipe_balanced.predict(X_test)
imbalance_results.append({
    "Strategy" : "class_weight =balanced",
    "Precision" : round(precision_score(y_test, pred_bal),3),
    "Recall" : round(recall_score(y_test, pred_bal),3),
    "F1" : round(f1_score(y_test, pred_bal),3)

})

pipe_smote  = ImbPipeline([
    ("prep", preprocessor),
    ("smote", SMOTE(random_state = 42)),
     ("clf", LogisticRegression()),
])

pipe_smote.fit(X_train, y_train)
pred_smote = pipe_smote.predict(X_test)
imbalance_results.append({
    "Strategy" : "SMOTE",
    "Precision" : round(precision_score(y_test, pred_smote), 3),
    "Recall" : round(recall_score(y_test, pred_smote), 3),
    "F1" : round(f1_score(y_test, pred_smote), 3),

})

imbalance_df = pd.DataFrame(imbalance_results)
print(imbalance_df.to_string(index=False))

best_f1_row = imbalance_df.loc[imbalance_df["F1"].idxmax()]
print(f"""
Conclusion: {best_f1_row['Strategy']} achieved the best F1 score ({best_f1_row['F1']}).
Baseline logistic regression favors the majority class (not-survived) since it
optimizes overall accuracy by default. class_weight='balanced' and SMOTE both
push the model to pay more attention to the minority (survived) class, usually
trading a little precision for better recall. Given the strategy with the
highest F1 above, this represents the best balance of precision and recall
for this moderately imbalanced (62/38) dataset.
""")

rf_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(oob_score=True, random_state = 42)),
])

param_grid = {
    "clf__n_estimators": [100,200],
    "clf__max_depth" : [5, 10, None],
    "clf__max_features" : ["sqrt", "log2"],

}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv = 5, scoring = "accuracy", n_jobs = -1)
grid_search.fit(X_train, y_train)

best_rf_pipeline = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-val accuracy: {grid_search.best_score_:.3f}")
oob = best_rf_pipeline.named_steps["clf"].oob_score_

print(f"OOB score of best model: {oob:.3f}")

reg_features = ["pclass", "sex", "age", "sibsp", "parch", "embarked", "survived"]
reg_target = "fare"

reg_data = df[reg_features + [reg_target]].copy()
Xr = reg_data.drop(columns=[reg_target])
yr = reg_data[reg_target]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

reg_numeric = ["pclass", "age", "sibsp", "parch", "survived"]
reg_categorical = ["sex", "embarked"]

reg_preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), reg_numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))]), reg_categorical),
])

reg_pipeline = Pipeline([
    ("prep", reg_preprocessor),
    ("model", LinearRegression()),
])

reg_pipeline.fit(Xr_train, yr_train)
yr_pred = reg_pipeline.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
mse = mean_squared_error(yr_test, yr_pred)
rmse = np.sqrt(mse)
r2 = r2_score(yr_test, yr_pred)

n = len(yr_test)
p = Xr_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"\nMAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2: {r2:.3f}")
print(f"Adjusted R2: {adj_r2:.3f}")

residuals = yr_test - yr_pred
plt.figure(figsize=(7, 5))
plt.scatter(yr_pred, residuals, alpha=0.5)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted fare")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residual Plot - Fare Regression")
plt.tight_layout()
plt.savefig("plots/regression_residuals.png", dpi=100)
plt.close()
print("plots/Saved regression_residuals.png")

print("""
Heteroscedasticity check: the residual plot shows a funnel shape -- residuals
are tightly clustered near zero for low predicted fares, but spread out much
more widely (both positive and negative) as predicted fare increases. This
non-random, fanning-out spread indicates heteroscedasticity: the model's
prediction error is not constant across the range of fare, and it is
noticeably less reliable for high-fare passengers than for low-fare ones.
""")

print("\n" + "=" * 70)
print("TASK 14: Final model comparison")
print("=" * 70)

tuned_pred = best_rf_pipeline.predict(X_test)
tuned_prob = best_rf_pipeline.predict_proba(X_test)[:, 1]
tuned_row = {
    "Model": "Random Forest (tuned)",
    "Accuracy": round(accuracy_score(y_test, tuned_pred), 3),
    "Precision": round(precision_score(y_test, tuned_pred), 3),
    "Recall": round(recall_score(y_test, tuned_pred), 3),
    "F1": round(f1_score(y_test, tuned_pred), 3),
    "AUC": round(roc_auc_score(y_test, tuned_prob), 3),
}
classification_comparison = pd.DataFrame(results + [tuned_row])

regression_comparison = pd.DataFrame([{
    "Model": "Linear Regression (fare)",
    "MAE": round(mae, 3),
    "RMSE": round(rmse, 3),
    "R2": round(r2, 3),
    "Adjusted_R2": round(adj_r2, 3),
}])


print(classification_comparison.to_string(index=False))


print(regression_comparison.to_string(index=False))

best_class_row = classification_comparison.loc[classification_comparison["AUC"].idxmax()]


best_pipeline = lr 

joblib.dump(best_pipeline, "best_titanic_pipeline.pkl")
print("Saved plots/best_titanic_pipeline.pkl")


loaded_pipeline = joblib.load("best_titanic_pipeline.pkl")

raw_sample = X_test.iloc[:3] 
original_preds = best_pipeline.predict(raw_sample)
reloaded_preds = loaded_pipeline.predict(raw_sample)

print("\nRaw sample input:")
print(raw_sample)
print(f"\nOriginal pipeline predictions: {original_preds}")
print(f"Reloaded pipeline predictions: {reloaded_preds}")
print(f"Predictions match: {(original_preds == reloaded_preds).all()}")