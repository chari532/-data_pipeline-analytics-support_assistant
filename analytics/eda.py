import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.makedirs("plots", exist_ok=True)

df = sns.load_dataset("titanic")

df.info()

print(df.describe())

print(df.shape)

df.to_csv("titanic.csv", index=False)

missing_pct = (df.isna().mean() * 100).round(2)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
print("Missing % per column:")
print(missing_pct)

df = pd.read_csv("titanic.csv")

df = df.drop(columns = ["deck"])

df["age"] = df["age"].fillna(df["age"].median())

df = df.dropna(subset=["embarked", "embark_town"])

print(df.isna().sum()[df.isna().sum() > 0])

df.to_csv("titanic.csv", index=False)

fig, axes = plt.subplots(2,2, figsize=(12,8))
axes[0,0].hist(df["age"],bins=30,color = "blue")
axes[0,0].set_title("Age - Histogram")
sns.boxplot(x=df["age"], ax=axes[0,1], color="blue")
axes[0,1].set_title("Age - Box Plot")
axes[1,0].hist(df["fare"], bins=30, color="orange")
axes[1,0].set_title("Fare - Histogram")
sns.boxplot(x=df["fare"], ax=axes[1,1], color="orange")
axes[1,1].set_title("Fare - Box Plot")
plt.tight_layout()
plt.savefig("plots/univariate_age_fare.png")
plt.close()
print("Saved plots/univariate_age_fare.png")

def remove_outlier(col_name):
  Q1 = col_name.quantile(0.25)
  Q3 = col_name.quantile(0.75)
  IQR = Q3 -Q1
  lower = Q1 - 1.5 * IQR
  upper = Q3 + 1.5 * IQR
  outliers = col_name[(col_name < lower) | (col_name > upper)]
  return len(outliers), lower, upper

age_count, age_low, age_up = remove_outlier(df["age"])
fare_count, fare_low, fare_up = remove_outlier(df["fare"])
print(f"Age outliers (IQR rule): {age_count}  (bounds: [{age_low:.2f}, {age_up:.2f}])")
print(f"Fare outliers (IQR rule): {fare_count}  (bounds: [{fare_low:.2f}, {fare_up:.2f}])")

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]

if fare_mean > fare_median > fare_mode:
  result = "right-skewed(mean > median > mode)"
elif fare_mean < fare_median < fare_mode:
  result = "left-skewed(mean < median < mode)"
else:
  result = "roughly symmetric"
print(f"\nFare -> mean: {fare_mean:.2f}, median: {fare_median:.2f}, mode: {fare_mode:.2f}")
print(f"Fare distribution is {result}")

servival_sex = df[df["sex"] == "female"]["survived"].mean(), df[df["sex"] == "male"]["survived"].mean()
print(f"\nSurvival rate by sex -> female: {servival_sex[0]:.3f}, male: {servival_sex[1]:.3f}")

servival_class = {}
for pclass in sorted(df["pclass"].unique()):
  rate = df[df["pclass"] == pclass]["survived"].mean()
  servival_class[pclass] = rate
  print(f"Survival rate by pclass {pclass}: {rate:.3f}")

for sex in ["female", "male"]:
  for pclass in sorted(df["pclass"].unique()):
    mask = (df["sex"] == sex) & (df["pclass"]== pclass)
    rate = df[mask]["survived"].mean()
    print(f"  sex={sex}, pclass={pclass}: {rate:.3f}")

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df[corr_cols].corr()
print("\nCorrelation matrix (6x6):")
print(corr_matrix.round(3))

plt.figure(figsize=(7,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlaction Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi = 100)
plt.close()

corr_pairs = []
for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        corr_pairs.append((corr_cols[i], corr_cols[j], corr_matrix.iloc[i, j]))
corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
print("\nTop 2 strongest correlations (by absolute value):")
for pair in corr_pairs_sorted[:2]:
    print(f"  {pair[0]} <-> {pair[1]}: {pair[2]:.3f}")

print("""
Interpretation: 'survived' and 'pclass' show a moderate negative correlation passengers in higher-numbered (lower) classes had lower survival odds. 'fare' and 'pclass' show a strong negative correlation, which makes sense since fare is essentially a proxy for class (1st class tickets cost more).""")

fig, ax= plt.subplots(figsize=(7,5))
sns.barplot(data=df,x="pclass", y="survived",hue="sex", ax=ax)
ax.set_title("Survival rate by class and sex")
plt.tight_layout()
plt.savefig("plots/class_sex.png", dpi=100)
plt.close()
print("""Interpretation: Across all three passenger classes, women survived at a much higher rate than men. The gap is largest in 1st and 2nd class, where women's survival rate is close to certain, while male survival drops sharply as class number increases -- suggesting both gender-based evacuation priority and class privilege combined to shape outcomes.""")

fig, ax = plt.subplots(figsize=(7,5))
sns.boxplot(data=df, x="survived", y = "age", ax=ax)
ax.set_title("Age distribution by survival")
plt.tight_layout()
plt.savefig("plots/age_survival.png", dpi=100)
plt.close()
print("""Interpretation: Survivors and non-survivors have broadly similar median ages (~28), but the survivor group shows a slightly higher concentration of very
young children -- its lower whisker extends almost to age 0, while young
children among non-survivors appear only as rare outlier points. Among older
passengers, the non-survivor group has a denser cluster of outliers in the
55-74 range, while the single oldest passenger in the dataset (around 80)
happened to survive.""")


fig, ax = plt.subplots(figsize=(7,5))
sns.scatterplot(data=df, x="age", y="fare", hue="survived", ax = ax)
ax.set_title("Age vs Fare, colored by survival")
plt.tight_layout()
plt.savefig("plots/age_fare_survival.png", dpi=100)
plt.close()
print("""
Interpretation: Passengers who paid higher fares (upper portion of the chart) survived at a visibly higher rate than those who paid low fares, regardless of age. This reinforces the class/fare effect seen in the correlation heatma money bought a materially better chance of survival on this ship.
""")

fig, ax = plt.subplots(figsize=(7,5))
sns.barplot(data=df, x="embarked", y="survived", ax=ax)
ax.set_title("Survival rate by port of embarkation")
plt.tight_layout()
plt.savefig("plots/embarked.png", dpi=100)
plt.close()
print("""
Interpretation: Passengers who embarked at Cherbourg ('C') had a noticeably higher survival rate than those from Southampton ('S') or Queenstown. This is likely a confound rather than a causal effect: Cherbourg passengers were disproportionately 1st class, so this chart is best read together with the class based charts above rather than in isolation.""")

print("\n" + "=" * 70)
print("Z score standardization (EDA sanity check)")
print("=" * 70)

for col in ["age", "fare"]:
    before_mean, before_std = df[col].mean(), df[col].std()
    z = (df[col] - before_mean) / before_std
    print(f"\n{col} BEFORE mean: {before_mean:.3f}, std: {before_std:.3f}")
    print(f"{col} AFTER    mean: {z.mean():.3f}, std: {z.std():.3f}")