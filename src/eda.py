import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
df = pd.read_csv("data/heart_disease.csv")
print(df.describe())
print(df.isnull().sum())
print(df["target"].value_counts(normalize=True))

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.savefig("data/correlation.png")
print("Saved data/correlation.png")

# Histograms
df.hist(figsize=(12,10))
plt.tight_layout()
plt.savefig("data/histograms.png")
print("Saved data/histograms.png")
