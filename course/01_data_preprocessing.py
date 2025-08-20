import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("processed.cleveland.data.txt", header=None)

# Add column names
df.columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert all columns to numeric (coerce errors into NaN)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop missing values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Now all numeric columns will work for histograms


# Summary stats
print(df.describe())

# Histograms
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# Scale numeric columns (except target)
numeric_cols = df.columns.drop('target')
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(df.head())
