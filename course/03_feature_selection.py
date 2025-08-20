import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression

# 1. Load the cleaned dataset
df = pd.read_csv("processed.cleveland.data.txt")

# 2. Replace '?' with NaN and convert to numeric
df.replace('?', np.nan, inplace=True)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Drop missing values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# 4. Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# 5. Standardize features for some methods
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Method 1: Feature Importance (Random Forest) ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45)
plt.tight_layout()
plt.show()

# --- Method 2: Recursive Feature Elimination (RFE) ---
lr = LogisticRegression(max_iter=1000)
rfe = RFE(lr, n_features_to_select=5)
rfe.fit(X_scaled, y)

print("\nRFE Selected Features:", X.columns[rfe.support_].tolist())

# --- Method 3: Chi-Square Test ---
# Chi2 requires non-negative values, so we use MinMaxScaler
X_minmax = MinMaxScaler().fit_transform(X)
chi2_selector = SelectKBest(chi2, k=5)
chi2_selector.fit(X_minmax, y)

print("\nChi-Square Selected Features:", X.columns[chi2_selector.get_support()].tolist())
