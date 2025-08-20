import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =====================
# Load data & preprocess
# =====================
df = pd.read_csv("processed.cleveland.data.txt")
df.replace('?', np.nan, inplace=True)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Select top features (from Step 3)
top_features = ['cp', 'thalach', 'oldpeak', 'ca', 'thal']
X = df[top_features]
y = (df['target'] > 0).astype(int)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the final model (Logistic Regression as example)
final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_train, y_train)

# =====================
# Save model & scaler
# =====================
joblib.dump(final_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved successfully!")
