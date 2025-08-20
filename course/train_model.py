import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ----- Load Feature-Selected dataset -----
df = pd.read_csv("processed.cleveland.data.txt")

# Clean dataset
df.replace('?', np.nan, inplace=True)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Top 5 features
top_features = ['cp', 'thalach', 'oldpeak', 'ca', 'thal']
X = df[top_features]
y = (df['target'] > 0).astype(int)  # Binary target

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "best_model.pkl")
print("✅ Model trained and saved as best_model.pkl using top 5 features")
