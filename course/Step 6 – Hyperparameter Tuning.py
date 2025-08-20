import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# =====================
# Load Feature-Selected dataset
# =====================
df = pd.read_csv("processed.cleveland.data.txt")

# Replace '?' with NaN and convert to numeric
df.replace('?', np.nan, inplace=True)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Select top features from Step 3 (example)
top_features = ['cp', 'thalach', 'oldpeak', 'ca', 'thal']
X = df[top_features]
y = (df['target'] > 0).astype(int)  # binary target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# =====================
# Logistic Regression - GridSearchCV
# =====================
log_reg = LogisticRegression(max_iter=1000)
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['liblinear', 'saga']
}

grid_lr = GridSearchCV(log_reg, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
grid_lr.fit(X_train, y_train)

best_lr = grid_lr.best_estimator_
print("\nBest Logistic Regression Params:", grid_lr.best_params_)

# =====================
# Random Forest - RandomizedSearchCV
# =====================
rf = RandomForestClassifier()
param_dist_rf = {
    'n_estimators': [50, 100, 200, 300, 400],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rand_rf = RandomizedSearchCV(rf, param_distributions=param_dist_rf, n_iter=30,
                             cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
rand_rf.fit(X_train, y_train)

best_rf = rand_rf.best_estimator_
print("\nBest Random Forest Params:", rand_rf.best_params_)

# =====================
# Evaluate both best models
# =====================
def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n{name} Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

evaluate_model(best_lr, X_train, X_test, y_train, y_test, "Logistic Regression (Tuned)")
evaluate_model(best_rf, X_train, X_test, y_train, y_test, "Random Forest (Tuned)")
