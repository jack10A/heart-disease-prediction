import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ----- Load PCA dataset -----
pca_df = pd.read_csv("heart_disease_pca.csv")
X_pca = pca_df.drop('target', axis=1)
y_pca = (pca_df['target'] > 0).astype(int)  # Binary target

# ----- Load Feature-Selected dataset -----
df = pd.read_csv("processed.cleveland.data.txt")

# Replace '?' with NaN and convert to numeric
df.replace('?', np.nan, inplace=True)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Select top features from Step 3
top_features = ['cp', 'thalach', 'oldpeak', 'ca', 'thal']
X_fs = df[top_features]
y_fs = (df['target'] > 0).astype(int)

# Store results for table
results = []

# Function to train and evaluate models
def evaluate_models(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }

    plt.figure(figsize=(7, 5))

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary')
        rec = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        auc = roc_auc_score(y_test, y_prob)

        results.append({
            "Dataset": dataset_name,
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
            "AUC": auc
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name}')
    plt.legend()

    # Save ROC curve image for Streamlit app
    filename = f"roc_curve_{dataset_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"✅ ROC Curve saved as {filename}")

    plt.show()

# Evaluate both datasets
evaluate_models(X_pca, y_pca, "PCA Dataset")
evaluate_models(X_fs, y_fs, "Feature-Selected Dataset")

# Create results DataFrame
results_df = pd.DataFrame(results)

# Show results table
print("\n=== Model Performance Comparison ===")
print(results_df.pivot(index="Model", columns="Dataset", values="Accuracy"))

# Save results to CSV
results_df.to_csv("model_comparison_results.csv", index=False)
print("\n✅ Results saved to model_comparison_results.csv")
