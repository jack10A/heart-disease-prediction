import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load the cleaned dataset from Step 1
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

# 5. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 7. Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

# 8. Scree plot
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center', label='Individual explained variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance', color='red')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA - Explained Variance')
plt.legend()
plt.show()

# 9. Keep components that explain ~90% variance
n_components = np.argmax(cumulative_variance >= 0.9) + 1
print(f"Optimal number of components: {n_components}")

pca_final = PCA(n_components=n_components)
X_pca_final = pca_final.fit_transform(X_scaled)

# 10. 2D scatter plot of first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_final[:, 0], X_pca_final[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=50)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - First Two Principal Components')
plt.colorbar(label='Target')
plt.show()
# Save PCA-transformed dataset
pca_df = pd.DataFrame(X_pca_final, columns=[f'PC{i+1}' for i in range(X_pca_final.shape[1])])
pca_df['target'] = y.values

pca_df.to_csv("heart_disease_pca.csv", index=False)
print("✅ PCA-transformed dataset saved as heart_disease_pca.csv")

# 11. 3D scatter plot of first three principal components
from mpl_toolkits.mplot3d import Axes3D  # Ensure 3D plotting is enabled

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create scatter plot
sc = ax.scatter(X_pca_final[:, 0], X_pca_final[:, 1], X_pca_final[:, 2],
                c=y, cmap='coolwarm', edgecolor='k', s=50)

# Labels
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA - First Three Principal Components')

# Add colorbar
fig.colorbar(sc, ax=ax, label='Target')

plt.show()

# Save PCA-transformed dataset
pca_df = pd.DataFrame(X_pca_final, columns=[f'PC{i+1}' for i in range(X_pca_final.shape[1])])
pca_df['target'] = y.values

pca_df.to_csv("heart_disease_pca.csv", index=False)
print("✅ PCA-transformed dataset saved as heart_disease_pca.csv")