import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import adjusted_rand_score

# =====================
# Load and preprocess original dataset
# =====================
df = pd.read_csv("processed.cleveland.data.txt")

# Replace '?' with NaN and convert to numeric
df.replace('?', np.nan, inplace=True)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Binary target for comparison
y_true = (df['target'] > 0).astype(int)

# Drop target for clustering
X_original = df.drop('target', axis=1)

# Standardize
scaler = StandardScaler()
X_original_scaled = scaler.fit_transform(X_original)

# =====================
# Load PCA dataset
# =====================
pca_df = pd.read_csv("heart_disease_pca.csv")
X_pca = pca_df.drop('target', axis=1)
y_pca = (pca_df['target'] > 0).astype(int)  # binary

# =====================
# Function for K-Means + Hierarchical
# =====================
def clustering_analysis(X, y, dataset_name):
    print(f"\n===== {dataset_name} =====")

    # ----- K-Means -----
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Elbow plot
    plt.figure(figsize=(5, 4))
    plt.plot(range(1, 11), wcss, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.title(f'Elbow Method - {dataset_name}')
    plt.show()

    # Final KMeans with k=2
    kmeans_final = KMeans(n_clusters=2, random_state=42)
    clusters_kmeans = kmeans_final.fit_predict(X)
    ari_kmeans = adjusted_rand_score(y, clusters_kmeans)
    print(f"Adjusted Rand Index (K-Means vs Target): {ari_kmeans:.4f}")

    # ----- Hierarchical Clustering -----
    linked = linkage(X, method='ward')

    plt.figure(figsize=(8, 5))
    dendrogram(linked, truncate_mode='level', p=5)
    plt.title(f'Hierarchical Clustering Dendrogram - {dataset_name}')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()

    clusters_hier = fcluster(linked, t=2, criterion='maxclust')
    ari_hier = adjusted_rand_score(y, clusters_hier)
    print(f"Adjusted Rand Index (Hierarchical vs Target): {ari_hier:.4f}")

# =====================
# Run analysis on both datasets
# =====================
clustering_analysis(X_original_scaled, y_true, "Original Dataset")
clustering_analysis(X_pca, y_pca, "PCA Dataset")
