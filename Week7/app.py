import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

# Title and introduction
st.title("Clustering Comparison: EM vs KMeans")
st.write("This app compares the clustering results between the Expectation-Maximization (EM) algorithm and the KMeans algorithm using the Iris dataset.")

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Preprocess the data
X = data.iloc[:, :-1]  # Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply the EM algorithm for clustering
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_clusters = gmm.fit_predict(X_scaled)

# Apply the k-Means algorithm for clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_scaled)

# Add the clustering results to the dataframe
data['GMM_Cluster'] = gmm_clusters
data['KMeans_Cluster'] = kmeans_clusters

# Display clustering results
st.write("## Clustering Results")
st.write("### Expectation-Maximization (EM) Clustering:")
st.write(data[['target', 'GMM_Cluster']])
st.write("### KMeans Clustering:")
st.write(data[['target', 'KMeans_Cluster']])

# Plot the clustering results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# EM Clustering
axes[0].set_title('EM Clustering')
for target in range(3):
    axes[0].scatter(data[data['target'] == target]['sepal length (cm)'],
                    data[data['target'] == target]['sepal width (cm)'],
                    c=data[data['target'] == target]['GMM_Cluster'],
                    cmap='viridis', label=f'Target {target}', s=40)
axes[0].set_xlabel('Sepal Length (cm)')
axes[0].set_ylabel('Sepal Width (cm)')
axes[0].legend()

# KMeans Clustering
axes[1].set_title('KMeans Clustering')
for target in range(3):
    axes[1].scatter(data[data['target'] == target]['sepal length (cm)'],
                    data[data['target'] == target]['sepal width (cm)'],
                    c=data[data['target'] == target]['KMeans_Cluster'],
                    cmap='viridis', label=f'Target {target}', s=40)
axes[1].set_xlabel('Sepal Length (cm)')
axes[1].set_ylabel('Sepal Width (cm)')
axes[1].legend()

# Calculate adjusted Rand index for EM clustering
gmm_adjusted_rand_score = adjusted_rand_score(data['target'], gmm_clusters)
st.write(f"Adjusted Rand Index for EM Clustering: {gmm_adjusted_rand_score:.4f}")

# Calculate adjusted Rand index for k-Means clustering
kmeans_adjusted_rand_score = adjusted_rand_score(data['target'], kmeans_clusters)
st.write(f"Adjusted Rand Index for KMeans Clustering: {kmeans_adjusted_rand_score:.4f}")

# Show the plots
st.pyplot(fig)
