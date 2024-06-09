import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
np.random.seed(1234)

# Load the dataset (assuming it's in the same directory as the script)
mcdonalds = pd.read_csv("mcdonalds.csv")

# Extract the first eleven columns and convert 'Yes'/'No' to binary
MD_x = mcdonalds.iloc[:, :11]
MD_x_matrix = (MD_x == "Yes").astype(int)

# Perform KMeans clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=1234).fit(MD_x_matrix)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Calculate similarities for each point to its respective cluster center
similarities = []
for i in range(4):
    cluster_points = MD_x_matrix[labels == i]
    distances = pairwise_distances_argmin_min(cluster_points, [centers[i]])[1]
    similarities.append(1 - distances)

# Plot the histograms of similarity scores for each cluster
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, sim in enumerate(similarities):
    sns.histplot(sim, bins=10, kde=False, ax=axes[i], color='gray')
    axes[i].set_xlim(0, 1)
    axes[i].set_ylim(0, 35)
    axes[i].set_title(f'Cluster {i + 1}')
    axes[i].set_xlabel('Similarity')
    axes[i].set_ylabel('Percent of Total')

plt.tight_layout()
plt.show()
