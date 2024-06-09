import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list

# Load dataset
mcdonalds = pd.read_csv("mcdonalds.csv")

# Select relevant columns (attributes) and convert them to numerical
attributes = mcdonalds.columns[:11]
MD_x = mcdonalds[attributes].applymap(lambda x: 1 if x == 'Yes' else 0).values

# Perform hierarchical clustering on the attributes
linkage_matrix = linkage(MD_x.T, method='ward')
ordered_attributes = np.array(attributes)[leaves_list(linkage_matrix)]

# Define cluster sizes
cluster_sizes = [470, 257, 324, 402]
cluster_labels = ['Cluster 1: 470 (32%)', 'Cluster 2: 257 (18%)', 'Cluster 3: 324 (22%)', 'Cluster 4: 402 (28%)']

# Create segment profile plots for each cluster
plt.figure(figsize=(12, 10))

for i, cluster_size in enumerate(cluster_sizes):
    plt.subplot(2, 2, i + 1)
    cluster_data = MD_x[:cluster_size, :11]  # Simulated cluster data
    proportions = cluster_data.mean(axis=0)
    
    # Plot bars for each attribute in the ordered list
    plt.barh(range(len(ordered_attributes)), proportions[leaves_list(linkage_matrix)], color='blue')
    plt.yticks(range(len(ordered_attributes)), ordered_attributes)
    plt.xlabel('Proportion')
    plt.title(cluster_labels[i])

plt.tight_layout()
plt.show()
