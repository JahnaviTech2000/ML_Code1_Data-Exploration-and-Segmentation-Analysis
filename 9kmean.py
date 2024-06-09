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

# Prepare data for box plot
similarity_data = []
for i in range(4):
    similarity_data.extend([(i + 1, sim) for sim in similarities[i]])

similarity_df = pd.DataFrame(similarity_data, columns=['Cluster', 'Similarity'])

# Plot the box plots of similarity scores for each cluster
plt.figure(figsize=(10, 8))
sns.boxplot(x='Cluster', y='Similarity', data=similarity_df, color='gray')
plt.xlabel('Cluster')
plt.ylabel('Similarity')
plt.title('Box Plot of Similarities for Each Cluster')
plt.show()

# Calculate Segment Level Stability within Solutions (SLSW)
def calculate_slsw(data, labels, k):
    slsw = []
    for i in range(k):
        cluster_data = data[labels == i]
        if len(cluster_data) == 0:
            slsw.append(0)
            continue
        kmeans_temp = KMeans(n_clusters=2, random_state=1234).fit(cluster_data)
        temp_labels = kmeans_temp.labels_
        cluster_1 = cluster_data[temp_labels == 0]
        cluster_2 = cluster_data[temp_labels == 1]
        stability = min(len(cluster_1), len(cluster_2)) / len(cluster_data)
        slsw.append(stability)
    return slsw

# Calculate stability for each segment
MD_r4 = calculate_slsw(MD_x_matrix, labels, 4)

# Plot segment stability
plt.figure(figsize=(8, 6))
plt.bar(range(1, 5), MD_r4, color='gray')
plt.ylim(0, 1)
plt.xlabel('Segment Number')
plt.ylabel('Segment Stability')
plt.title('Segment Level Stability within Solutions')
plt.show()
