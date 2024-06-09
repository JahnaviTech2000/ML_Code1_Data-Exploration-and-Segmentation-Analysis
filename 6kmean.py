import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Set seed for reproducibility
np.random.seed(1234)

# Load the dataset
mcdonalds = pd.read_csv("mcdonalds.csv")

# Extract the first eleven columns and convert 'Yes'/'No' to binary
MD_x = mcdonalds.iloc[:, :11]
MD_x_matrix = (MD_x == "Yes").astype(int)

# Function to perform KMeans clustering for a range of cluster numbers
def step_kmeans(data, k_range, nrep):
    results = {}
    for k in k_range:
        best_inertia = None
        best_model = None
        for _ in range(nrep):
            model = KMeans(n_clusters=k, random_state=np.random.randint(10000)).fit(data)
            if best_inertia is None or model.inertia_ < best_inertia:
                best_inertia = model.inertia_
                best_model = model
        results[k] = best_model
    return results

# Perform KMeans clustering with 2 to 8 clusters, 10 repetitions each
MD_km28 = step_kmeans(MD_x_matrix, range(2, 9), 10)

# Function to relabel clusters to minimize label switching issue
def relabel_kmeans(results):
    base_labels = results[min(results.keys())].labels_
    for k in sorted(results.keys()):
        model = results[k]
        new_labels = model.labels_
        encoder = LabelEncoder()
        encoder.fit(new_labels)
        results[k].labels_ = encoder.transform(new_labels)
    return results

# Relabel the clusters
MD_km28 = relabel_kmeans(MD_km28)

# Prepare data for plotting
x = list(MD_km28.keys())
y = [model.inertia_ for model in MD_km28.values()]

# Plot the bar chart
plt.bar(x, y, color='grey')
plt.xlabel('Number of segments')
plt.ylabel('Inertia')
plt.title('KMeans Clustering: Number of Segments vs Inertia')
plt.show()
