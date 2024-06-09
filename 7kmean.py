import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Set seed for reproducibility
np.random.seed(1234)

# Load the dataset
mcdonalds = pd.read_csv("mcdonalds.csv")

# Extract the first eleven columns and convert 'Yes'/'No' to binary
MD_x = mcdonalds.iloc[:, :11]
MD_x_matrix = (MD_x == "Yes").astype(int)

# Function to perform KMeans clustering and calculate the adjusted Rand index
def boot_kmeans(data, k_range, nrep, nboot):
    rand_indices = {k: [] for k in k_range}
    
    def kmeans_for_k(k):
        kmeans_models = [KMeans(n_clusters=k, random_state=np.random.randint(10000)).fit(data) for _ in range(nrep)]
        labels = [model.labels_ for model in kmeans_models]
        return [adjusted_rand_score(labels[i], labels[j]) for i in range(nrep) for j in range(i + 1, nrep)]
    
    for k in k_range:
        results = Parallel(n_jobs=-1)(delayed(kmeans_for_k)(k) for _ in range(nboot))
        rand_indices[k] = [score for result in results for score in result]
    
    return rand_indices

# Perform bootstrapped KMeans clustering with 2 to 8 clusters, 5 repetitions each, and 50 bootstrap samples
MD_b28 = boot_kmeans(MD_x_matrix, range(2, 9), 5, 50)

# Calculate the mean adjusted Rand index for each number of clusters
mean_rand_indices = {k: np.mean(v) for k, v in MD_b28.items()}

# Prepare data for plotting
x = list(mean_rand_indices.keys())
y = list(mean_rand_indices.values())

# Plot the bar chart
plt.bar(x, y, color='grey')
plt.xlabel('Number of segments')
plt.ylabel('Adjusted Rand Index')
plt.title('Bootstrapped KMeans Clustering: Adjusted Rand Index vs Number of Segments')
plt.show()
