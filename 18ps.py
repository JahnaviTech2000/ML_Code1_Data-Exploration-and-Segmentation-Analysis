import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
mcdonalds = pd.read_csv("mcdonalds.csv")

# Preprocess the data: Convert categorical variables to numerical
mcdonalds_processed = pd.get_dummies(mcdonalds.drop(columns=['Age', 'VisitFrequency', 'Gender']), drop_first=True)

# Standardize the data
scaler = StandardScaler()
mcdonalds_scaled = scaler.fit_transform(mcdonalds_processed)

# Apply PCA
pca = PCA(n_components=2)
mcdonalds_pca = pca.fit_transform(mcdonalds_scaled)

# Fit KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(mcdonalds_scaled)

# Create a DataFrame for PCA results and clusters
pca_df = pd.DataFrame(data=mcdonalds_pca, columns=['principal component 1', 'principal component 2'])
pca_df['Cluster'] = clusters

# Plot the clusters
plt.figure(figsize=(10, 8))
for cluster in range(4):
    cluster_data = pca_df[pca_df['Cluster'] == cluster]
    plt.scatter(cluster_data['principal component 1'], cluster_data['principal component 2'], label=f'Cluster {cluster + 1}')

plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.title('Clusters projected onto the first two principal components')
plt.legend()
plt.show()
