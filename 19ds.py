import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.graphics.mosaicplot import mosaic

# Load dataset
mcdonalds = pd.read_csv("mcdonalds.csv")

# Preprocess the data: Convert categorical variables to numerical
mcdonalds_processed = pd.get_dummies(mcdonalds.drop(columns=['Age', 'VisitFrequency', 'Gender']), drop_first=True)

# Standardize the data
scaler = StandardScaler()
mcdonalds_scaled = scaler.fit_transform(mcdonalds_processed)

# Fit KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
mcdonalds['Cluster'] = kmeans.fit_predict(mcdonalds_scaled)

# Create a table of cluster assignments vs. 'Like' variable
cluster_like_table = pd.crosstab(mcdonalds['Cluster'], mcdonalds['Like'])

# Define a function to format labels
def labelizer(key):
    return f'{key[0]}, {key[1]}'

# Plot the mosaic plot
fig, _ = mosaic(cluster_like_table.stack(), gap=0.02, labelizer=labelizer, title='', axes_label=False)
plt.xlabel('Segment Number')
plt.ylabel('Count')
plt.show()
