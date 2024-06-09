import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
mcdonalds = pd.read_csv("mcdonalds.csv")

# Convert 'Like' to numeric by stripping out non-numeric characters
mcdonalds['Like'] = mcdonalds['Like'].str.replace(r'[^-+\d.]', '', regex=True).astype(float)

# Encode categorical variables
mcdonalds['Gender'] = LabelEncoder().fit_transform(mcdonalds['Gender'])
mcdonalds['VisitFrequency'] = LabelEncoder().fit_transform(mcdonalds['VisitFrequency'])

# Preprocess the data for clustering
mcdonalds_clustering = pd.get_dummies(mcdonalds.drop(columns=['Age', 'VisitFrequency', 'Gender']), drop_first=True)
scaler = StandardScaler()
mcdonalds_scaled = scaler.fit_transform(mcdonalds_clustering)

# Fit KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
mcdonalds['Cluster'] = kmeans.fit_predict(mcdonalds_scaled)

# Calculate mean visit frequency, liking, and proportion of females for each cluster
visit_mean = mcdonalds.groupby('Cluster')['VisitFrequency'].mean()
like_mean = mcdonalds.groupby('Cluster')['Like'].mean()
female_mean = mcdonalds.groupby('Cluster')['Gender'].mean()  # Assuming 'Gender' is encoded as 1 for Female and 0 for Male

# Ensure all clusters are represented
clusters = range(4)

# Textual output
print("Mean Visit Frequency by Cluster:")
print(visit_mean)
print("\nMean Liking by Cluster:")
print(like_mean)
print("\nProportion of Females by Cluster:")
print(female_mean)

# Plotting
plt.figure(figsize=(10, 8))

# Bubble plot
plt.scatter(visit_mean, like_mean, s=1000 * female_mean, alpha=0.5)
plt.xlim(0, 4.5)
plt.ylim(-3, 3)
plt.xlabel('Mean Visit Frequency')
plt.ylabel('Mean Liking')
plt.title('Segment Evaluation Plot')

# Add cluster labels
for i in clusters:
    plt.annotate(i+1, (visit_mean[i], like_mean[i]), fontsize=12, ha='right')

plt.show()
