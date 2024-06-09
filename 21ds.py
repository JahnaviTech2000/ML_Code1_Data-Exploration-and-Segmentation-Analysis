import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load dataset
mcdonalds = pd.read_csv("mcdonalds.csv")

# Preprocess the data: Encode categorical variables
mcdonalds['Gender'] = LabelEncoder().fit_transform(mcdonalds['Gender'])
mcdonalds['VisitFrequency'] = LabelEncoder().fit_transform(mcdonalds['VisitFrequency'])

# Convert 'Like' to numeric by stripping out non-numeric characters
mcdonalds['Like'] = mcdonalds['Like'].str.replace(r'[^-+\d.]', '', regex=True).astype(float)

# Preprocess the data for clustering
mcdonalds_clustering = pd.get_dummies(mcdonalds.drop(columns=['Age', 'VisitFrequency', 'Gender']), drop_first=True)
scaler = StandardScaler()
mcdonalds_scaled = scaler.fit_transform(mcdonalds_clustering)

# Fit KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
mcdonalds['Cluster'] = kmeans.fit_predict(mcdonalds_scaled)

# Create a new binary variable indicating whether each observation belongs to cluster 3
mcdonalds['IsCluster3'] = (mcdonalds['Cluster'] == 3).astype(int)

# Define the predictors and the target variable
predictors = ['Like', 'Age', 'VisitFrequency', 'Gender']
X = mcdonalds[predictors]
y = mcdonalds['IsCluster3']

# Train a decision tree classifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=predictors, class_names=['Not Cluster 3', 'Cluster 3'], filled=True, rounded=True)
plt.show()
