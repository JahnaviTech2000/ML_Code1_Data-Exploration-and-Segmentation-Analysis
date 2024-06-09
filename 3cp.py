import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Load the dataset
mcdonalds = pd.read_csv("mcdonalds.csv")

# Extract the first eleven columns
MD_x = mcdonalds.iloc[:, :11]

# Convert the data to a matrix and convert 'Yes' to 1 and 'No' to 0
MD_x_matrix = (MD_x == "Yes").astype(int)

# Perform PCA
pca = PCA()
MD_pca = pca.fit(MD_x_matrix)

# Get the explained variance and cumulative explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Print the summary
print("Importance of components:")
print("Standard deviation:", np.sqrt(pca.explained_variance_))
print("Proportion of Variance:", explained_variance)
print("Cumulative Proportion:", cumulative_variance)
