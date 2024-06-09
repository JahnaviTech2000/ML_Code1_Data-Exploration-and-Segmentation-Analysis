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

# Get the explained variance and components
explained_variance = pca.explained_variance_
components = pca.components_

# Print standard deviations
print("Standard deviations (1, .., p={}):".format(MD_x_matrix.shape[1]))
print(np.round(np.sqrt(explained_variance), 1))

# Print rotation (components)
print("\nRotation (n x k) = ({} x {}):".format(components.shape[1], components.shape[0]))
for i, col in enumerate(MD_x.columns):
    print("{:<12}".format(col), np.round(components[:, i], 2))
