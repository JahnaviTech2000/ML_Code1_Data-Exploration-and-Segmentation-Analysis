import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
mcdonalds = pd.read_csv("mcdonalds.csv")

# Extract the first eleven columns
MD_x = mcdonalds.iloc[:, :11]

# Convert the data to a matrix and convert 'Yes' to 1 and 'No' to 0
MD_x_matrix = (MD_x == "Yes").astype(int)

# Perform PCA
pca = PCA()
MD_pca = pca.fit_transform(MD_x_matrix)

# Plot the PCA results
plt.scatter(MD_pca[:, 0], MD_pca[:, 1], color='grey')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of McDonald\'s Data')

# Function to plot projection axes
def plot_projection_axes(pca, ax):
    for i, (component, variance) in enumerate(zip(pca.components_, pca.explained_variance_)):
        component = component * np.sqrt(variance)  # scale component by sqrt of explained variance
        start, end = np.zeros_like(component), component
        ax.annotate(
            '',
            xy=end[:2],  # use only the first two dimensions
            xycoords='data',
            xytext=start[:2],
            textcoords='data',
            arrowprops=dict(facecolor='red', width=1.5, headwidth=6)
        )
        ax.text(end[0], end[1], f'PC{i+1}', ha='center', va='center', fontsize=12, color='red')

# Create the plot
fig, ax = plt.subplots()
ax.scatter(MD_pca[:, 0], MD_pca[:, 1], color='grey')
plot_projection_axes(pca, ax)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('PCA of McDonald\'s Data with Projection Axes')
plt.show()
