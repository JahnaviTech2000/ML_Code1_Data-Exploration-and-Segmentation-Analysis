import pandas as pd
import numpy as np

# Load the dataset
mcdonalds = pd.read_csv("mcdonalds.csv")

# Extract the first eleven columns
MD_x = mcdonalds.iloc[:, :11]

# Convert the data to a matrix and convert 'Yes' to 1 and 'No' to 0
MD_x_matrix = (MD_x == "Yes").astype(int)

# Calculate the column means and round to 2 decimal places
column_means = MD_x_matrix.mean().round(2)
print(column_means)
