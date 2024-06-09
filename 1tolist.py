import pandas as pd

# Load the dataset
mcdonalds = pd.read_csv("mcdonalds.csv")

# Display the column names
print("Column names:", mcdonalds.columns.tolist())

# Display the dimensions of the dataset
print("Dimensions:", mcdonalds.shape)

# Display the first three rows of the dataset
print("First three rows:\n", mcdonalds.head(3))
