import pandas as pd

# Read data from CSV file into a DataFrame
mcdonalds = pd.read_csv('mcdonalds.csv')

# Extract the column names of the first 11 independent variables
independent_vars = mcdonalds.columns[:11]

# Create a string of column names separated by '+' for the formula
columns_str = '+'.join(independent_vars)

# Construct the formula string with the correct dependent variable name
formula_str = 'Like.n ~ ' + columns_str

# Print the formula string
print(formula_str)
