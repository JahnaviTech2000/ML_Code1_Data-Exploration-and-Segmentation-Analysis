import pandas as pd

# Load the dataset from the CSV file
mcdonalds = pd.read_csv("mcdonalds.csv")

# Reverse the frequency table of the 'Like' variable
like_freq_table = mcdonalds['Like'].value_counts().sort_index(ascending=False)

# Display the reversed frequency table
print(like_freq_table)
