import pandas as pd

# Load the dataset from the CSV file
mcdonalds = pd.read_csv("mcdonalds.csv")

# Convert the 'Like' column to numeric, ignoring errors
mcdonalds['Like'] = pd.to_numeric(mcdonalds['Like'], errors='coerce')

# Subtract each value of 'Like' from 6
mcdonalds['Like.n'] = 6 - mcdonalds['Like']

# Display the frequency table of the new variable 'Like.n'
like_n_freq_table = mcdonalds['Like.n'].value_counts().sort_index()
print(like_n_freq_table)
