import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# Read data from CSV file into a DataFrame
mcdonalds = pd.read_csv('mcdonalds.csv')

# Set the seed
robjects.r('set.seed(1234)')

# Load the flexmix package
flexmix = importr('flexmix')

# Define the formula string 'f'
f = "Like.n ~ yummy + convenient + spicy + fattening + greasy + fast + cheap + tasty + expensive + healthy + disgusting"

# Fit the finite mixture of linear regression models
MD_reg2 = flexmix.stepFlexmix(f, data=mcdonalds, k=2, nrep=10, verbose=False)

# Print the results
print(MD_reg2)
