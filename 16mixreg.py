import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load dataset
mcdonalds = pd.read_csv("mcdonalds.csv")

# Ensure 'Like.n' is numeric
mcdonalds['Like.n'] = pd.to_numeric(mcdonalds['Like'].replace({'I hate it!-5': -5, 'I love it!+5': 5}), errors='coerce')

# Convert categorical variables to numerical format
attributes = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']
for attribute in attributes:
    mcdonalds[attribute + 'Yes'] = mcdonalds[attribute].replace({'Yes': 1, 'No': 0})

# Drop any rows with missing values
mcdonalds = mcdonalds.dropna()

# Extract features and dependent variable
X = mcdonalds[['yummyYes', 'convenientYes', 'spicyYes', 'fatteningYes', 'greasyYes', 'fastYes', 'cheapYes', 'tastyYes', 'expensiveYes', 'healthyYes', 'disgustingYes']]
y = mcdonalds['Like.n']

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, n_init=10, random_state=1234)
mcdonalds['Cluster'] = gmm.fit_predict(X)

# Fit linear regression models for each cluster
regression_results = {}
for cluster in mcdonalds['Cluster'].unique():
    cluster_data = mcdonalds[mcdonalds['Cluster'] == cluster]
    X_cluster = sm.add_constant(cluster_data[['yummyYes', 'convenientYes', 'spicyYes', 'fatteningYes', 'greasyYes', 'fastYes', 'cheapYes', 'tastyYes', 'expensiveYes', 'healthyYes', 'disgustingYes']].astype(float))
    y_cluster = cluster_data['Like.n']
    model = sm.OLS(y_cluster, X_cluster).fit()
    regression_results[cluster] = model

# Display regression summaries
for cluster, result in regression_results.items():
    print(f"Cluster {cluster+1} Regression Summary:")
    print(result.summary())

# Plotting regression coefficients with significance
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
for i, (cluster, result) in enumerate(regression_results.items()):
    coef = result.params[1:]  # Skip the intercept
    errors = result.bse[1:]
    significance = result.pvalues[1:] < 0.05
    colors = ['darkgrey' if sig else 'lightgrey' for sig in significance]
    
    axes[i].barh(coef.index, coef, xerr=errors, color=colors, edgecolor='black')
    axes[i].set_xlim(-5, 6)
    axes[i].set_title(f'Cluster {cluster+1} Coefficients')
    axes[i].axvline(x=0, color='black', linewidth=0.8, linestyle='--')

plt.tight_layout()
plt.show()
