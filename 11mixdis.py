import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset (adjust the path to your actual file location)
mcdonalds = pd.read_csv("mcdonalds.csv")

# Extract the first eleven columns and convert 'Yes'/'No' to binary
MD_x = mcdonalds.iloc[:, :11]
MD_x_matrix = (MD_x == "Yes").astype(int)

# Function to fit GMM and calculate information criteria
def fit_gmm_and_calculate_criteria(data, k_range, n_init):
    aic = []
    bic = []
    icl = []  # ICL is approximated using BIC
    models = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, n_init=n_init, random_state=1234)
        gmm.fit(data)
        aic.append(gmm.aic(data))
        bic.append(gmm.bic(data))
        
        # Calculate ICL
        log_likelihoods = gmm.score_samples(data)
        penalty = 0.5 * (k * np.log(data.shape[0]) + k * (data.shape[1] + 1))
        icl.append(np.sum(log_likelihoods - penalty))
        
        models.append(gmm)
    return aic, bic, icl, models

# Define the range of clusters and number of random starts
k_range = range(2, 9)
n_init = 10

# Fit GMM and calculate information criteria
aic, bic, icl, models = fit_gmm_and_calculate_criteria(MD_x_matrix, k_range, n_init)

# Plot the information criteria
plt.figure(figsize=(10, 8))
plt.plot(k_range, aic, label='AIC', marker='o', linestyle='-', color='red')
plt.plot(k_range, bic, label='BIC', marker='s', linestyle='-', color='green')
plt.plot(k_range, icl, label='ICL', marker='d', linestyle='-', color='blue')
plt.xlabel('Number of segments (k)')
plt.ylabel('Value of information criteria (AIC, BIC, ICL)')
plt.title('Information Criteria for Different Numbers of Segments')
plt.legend()
plt.grid(True)
plt.show()

# Choosing the 4-component model and comparing with k-means
best_model_4 = models[2]  # models[2] corresponds to k=4
clusters_mixture = best_model_4.predict(MD_x_matrix)

# Performing k-means with 4 clusters
kmeans_model = KMeans(n_clusters=4, random_state=1234)
kmeans_model.fit(MD_x_matrix)
clusters_kmeans = kmeans_model.predict(MD_x_matrix)

# Cross-tabulation
cross_tab = pd.crosstab(clusters_kmeans, clusters_mixture, rownames=['kmeans'], colnames=['mixture'])
print("Initial cross-tabulation:\n", cross_tab)

# Fit a new mixture model initialized with k-means clusters
gmm_initialized = GaussianMixture(n_components=4, n_init=1, random_state=1234)
gmm_initialized.fit(MD_x_matrix, clusters_kmeans)
clusters_mixture_initialized = gmm_initialized.predict(MD_x_matrix)

# New cross-tabulation
cross_tab_initialized = pd.crosstab(clusters_kmeans, clusters_mixture_initialized, rownames=['kmeans'], colnames=['mixture'])
print("Cross-tabulation after reinitialization with k-means:\n", cross_tab_initialized)

# Log-likelihoods of the models
log_lik_initial = best_model_4.score(MD_x_matrix) * MD_x_matrix.shape[0]
log_lik_reinitialized = gmm_initialized.score(MD_x_matrix) * MD_x_matrix.shape[0]

print(f"\nLog-likelihood of initial model: {log_lik_initial:.3f}")
print(f"Log-likelihood of reinitialized model: {log_lik_reinitialized:.3f}")
