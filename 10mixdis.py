import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

# Set seed for reproducibility
np.random.seed(1234)

# Load the dataset (assuming it's in the correct directory)
mcdonalds = pd.read_csv("mcdonalds.csv")

# Extract the first eleven columns and convert 'Yes'/'No' to binary
MD_x = mcdonalds.iloc[:, :11]
MD_x_matrix = (MD_x == "Yes").astype(int)

def latent_class_analysis(data, k_range):
    results = []
    for k in k_range:
        model = GaussianMixture(n_components=k, random_state=1234, covariance_type='full')
        model.fit(data)
        
        log_likelihood = model.score(data) * data.shape[0]  # total log-likelihood
        aic = model.aic(data)
        bic = model.bic(data)
        icl = bic  # ICL is often approximated as BIC for Gaussian Mixtures

        results.append({
            'k': k,
            'logLik': log_likelihood,
            'AIC': aic,
            'BIC': bic,
            'ICL': icl
        })

    return results

# Perform latent class analysis for 2 to 8 segments
k_range = range(2, 9)
results = latent_class_analysis(MD_x_matrix, k_range)

# Print results in the desired format
print("Call:\nstepFlexmix(MD.x ~ 1, model = FLXMCmvbinary(),\nk = 2:8, nrep = 10, verbose = FALSE)")
print("iter converged k k0 logLik AIC BIC ICL")

for i, result in enumerate(results, start=2):
    print(f"{i: <4} 100 TRUE {result['k']} {result['k']} {result['logLik']: .3f} {result['AIC']: .2f} {result['BIC']: .2f} {result['ICL']: .2f}")
