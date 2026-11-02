"""
monte-carlo simulation for comparing estimators of a random variable
defined as mu^2 + sigma^2 of the random variable X~N(mu, sigma squared)

where estimator 1 is given by the mean of X^2 since E[X^2] = mu squared + sigma squared
and estimator 2 is given by the sample mean of X squared plus the sample variance
"""

import numpy as np
import matplotlib.pyplot as plt

# Set parameters
mu = 0  # population mean
sigma_sq = 1  # population variance
sigma = np.sqrt(sigma_sq)
n = 100  # sample size
M = 1000  # number of simulations

true_value = mu ** 2 + sigma_sq  # true value we're estimating

# Storage for estimates
plugin_estimates = np.zeros(M)
sample_analog_estimates = np.zeros(M)

# Run simulations
np.random.seed(42)  # for reproducibility
for m in range(M):
    # Generate sample from N(mu, sigma^2)
    X = np.random.normal(mu, sigma, n)

    # Plug-in estimator: X̄² + S²
    X_bar = np.mean(X)
    S_sq = np.var(X, ddof=1)  # sample variance with n-1
    plugin_estimates[m] = X_bar ** 2 + S_sq

    # Sample analog estimator: (1/n)ΣXᵢ²
    sample_analog_estimates[m] = np.mean(X ** 2)

# Calculate bias, variance, and MSE
plugin_bias = np.mean(plugin_estimates) - true_value
plugin_var = np.var(plugin_estimates)
plugin_mse = np.mean((plugin_estimates - true_value) ** 2)

analog_bias = np.mean(sample_analog_estimates) - true_value
analog_var = np.var(sample_analog_estimates)
analog_mse = np.mean((sample_analog_estimates - true_value) ** 2)

# Print results
print(f"True value: {true_value}")
print(f"\nPlug-in Estimator (X̄² + S²):")
print(f"  Mean: {np.mean(plugin_estimates):.4f}")
print(f"  Bias: {plugin_bias:.4f} (theoretical: {sigma_sq / n:.4f})")
print(f"  Variance: {plugin_var:.4f}")
print(f"  MSE: {plugin_mse:.4f}")

print(f"\nSample Analog Estimator ((1/n)ΣXᵢ²):")
print(f"  Mean: {np.mean(sample_analog_estimates):.4f}")
print(f"  Bias: {analog_bias:.4f}")
print(f"  Variance: {analog_var:.4f}")
print(f"  MSE: {analog_mse:.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(plugin_estimates, bins=50, alpha=0.6, label='Plug-in', density=True)
axes[0].hist(sample_analog_estimates, bins=50, alpha=0.6, label='Sample Analog', density=True)
axes[0].axvline(true_value, color='red', linestyle='--', linewidth=2, label='True Value')
axes[0].set_xlabel('Estimate')
axes[0].set_ylabel('Density')
axes[0].set_title('Distribution of Estimates')
axes[0].legend()

axes[1].boxplot([plugin_estimates, sample_analog_estimates], labels=['Plug-in', 'Sample Analog'])
axes[1].axhline(true_value, color='red', linestyle='--', linewidth=2, label='True Value')
axes[1].set_ylabel('Estimate')
axes[1].set_title('Comparison of Estimators')
axes[1].legend()

plt.tight_layout()
plt.show()

# Sort the estimates
plugin_sorted = np.sort(plugin_estimates)
analog_sorted = np.sort(sample_analog_estimates)

# Create empirical CDF values
empirical_cdf = np.arange(1, M+1) / M

# Plot
plt.figure(figsize=(10, 6))
plt.plot(plugin_sorted, empirical_cdf, label='Plug-in Estimator', alpha=0.7)
plt.plot(analog_sorted, empirical_cdf, label='Sample Analog Estimator', alpha=0.7)
plt.axvline(true_value, color='red', linestyle='--', linewidth=2, label='True Value')
plt.xlabel('Estimate Value')
plt.ylabel('Empirical CDF')
plt.title('Empirical CDFs of Both Estimators')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plugin_vs_analog_estimator.png')
plt.show()
