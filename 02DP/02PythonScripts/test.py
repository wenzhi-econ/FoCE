import numpy as np
from scipy.stats import lognorm
from scipy.integrate import quad

# Define the integrand function
def integrand(x):
    return np.log(x)  # Example function

# Parameters for the log-normal distribution
mu = 0  # Mean of the underlying normal distribution
sigma = 1  # Standard deviation of the underlying normal distribution

# Monte Carlo Simulation
def monte_carlo_expectation(n_samples=100000):
    samples = lognorm(sigma, scale=np.exp(mu)).rvs(n_samples)
    return np.mean(integrand(samples))

# Gaussian Quadrature
def gaussian_quadrature_expectation():
    def transformed_integrand(x):
        return integrand(np.exp(x)) * lognorm.pdf(np.exp(x), sigma, scale=np.exp(mu)) * np.exp(x)
    
    # Use finite limits for numerical stability
    result, _ = quad(transformed_integrand, -10, 10)
    return result

# Calculate expectations
mc_expectation = monte_carlo_expectation()
gq_expectation = gaussian_quadrature_expectation()

print(f"Monte Carlo Expectation: {mc_expectation}")
print(f"Gaussian Quadrature Expectation: {gq_expectation}")