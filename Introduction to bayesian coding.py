# Importing necessary libraries
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import bambi as bmb

# Set random seed for reproducibility
np.random.seed(42)

# ---- Bayesian Linear Regression Using PyMC3 ----

# Simulating some data for a linear regression
data_df1 = pd.DataFrame({
    'y': [2.3, 1.9, 3.7, 2.5, 4.1, 5.2],
    'x_1': [1.1, 0.8, 1.4, 1.0, 2.0, 1.5],
    'x_2': [0.7, 1.2, 0.5, 1.1, 1.3, 0.9]
})

# Running a basic Bayesian linear regression model using bambi
model_1 = bmb.Model("y ~ x_1 + x_2", data_df1)
M1 = model_1.fit(draws=2000, chains=4)
az.summary(M1)

# Visualizing Bayesian linear models and posterior distributions
az.plot_trace(M1)
az.plot_posterior(M1)

# ---- Inspecting Prior Defaults ----
# Checking priors for the model
print(model_1.prior)

# ---- Changing Default Settings ----
# Adjusting iterations, warmup, chains, and setting a custom prior
model_2 = bmb.Model("y ~ x_1 + x_2", data_df1)
M2 = model_2.fit(draws=2500, chains=4, tune=500,
                 priors={"x_1": "Normal(0, 100)", "x_2": "Normal(0, 100)"})
az.summary(M2)

# ---- Viewing Coefficients ----
# Using bambi’s model object to inspect fixed effects coefficients
print(M1.posterior.mean().filter(like="x_"))

# ---- Bayesian R^2 ----
# Bayesian R-squared to evaluate fit
r2_M1 = az.r2_score(M1)
r2_M2 = az.r2_score(M2)

# ---- Prior vs Posterior Plots ----

# Creating data for a model with only intercept
data_df_prior = pd.DataFrame({"x": np.random.normal(size=10)})

model_prior = bmb.Model("x ~ 1", data_df_prior)
A = model_prior.fit(draws=2000, chains=4)

model_posterior = bmb.Model("x ~ 1", data_df_prior, priors={"x": "Normal(0, 1)"})
B = model_posterior.fit(draws=2000, chains=4)

# Plotting prior and posterior densities
samples = pd.concat([A.posterior.stack(samples=("chain", "draw")).assign(type="prior"),
                     B.posterior.stack(samples=("chain", "draw")).assign(type="posterior")])

samples = samples.melt(var_name="parameter", value_name="value", ignore_index=False)
plt.figure(figsize=(10, 6))
az.plot_kde(samples, hue="type")
plt.show()

# ---- Custom Priors for Coefficients ----
# Custom priors for x_1 and x_2 coefficients
model_3 = bmb.Model("y ~ x_1 + x_2", data_df1)
M3 = model_3.fit(priors={"x_1": "Normal(0, 10)", "x_2": "Normal(0, 10)", "sigma": "StudentT(1, 0, 30)"})

# ---- Bayesian Logistic Regression ----
# Simulating binary outcome data
data_df2 = pd.DataFrame({
    'y': np.random.binomial(1, 0.5, 100),
    'x_1': np.random.normal(170, 10, 100),
    'x_2': np.random.normal(25, 5, 100)
})

model_logistic = bmb.Model("y ~ x_1 + x_2", data_df2, family="bernoulli")
M4 = model_logistic.fit()

# ---- Bayesian Poisson Regression ----
# Simulating Poisson-distributed outcome data
x_1 = np.random.normal(5, 2, 100)
x_2 = np.random.normal(10, 3, 100)
lambda_ = np.exp(0.5 * x_1 + 0.3 * x_2)
y = np.random.poisson(lambda_)

data_df3 = pd.DataFrame({"y": y, "x_1": x_1, "x_2": x_2})
model_poisson = bmb.Model("y ~ x_1 + x_2", data_df3, family="poisson")
M5 = model_poisson.fit()

# ---- Multilevel Models ----

# Simulating data for multilevel model
n_subjects, n_obs_per_subject = 10, 10
x_2 = np.repeat(np.arange(n_subjects), n_obs_per_subject)
x_1 = np.tile(np.arange(n_obs_per_subject), n_subjects)

random_intercepts = np.random.normal(0, 5, n_subjects)
random_slopes = np.random.normal(0.5, 0.2, n_subjects)
y = 15 + 2 * x_1 + random_intercepts[x_2] + random_slopes[x_2] * x_1 + np.random.normal(0, 3, n_subjects * n_obs_per_subject)
data_df4 = pd.DataFrame({"y": y, "x_1": x_1, "x_2": x_2})

# Fitting a multilevel model with varying intercepts and slopes
model_multilevel = bmb.Model("y ~ x_1 + (1 + x_1 | x_2)", data_df4)
M10 = model_multilevel.fit()

# ---- Model Comparison ----
# Comparing model fit using WAIC or LOO
waic_M1 = az.waic(M1)
waic_M10 = az.waic(M10)
model_comparison = az.compare({"M1": M1, "M10": M10})

# ---- Bayes Factor ----
# Using bayes_factor for model comparison
# Note: Currently, Bayes factors can be approximated but may not be natively available in `bambi` or `pymc3`
