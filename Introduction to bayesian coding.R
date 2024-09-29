#This R script provides an introduction to various bayesian models,
# code to change default brms functions, validation code, and model comparison


library(devtools)
library(rstan)
library(brms) #the important one
library(tidyverse)
library(lme4)

#bayesian linear regression using stan via BRMS (MCMC methods)----

#simulating some data

# Example dataframe
data_df1 <- data.frame(
  y = c(2.3, 1.9, 3.7, 2.5, 4.1, 5.2),
  x_1 = c(1.1, 0.8, 1.4, 1.0, 2.0, 1.5),
  x_2 = c(0.7, 1.2, 0.5, 1.1, 1.3, 0.9)
)

#running a simplistic model
M1 <- brm(y ~ x_1 + x_2, data = data_df1)
summary(M1)
# family: gaussian - means outcome variable is assumed to come from a normal dist
# a link function is used to connect the linear predictor (which is a linear,
#combination of the predictors or covariates) to the mean of the response variable,
# Bayesian models.
# the link function is sed for models where the response variable is continuous
# and follows a normal (Gaussian) distribution, such as regression
# Rhat - what is calculated from the 4 chains - roughly equivalent to a one way anova,
# showing the variance of samples within each chain and the variation between chains
# a Rhat of 1 is perfect - want values as close to 1 as possible
# sampling(NUTS) is hamiliton monte carlo sampler


#visualising bayesian linear models----
plot(M1)

#plotting the posterior distributions----
mcmc_plot(M1)
mcmc_plot(M1, type = 'hist')
mcmc_plot(M1, type = 'hist', binwidth = 0.05)
mcmc_plot(M1, type = 'areas') # shows interquartile range
mcmc_plot(M1, type = 'areas_ridges')


#looking at prior defaults within the previous examples----
prior_summary(M1) 
#or
get_prior(y ~ x_1 + x_2, data = data_df1)
#class 'b' are coefficient of fixed effects
#where y_i is normally distributed with a linear funtion of mu = B_0 + B_1(x_1i) + B_2(x_2i)
#prior = flat therefore is a uniform prior
#intercept and sigma are t distributions


#changing defaults----
#(telling what the MCMC what to do)
#iter to change the number of iterations in a MCMC
#warmup to say how many iterations are discarded at first in an MCMC.
#chains changes the number of chains in an MCMC
#prior = set_prior - gives a prior distribution, in this case x ~ N(mu, sigma)
M2 <- brm(y ~ x_1 + x_2, data = data_df1,
                iter = 2500,
                warmup = 500,
                chains = 4,
                prior = set_prior('normal(0, 100)'))
summary(M2)

#seeing coefficients----
#an alternative to the summary() function, and useful in different contexts
fixef(M1)
fixef(M2)

#Bayesian R^2----
#finding out how well the residuals fit the model
bayes_R2(M1)
bayes_R2(M2)


#visualising posterior vs prior plots----

data_df <- tibble(x = rnorm(10))

A <- brm(x ~ 1, data = data_df)
mcmc_plot(A)

B <- brm(x ~ 1, data = data_df, sample_prior = 'only')
mcmc_plot(B)

# put prior and posterior samples together
samples <- bind_rows(
  as_draws_matrix(B) %>% 
    as_tibble() %>% 
    select(-lp__) %>% 
    mutate(type = 'prior'),
  as_draws_matrix(A) %>% 
    as_tibble() %>% 
    select(-lp__) %>% 
    mutate(type = 'posterior')
) %>% pivot_longer(cols = -type, 
                   names_to = 'parameter', 
                   values_to = 'value') %>% 
  mutate(value = as.numeric(value))

ggplot(samples,
       aes(x = value, fill = type, colour = type)
) + geom_density(position = 'identity', alpha = 0.5) +
  facet_wrap(~parameter, scales = 'free') + 
  theme_minimal()


#changing priors for coefficients----
# where the prior is a distribution,
# in this case x ~ N(mu, sigma)
new_prior <- c(set_prior('normal(0, 10)', class = 'b', coef = 'x_1'),
               set_prior('normal(0, 10)', class = 'b', coef = 'x_2'),
               set_prior('student_t(1, 0, 30)', class = 'sigma'))

#changing priors in a model
M3 <- brm(y ~ x_1 + x_2,
          prior = new_prior,
          data = data_df1)

summary(M3)


#bayesian logistic regression----

# Example dataframe for a binary outcome (Bernoulli distribution)

# Simulate example data
data_df2 <- data.frame(
  y = rbinom(100, 1, 0.5),    # Binary outcome (0 or 1)
  x_1 = rnorm(100, mean = 170, sd = 10),  # Predictor 1: Simulated 'height' (in cm)
  x_2 = rnorm(100, mean = 25, sd = 5)     # Predictor 2: Simulated 'age' (in years)
)


M4 <- brm(y ~ x_1 + x_2, data = data_df2, family = bernoulli())

#bayesian poisson regression----

# Simulating data
x_1 <- rnorm(100, mean = 5, sd = 2)  # Continuous predictor 1
x_2 <- rnorm(100, mean = 10, sd = 3) # Continuous predictor 2

# Generate the response variable (y) as a Poisson-distributed count variable
# The linear predictor (log(lambda)) is based on x_1 and x_2
lambda <- exp(0.5 * x_1 + 0.3 * x_2)  # Poisson rate (lambda) based on predictors
y <- rpois(100, lambda)  # Response variable from a Poisson distribution

# Create the dataframe
data_df3 <- data.frame(y = y, x_1 = x_1, x_2 = x_2)

#run the model
M5 <- brm(y ~ x_1 + x_2, data = data_df3, family = poisson())

#zero-inflated poisson regression----
M6 <- brm(y ~ x_1 + x_2, data = data_df3, family = zero_inflated_poisson())

#negative binomial regression----
M7 <- brm(y ~ x_1 + x_2, data = data_df3, family = negbinomial())

#zero inflated negative binomial regression----
M8 <- brm(y ~ x_1 + x_2, data = data_df3, family = zero_inflated_negbinomial())


#multilevel models----

# Simulating dat
#for 10 subjects
n_subjects <- 10
n_obs_per_subject <- 10

# Simulate grouping factor (Subject)
x_2 <- factor(rep(1:n_subjects, each = n_obs_per_subject))

# Simulate predictor variable (x_1, similar to "Days")
x_1 <- rep(0:9, times = n_subjects)

# Random effects: Random intercepts and slopes for each subject
random_intercepts <- rnorm(n_subjects, mean = 0, sd = 5)
random_slopes <- rnorm(n_subjects, mean = 0.5, sd = 0.2)

# Create the response variable y based on fixed and random effects
y <- 15 + 2 * x_1 + random_intercepts[x_2] + random_slopes[x_2] * x_1 + rnorm(n_subjects * n_obs_per_subject, sd = 3)

# Create the dataframe
data_df4 <- data.frame(y = y, x_1 = x_1, x_2 = x_2)

#a normal multilevel model
M9 <- lmer(y ~ x_1 + (x_1 | x_2), data = data_df4)

#bayesian multilevel model
M10 <- brm(y ~ x_1 + (x_1 | x_2), data = data_df4)

#random intercept only models
M11 <- brm(y ~ x_1 + (1 | x_2), data = data_df4,
           save_pars = save_pars(all = TRUE)) #ensures that bayes_factor works

#random slopes only models
M12 <- brm(y ~ x_1 + (0+ x_1 | x_2), data = data_df4)

#random slopes and random intercept, with 0 correlation
M13 <- brm(y ~ x_1 + (x_1 | x_2), data = data_df4)

#all these model's defaults can be changed, for example:

MLM_priors <- c(
  prior(normal(10, 5), class = "Intercept"),        # Prior for the intercept
  prior(normal(2, 1), class = "b", coef = "x_1"),   # Prior for the fixed effect of x_1 (slope)
  prior(cauchy(0, 2), class = "sd", group = "x_2"), # Prior for the random intercept sd for x_2
  prior(cauchy(0, 2), class = "sd", group = "x_2", coef = "x_1"), # Prior for the random slope sd for x_1 in x_2
  prior(exponential(1), class = "sigma")            # Prior for the residual error (sigma)
)

M14 <- brm(y ~ x_1 + (x_1 | x_2), data = data_df4,
           iter = 2000,
           chain = 2,
           prior = MLM_priors,
           save_pars = save_pars(all = TRUE)) #ensures that bayes_factor works


#model comparison----
#comparing these models where elpd diff and se diff = 0 is the best model
loo(M10, M11, M12, M13, M14)

#Bayesfactor----
# used to compare the evidence in favour of a model
bayes_factor(M11, M14)
