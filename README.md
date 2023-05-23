# BayesianOptimisationNAS

A simple Bayesian Optimiser for finding optimal hyperparameters for a given function.
In the MNIST test, the optimiser is initialised with 15 random searches, following by optimisation for 15 steps. When comparing this to the 30 random searches, 9 of the 15 chosen arguments resulted in higher scoring models than the best found using random search.
