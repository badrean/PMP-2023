import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def posterior_grid(grid_points=50, heads=6, tails=9):

    grid = np.linspace(0, 1, grid_points)
    prior = (grid<= 0.5).astype(int)
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()

    return grid, posterior


def metropolis(func, draws=10000):
    trace = np.zeros(draws)
    old_x = 0.5
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace

params_beta = [(1, 1), (20, 20), (1, 4)]