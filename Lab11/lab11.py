import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt

#1
clusters = 3
n_cluster = [200, 150, 150]
n_total = sum(n_cluster)
means = [5, 0, 2]
std_devs = [2, 2, 2]
mix = np.random.normal(np.repeat(means, n_cluster),np.repeat(std_devs, n_cluster))
mix=np.array(mix)
az.plot_kde(mix)

#2
clusters = [2,3,4]
models = []
idatas = []
for cluster in clusters:
    with pm.Model() as model:
        p = pm.Dirichlet('p', a=np.ones(cluster))
        means = pm.Normal('means', mu=np.linspace(mix.min(), mix.max(), cluster), sigma=10, shape=cluster, transform=pm.distributions.transforms.ordered)

        sd = pm.HalfNormal('sd', sigma=10)
        y = pm.NormalMixture('y', w=p, mu=means, sigma=sd, observed=mix)
        idata = pm.sample(1000, tune=2000, target_accept=0.9,random_seed=123, return_inferencedata=True)
        idatas.append(idata)
        models.append(model)

#3
ppc_mm = []
for i in range(3):
    ppc_mm.append(pm.sample_posterior_predictive(idatas[i], 1000, models[i]))

fig, ax = plt.subplots(1, 3, figsize=(9, 6), sharex=True, constrained_layout=True)
ax = np.ravel(ax)

T_obs = np.subtract(*np.percentile(mix, [75, 25], axis=0))
for idx, d_sim in enumerate(ppc_mm):
    T_sim = np.subtract(*np.percentile(d_sim.posterior_predictive['y'][:100].T, [75, 25], axis=1))
    p_value = np.mean(T_sim >= T_obs)
    az.plot_kde(T_sim, ax=ax[idx])
    ax[idx].axvline(T_obs, 0, 1, color='k', ls='--')
    ax[idx].set_title(f'K = {clusters[idx]} \n p-value {p_value:.2f}')
    ax[idx].set_yticks([])