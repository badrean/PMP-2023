"""
1. Pentru fiecare model, identificaţi numărul de lanţuri, mărimea totală a eşantionului generat şi vizual-
izaţi distribuţia a posteriori. (0.5p)

2. Folosiţi ArviZ pentru a compara cele două modele, după criteriile R^ (Rhat) şi autocorelaţie. Concentraţi-
vă pe parametrii mu şi tau . (1p)

3. Număraţi numărul de divergenţe din fiecare model (cu sample_stats.diverging.sum() ), iar apoi
identificaţi unde acestea tind să se concentreze în spaţiul parametrilor (mu şi tau ). Puteţi folosi mo-
delul din curs, cu az.plot pair sau az.plot parallel . (0.5p)
#
"""

import arviz as az
import matplotlib.pyplot as plt

centered_eight_data = az.load_arviz_data("centered_eight")
non_centered_eight_data = az.load_arviz_data("non_centered_eight")

# 1.
num_chains_centered = centered_eight_data.posterior.chain.size
num_samples_centered = centered_eight_data.posterior.mu.size * num_chains_centered

num_chains_non_centered = non_centered_eight_data.posterior.chain.size
num_samples_non_centered = non_centered_eight_data.posterior.mu.size * num_chains_non_centered

# Vizualizarea
az.plot_posterior(centered_eight_data)
plt.savefig("centered_posterior_plot.png")
plt.close()
az.plot_posterior(non_centered_eight_data)
plt.savefig("non_centered_posterior_plot.png")
plt.close()

# 2.
rhat_centered = az.rhat(centered_eight_data, var_names=['mu', 'tau'])
autocorr_mu_centered = az.autocorr(centered_eight_data.posterior["mu"].values)
autocorr_tau_centered = az.autocorr(centered_eight_data.posterior["tau"].values)

rhat_non_centered = az.rhat(non_centered_eight_data, var_names=['mu', 'tau'])
autocorr_mu_non_centered = az.autocorr(non_centered_eight_data.posterior["mu"].values)
autocorr_tau_non_centered = az.autocorr(non_centered_eight_data.posterior["tau"].values)

print("Ex 2:")
print("Centered:")
print(f"Rhat mu: {rhat_centered['mu'].item()}")
print(f"Rhat tau: {rhat_centered['tau'].item()}")
print(f"Autocorrelation mu: {autocorr_mu_centered.mean().item()}")
print(f"Autocorrelation tau: {autocorr_tau_centered.mean().item()}\n")

print("Non centered:")
print(f"Rhat mu: {rhat_non_centered['mu'].item()}")
print(f"Rhat tau: {rhat_non_centered['tau'].item()}")
print(f"Autocorrelation mu: {autocorr_mu_non_centered.mean().item()}")
print(f"Autocorrelation tau: {autocorr_tau_non_centered.mean().item()}")

az.plot_autocorr(centered_eight_data, var_names=["mu", "tau"], combined=True)
plt.savefig("centered_autocorr_plot.png")
plt.close()
az.plot_autocorr(non_centered_eight_data, var_names=["mu", "tau"], combined=True)
plt.savefig("non_centered_autocorr_plot.png")
plt.close()

# 3.
divergences_centered = centered_eight_data.sample_stats["diverging"].sum()
divergences_non_centered = non_centered_eight_data.sample_stats["diverging"].sum()

az.plot_pair(centered_eight_data, var_names=["mu", "tau"], divergences=True)
plt.savefig("centered_diverg_plot.png")
plt.close()
az.plot_pair(non_centered_eight_data, var_names=["mu", "tau"], divergences=True)
plt.savefig("non_centered_diverg_plot.png")
plt.close()
