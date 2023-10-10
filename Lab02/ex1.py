import numpy as np
from scipy.stats import expon

import matplotlib.pyplot as plt

np.random.seed(1)

# Exercitiul 1

lambda_m1 = 4
lambda_m2 = 6

prob_m1 = 0.4

timp_servire_m1 = expon.rvs(scale=1/lambda_m1, size=int(10000 * (prob_m1)))
timp_servire_m2 = expon.rvs(scale=1/lambda_m2, size=int(10000 * (1 - prob_m1)))

timp_servire_clienti = np.concatenate([timp_servire_m1, timp_servire_m2]) # compunerea

media_x = np.mean(timp_servire_clienti)
deviatia_x = np.std(timp_servire_clienti)

plt.hist(timp_servire_clienti, bins=50, density=True, alpha=0.5, color='b', label='Histogram')
plt.xlabel('Timp Servire Clienti')
plt.ylabel('Densitate')
plt.axvline(media_x, color='r', linestyle='dashed', linewidth=2, label='Mean')
plt.legend()
plt.show()