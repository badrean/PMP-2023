#1

import numpy as np
from scipy.stats import poisson, norm, expon, gamma
import random

lambda_poisson = 20

mean_normal = 2
std_normal = 0.5

limita_inferioara_alpha = 3.0
lim_superioara_alpha = 10.0
alpha_expon = random.uniform(limita_inferioara_alpha, lim_superioara_alpha)

poisson_dist = poisson(mu=lambda_poisson)
normal_dist = norm(loc=mean_normal, scale=std_normal)
expon_dist = expon(scale=alpha_expon)

#2

timp_maxim = 0.25
probabilitate_dorita = 0.95
numar_clienti_pe_ora = lambda_poisson

toleranta = 0.1

while lim_superioara_alpha - limita_inferioara_alpha > toleranta:
    alpha_intermediar = (limita_inferioara_alpha + lim_superioara_alpha) / 2

    probabilitate_intermediara = gamma.cdf(timp_maxim, numar_clienti_pe_ora, scale=1 / alpha_intermediar)

    if probabilitate_intermediara > probabilitate_dorita:
        lim_superioara_alpha = alpha_intermediar
    else:
        limita_inferioara_alpha = alpha_intermediar

alpha = (limita_inferioara_alpha + lim_superioara_alpha) / 2

print(f"Valoare maxima a lui alpha pentru a servi toti clientii intr-un timp de 15 minute cu o probabilitate de 0.95 este aprox: {alpha:.2f} pe ora.")

#3

timpul_mediu_asteptare = (1 / alpha) / 2

print(f"Timpul este {timpul_mediu_asteptare:.2f} min.")
