import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

data = pd.read_csv('/Users/adrian/Facultate/An3/Sem_1/PMP-2023/Lab05/trafic.csv')
minute = data['minut']
nr_masini = data['nr. masini']

intervale = [(0, 7), (7, 8), (8, 16), (16, 19), (19, 24)]

with pm.Model() as model:
    # parametrii priori pt intervale
    lambda_values = []
    for interval in intervale:
        start, end = interval
        lambda_i = pm.Exponential(f'lambda_{start}_{end}', lam=1)  # Folosim 'lam' în loc de 'beta'
        lambda_values.append(lambda_i)

    # distributia poisson pt fiecare interval
    for i in range(len(intervale)):
        start, end = intervale[i]
        pm.Poisson(f'obs_{start}_{end}', mu=lambda_values[i], observed=nr_masini[(minute >= start) & (minute < end)])

    # Determinam lambda
    trace = pm.sample(2000, tune=1000, cores=1)
    lambda_means = []

    for start, end in intervale:
        variable_name = f'lambda_{start}_{end}'
        lambda_mean = np.mean(trace.posterior[variable_name].values)
        lambda_means.append(lambda_mean)

    print("Valorile lambda cele mai probabile:")
    for i in range(len(intervale)):
        start, end = intervale[i]
        print(f"Intervalul {start}-{end}: {lambda_means[i]}")

az.plot_trace(trace)