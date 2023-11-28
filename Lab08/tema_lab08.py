import pandas as pd

df = pd.read_csv('/Users/adrian/Facultate/An3/Sem_1/PMP-2023/Lab08/Prices.csv')

import pymc3 as pm
import numpy as np

price = df['Price'].values
speed = df['Speed'].values
df['HardDrive'] = df['HardDrive']
hard_drive = np.log(df['HardDrive'].values)

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=10)
    beta2 = pm.Normal('beta2', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)

    mu = alpha + beta1 * speed + beta2 * hard_drive
    y = pm.Normal('y', mu=mu, sd=sigma, observed=price)

    trace = pm.sample(2000, tune=1000, cores=1)

pm.summary(trace, hdi_prob=0.95)[['hdi_2.5%', 'hdi_97.5%']]