import pymc as pm
from scipy.stats import geom
import numpy as np
import pandas as pd
import arviz as az

# Citirea datelor
data = pd.read_csv("C:\\Users\\Adi\\Desktop\\Facultate\\PMP-2023\\Examen\\Titanic.csv")

# Prelucrarea datelor: daca un pasager nu are Age, atunci completam cu media varstelor
data['Age'].fillna(data['Age'].mean(), inplace=True)

with pm.Model() as model:
    # Parametrii pentru coeficientii de regresie
    beta_age = pm.Normal('beta_age', mu=0, sd=10)
    beta_class = pm.Normal('beta_class', mu=0, sd=10)
    intercept = pm.Normal('intercept', mu=0, sd=10)

    # Logitul probabilitatii de supravietuire
    logit_p = intercept + beta_age * data['Age'] + beta_class * data['Pclass_2'] 

    # Modelul de regresie logistica
    survived = pm.Bernoulli('survived', logit_p=logit_p, observed=data['Survived'])
    trace = pm.sample(2000, tune=1000, cores=1)

age_to_predict = 30
class_to_predict = 2

# Calculul probabilitatii pentru pasager
logit_p_predict = trace['intercept'] + trace['beta_age'] * age_to_predict + trace['beta_class'] * class_to_predict
p_predict = 1 / (1 + np.exp(-logit_p_predict))

# Construirea intervalului HDI pentru probabilitate
hdi_90 = pm.hdi(p_predict, hdi_prob=0.9)

print(f"Intervalul de 90% HDI pentru probabilitatea de a supravietui a unui pasager de 30 de ani din clasa a 2a: {hdi_90}")

#2a
theta_X = 0.3
theta_Y = 0.5
evenimenteFav = 0

for _ in range(10000):
    # Simulam variabilele aleatoare repartizate geometric folosind geom.rvs(theta)
    x = geom.rvs(theta_X)
    y = geom.rvs(theta_Y)
    # Verificam conditia cerintei si numaram evenimentele care o indeplinesc
    if x > y**2:
        evenimenteFav += 1

aprox = evenimenteFav*4 / 10000 
print("Aproximarea pi:", aprox)
# Rezultat Aproximarea pi: 1.672 - variabil
