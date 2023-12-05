import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

data = pd.read_csv("/Users/adrian/Facultate/An3/Sem_1/PMP-2023/Lab09/Admission.csv")

gre = data['GRE'].values
gpa = data['GPA'].values
admission = data['Admission'].values

df = data.query("Admission == (0, 1)")
print(df.head())
y_1 = pd.Categorical(df['Admission']).codes
x_n = ['GRE', 'GPA']
x_1 = df[x_n].values

#1
with pm.Model() as model:
    a = pm.Normal('a', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=2, shape=len(x_n))
    Mu = a + pm.math.dot(x_1, beta)
    tetha = pm.Deterministic('tetha', 1 / (1 + pm.math.exp(- Mu)))
    bd = pm.Deterministic('bd', -a/beta[1] - beta[0]/beta[1] * x_1[:,0])
    yl = pm.Bernoulli('yl', p=tetha, observed=y_1)
    trace = pm.sample(2000, return_inferencedata=True, cores=1)

#2  
idx = np.argsort(x_1[:, 0])
bd = trace.posterior['bd'].mean(("chain", "draw"))[idx]

plt.scatter(x_1[:, 0], x_1[:, 1], c=[f'C{x}' for x in y_1]) 
plt.plot(x_1[:, 0][idx], bd, color='k')
az.plot_hdi(x_1[:, 0], trace.posterior['bd'], color='k', hdi_prob=0.94)
plt.xlabel(x_n[0])
plt.ylabel(x_n[1])
plt.legend()
plt.show()

#3
date_student1 = np.array([[1, 550, 3.5]])

prob_student1_post = 1 / (1 + np.exp(-trace.posterior['a'].values - np.dot(date_student1[:, 1:], trace.posterior['beta'].values.T)))
hdi_prob = az.hdi(prob_student1_post, hdi_prob=0.9)
print(f"Intervalul HDI primul student: [{hdi_prob[0]:.4f}, {hdi_prob[1]:.4f}]")

#4
date_student2 = np.array([[1, 500, 3.2]])

prob_student2_post = 1 / (1 + np.exp(-trace.posterior['a'].values - np.dot(date_student2[:, 1:], trace.posterior['beta'].values.T)))
hdi_prob = az.hdi(prob_student2_post, hdi_prob=0.9)
print(f"Intervalul HDI al doilea student: [{hdi_prob[0]:.4f}, {hdi_prob[1]:.4f}]")

# Diferenta se justifica prin faptul ca GRE si GPA pot afecta in mod diferit probabilitatea de a fi admis sau nu.