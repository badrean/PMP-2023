import random
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import matplotlib.pyplot as plt
import networkx as nx
import pymc3 as pm
import numpy as np

# Subiect 1

# 1.
def simulate_game():
    p0_prob = 1/3
    p1_prob = 1/2

    p0_wins = 0
    p1_wins = 0

    for i in range(20000):
        # alegem cine incepe
        start_player = random.choice(['P0', 'P1'])

        if start_player == 'P0':
            # p0 arunca moneda o data
            if random.random() < p0_prob:
                n = 1
            else:
                n = 0

            # p1 arunca moneda de n+1 ori
            m = 0
            for i in range(n+1):
                if random.random() < p1_prob:
                    m += 1
        else:
            # p1 arunca moneda o data
            if random.random() < p1_prob:
                n = 1
            else:
                n = 0
            # p0 arunca moneda de n+1 ori
            m = 0
            for i in range(n+1):
                if random.random() < p0_prob:
                    m += 1

        if n >= m:
            p0_wins += 1
        else:
            p1_wins += 1

    return p0_wins / 20000, p1_wins / 20000

p0_win_rate, p1_win_rate = simulate_game()
print(f'P0: {p0_win_rate * 100}%')
print(f'P1:: {p1_win_rate * 100}%')

# 2.

# definim modelul
model = BayesianNetwork([('StartPlayer', 'N'), ('N', 'M'), ('StartPlayer', 'M')])

# definim prob conditionate
cpd_start_player = TabularCPD('StartPlayer', 2, [[0.5], [0.5]])
cpd_n = TabularCPD('N', 2, [[1/3, 1/2], [2/3, 1/2]], evidence=['StartPlayer'], evidence_card=[2])
cpd_m = TabularCPD('M', 3, [[1/9, 1/4, 1/2, 1/2], [2/3, 1/2, 1/2, 1/2], [2/9, 1/4, 0, 0]], 
                   evidence=['StartPlayer', 'N'], evidence_card=[2, 2])

# asociem prob conditionate cu modelul
model.add_cpds(cpd_start_player, cpd_n, cpd_m)

# verificam daca modelul e definit corect
assert model.check_model()

# afisam modelul (Comentat pt ca am facut push la imagine. Nume: NetworkSub1.jpg)
# pos = nx.circular_layout(model)
# nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
# plt.show()

# Subiect 2

# 1
# timpul mediu de asteptare
mu = 2
# deviatia standard
sigma = 6

# generam 200 de valori de timpi de asteptare
wait_times = np.random.normal(mu, sigma, 200)