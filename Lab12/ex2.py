import numpy as np
import matplotlib.pyplot as plt

def calc_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum()*4/N
    error = abs((pi - np.pi) / pi) * 100
    return pi, error

N_vals = [100, 1000, 10000]

pi_vals = []
err_vals = []

for N in N_vals:
    pi, error = calc_pi(N)
    pi_vals.append(pi)
    err_vals.append(error)

mean_err = np.mean(err_vals)
standard_dev_err = np.std(err_vals)

plt.errorbar(N_vals, pi_vals, yerr=err_vals, fmt='o-', capsize=5)
plt.xscale('log')
plt.xlabel('Number of Points (N)')
plt.ylabel('Estimated π')
plt.title('Error Bars estimation for π')
plt.legend()
plt.show()

print(f'Mean Error: {mean_err:.3f}%')
print(f'Standard Deviation of Error: {standard_dev_err:.3f}%')