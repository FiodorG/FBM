import math
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import scipy.integrate as integrate
from utilities import index_of_closest, eigenvalues, drift, fbm_noise, G, G0, G1, G2Mu, N_cst


### PARAMETERS ###
# For a diffusion of the form z_t = x_t + mu*t + nu*t**(2H)

N = 2**12          # Size of the lattice
npaths = 10000     # Numbers of MC paths
H = 0.55           # Hurst Exponent
epsilon = H - 0.5  # Expansion of H around 0.5

mu = 0.1           # drift
nu = 0.0           # autocorrelation drift

stopping_values = np.linspace(0.1, 2, num=20)  # values of m to store

### MAIN ###
# Using Davies and Harte method

eigenvalues = eigenvalues(N, H)
drift = drift(mu, nu, N, H)

stopping_times = np.zeros([npaths, len(stopping_values)])


for i in range(0, npaths):
    Z = fbm_noise(N, eigenvalues)
    X = np.cumsum(Z[0:N]) / N**H + drift  # X is the fBm path defined on [0,1]
    # X[0] is the process at time 1/N, and X[N-1] is the process at time 1.

    X_max = np.maximum.accumulate(X)
    X_max = np.maximum(X_max, 0)  # X is missing the first data-point 0

    stopping_times[i, :] = np.searchsorted(X_max, stopping_values)


### DISPLAY ###
m = 0.5
m_index = index_of_closest(stopping_values, m)

constant = integrate.quad(lambda x: G(m, x, mu, nu, H, epsilon, N), 0.01, np.inf)[0]
values_analytical = [G(m, t, mu, nu, H, epsilon, N) / constant for t in np.linspace(0.01, 1, num=N)]
values_simulated = stopping_times[:, m_index] / float(N)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
hist = ax.hist(values_simulated, bins=80, range=(0, 1), density=True)
plot = ax.plot(np.linspace(0.01, 1, num=N), values_analytical)
ax.set_xlim([0, 0.98])
ax.set_ylim([0, max(values_analytical) + 0.2])
plt.ylabel('CDF')
plt.xlabel('Time')
plt.show()


### CHECKING ###
plt.plot(np.linspace(0.01, 1, num=N), [empirical_cdf(values_simulated, t) for t in np.linspace(0.01, 1, num=N)])
plt.plot(np.linspace(0.01, 1, num=N), [G(m, t, mu, nu, H, epsilon, N) / constant for t in np.linspace(0.01, 0.99, num=N)])

## check G1
check1 = [math.log(x * constant / G0(m, t)) for x, t in zip(hist[0][1:-1], hist[1][1:-1])]
check2 = [2. * epsilon * G1(m, t, 0.) / G0(m, t) for t in np.linspace(0.01, 1, num=N)]

## check G2alpha
check3 = [(math.log(x * constant / G0(m, t)) + N_cst(m, t, mu, nu, H, epsilon, 1. / N)) / 2. / epsilon - G1(m, t, 1. / N) / G0(m, t) \
          for x, t in zip(hist[0][1:-1], hist[1][1:-1])]
check4 = [-mu * G2Mu(m, t, 1. / N) / G0(m, t) for t in np.linspace(0.01, 1, num=N)]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
plot1 = ax.scatter(0.5 * (hist[1][2:-1] + hist[1][1:-2]), check1, c='r', marker='+')
plot2 = ax.plot(np.linspace(0.01, 1, num=N), check2)
plt.xlabel('Time')
plt.ylabel('log(G/G0)')
ax.legend(['analytic', 'empirical'])
ax.set_title('log(G/G0) vs 2*Eps*G1/G0 (%s paths)' % str(npaths))
ax.grid(True)
plt.show()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
plot1 = ax.scatter(0.5 * (hist[1][2:-1] + hist[1][1:-2]), check3, c='r', marker='+')
plot2 = ax.plot(np.linspace(0.01, 1, num=N), check4)
plt.xlabel('Time')
plt.ylabel('')
ax.legend(['analytic', 'empirical'])
ax.set_title('G2Mu checking' % str(npaths))
ax.grid(True)
plt.show()