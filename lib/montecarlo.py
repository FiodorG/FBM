import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

np.random.seed(seed=69)


def d1(S0, K, r, sigma, T) -> float:
    return (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))


def d2(S0, K, r, sigma, T) -> float:
    return (np.log(S0 / K) + (r - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))


def BlackScholes(type, S0, K, r, sigma, T):
    if type == "C":
        return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    elif type == "P":
        return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T))
    else:
        raise ValueError('option type unrecognized')


def callfunction(x,K):
    return np.maximum(x - K, 0)


def putfunction(x,K):
    return np.maximum(K - x, 0)


def optionMC(paths, i_mc_paths, K, type):
    last_values = np.array([paths[i][-1] for i in i_mc_paths])

    print('Mean: %5.3f' % np.average(last_values))
    print('Std:  %5.3f' % np.std(last_values))

    if type == "C":
        return np.average(callfunction(last_values, K))
    elif type == "P":
        return np.average(putfunction(last_values, K))
    else:
        raise ValueError('option type unrecognized')


def random_number_generator(N, times):
    return np.random.standard_normal(size=(N, times))


def gmb_path(S0, sigma, times, dt, i_mc_paths, Z):
    return [S0 + sigma * np.sqrt(dt) * np.cumsum(Z[i]) for i in i_mc_paths]


def exp_gmb_path(S0, sigma, times, dt, i_mc_paths, Z):
    return [S0 * np.exp((-0.5 * sigma**2) * times + sigma * np.sqrt(dt) * np.cumsum(Z[i])) for i in i_mc_paths]


def gamma(t, dt, H):
    return 0.5 * ((np.abs(t - dt))**(2 * H) + (np.abs(t + dt))**(2 * H) - 2 * (np.abs(t))**(2 * H))


def gamma_discrete(k, H):
    return 0.5 * ((np.abs(k - 1))**(2 * H) + (np.abs(k + 1))**(2 * H) - 2 * (np.abs(k))**(2 * H))


def gamma_matrix(times, dt, H):
    matrix = np.zeros((len(times), len(times)))
    range = np.arange(len(times))
    for i in range:
        for j in range:
            matrix[i][j] = gamma(times[i] - times[j], dt, H)

    return matrix


def gamma_matrix_discrete(times, H):
    matrix = np.zeros((len(times), len(times)))
    range = np.arange(len(times))
    for i in range:
        for j in range:
            matrix[i][j] = gamma_discrete(i - j, H)

    return matrix


def fractional_gmb_path(S0, sigma, times, dt, N, H, Z):
    var_cov_matrix = gamma_matrix(times, dt, H)
    std_dev_matrix = np.linalg.cholesky(var_cov_matrix)
    i_mc_paths = np.arange(N)
    return [S0 + sigma * np.cumsum(np.array(std_dev_matrix * np.transpose(np.sqrt(dt) * np.asmatrix(Z[i])))) for i in i_mc_paths]


def fractional_gmb_path_discrete(S0, sigma, times, dt, N, H, Z):
    var_cov_matrix = gamma_matrix_discrete(times, H)
    std_dev_matrix = np.linalg.cholesky(var_cov_matrix)
    i_mc_paths = np.arange(N)
    frac_noise = [sigma * std_dev_matrix * np.transpose(np.asmatrix(Z[i])) for i in i_mc_paths]
    return [S0 + dt**H * np.cumsum(np.array(frac_noise[i])) for i in i_mc_paths]


def fractional_exp_gmb_path(S0, sigma, times, dt, N, H, Z):
    var_cov_matrix = gamma_matrix(N, times, dt, H)
    std_dev_matrix = np.linalg.cholesky(var_cov_matrix)
    i_mc_paths = np.arange(N)
    return [S0 * np.exp(-0.5 * sigma**2 * np.power(times, 2 * H) + sigma * np.cumsum(np.array(std_dev_matrix * np.transpose(np.sqrt(dt) * np.asmatrix(Z[i]))))) for i in i_mc_paths]


def fractional_exp_gmb_path_discrete(S0, sigma, times, dt, N, H, Z):
    var_cov_matrix = gamma_matrix_discrete(times, H)
    std_dev_matrix = np.linalg.cholesky(var_cov_matrix)
    i_mc_paths = np.arange(N)
    frac_noise = [sigma * std_dev_matrix * np.transpose(np.asmatrix(Z[i])) for i in i_mc_paths]
    return [S0 * np.exp(-0.5 * sigma ** 2 * np.power(times, 2 * H) + dt**H * np.cumsum(np.array(frac_noise[i]))) for i in i_mc_paths]





S0 = 100
sigma = 0.20
N = 1000
T = 1
timesteps = 365 * T
H = 0.7


i_mc_paths = np.arange(N)
times = np.linspace(0, T, timesteps)
dt = times[1]
Z = random_number_generator(N, timesteps)

'''
paths = exp_gmb_path(S0, sigma, times, dt, i_mc_paths, Z)
for i in i_mc_paths:
    plt.plot(times, paths[i])

type = "C"
K = S0
BSclosed = BlackScholes(type, S0, K, 0, sigma, T)
BSMC = optionMC(paths, i_mc_paths, K, type)

print('BS Closed Form: %5.3f' % BSclosed)
print('BS Monte Carle: %5.3f' % BSMC)

axes = plt.gca()
axes.set_xlim([0, T])
#axes.set_ylim([S0 - 50, S0 + 50])
plt.legend(loc='best')
#plt.show()
'''

'''
for H in np.linspace(0.1, 0.9, 9):
    plt.plot(times, fractional_gmb_path(S0, sigma, times, dt, N, H, Z)[0], label=str(H))

axes = plt.gca()
axes.set_xlim([0, T])
plt.legend(loc='best')
plt.show()
'''

paths = fractional_exp_gmb_path_discrete(S0, sigma, times, dt, N, H, Z)
for i in i_mc_paths:
    plt.plot(times, paths[i])

last_values = np.array([paths[i][-1] for i in i_mc_paths])

print('Mean: %5.3f' % np.average(last_values))
print('Std:  %5.3f' % np.std(last_values))


axes = plt.gca()
axes.set_xlim([0, T])
#axes.set_ylim([S0 - 0.20, S0 + 0.20])
plt.legend(loc='best')
plt.show()