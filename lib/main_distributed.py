import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import scipy.integrate as integrate
import itertools
from utilities import index_of_closest, eigenvalues, drift, fbm_path


def func(npaths, N, H, mu, nu, m):
    """
    Numerical simulation of a fractional Brownian Motion
    using Davies and Harte's method. Computation of the
    first stopping time at different (positive) values.
    npaths: Number of MC paths
    N: lattice size
    H: Hurst coefficient
    mu: deterministic drift
    nu: fbm-related drift
    m: values to find stopping times of
    """

    lambdas = eigenvalues(N, H)
    dr = drift(mu, nu, N, H)

    stopping_times = np.zeros([npaths, len(m)])

    for i in range(0, npaths):
        X = fbm_path(N, H, lambdas, dr)

        X_max = np.maximum.accumulate(X)
        stopping_times[i, :] = np.searchsorted(X_max, m)

    return stopping_times


def main():
    """
    For a diffusion of the form z_t = x_t + mu*t + nu*t**(2H)
    """

    npaths = 100000    # Numbers of MC paths
    N = 2**12          # Size of the lattice
    H = 0.55           # Hurst exponent

    mu = 0.1           # Drift
    nu = 0.0           # Autocorrelation drift

    m = np.linspace(0.1, 2, num=20)  # stopping values to store

    processes = mp.cpu_count()
    args = itertools.repeat([int(npaths/processes), N, H, mu, nu, m], processes)

    with mp.Pool(processes=processes) as pool:
        results = pool.starmap(func, args)

    return np.concatenate(results)


if __name__ == '__main__':
    start = time.time()
    results = main()
    end = time.time()

    print(end - start)
