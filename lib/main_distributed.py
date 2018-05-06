import time
import numpy as np
import multiprocessing as mp
import itertools
from utilities import eigenvalues, drift, fbm_path
from utilities import plot_pdf, plot_cdf, plot_G0, plot_G1, plot_G2Mu, plot_G2Nu


def func(npaths, N, H, mu, nu, m):
    """
    Numerical simulation of a fractional Brownian Motion
    using Davies and Harte's method. Computation of the
    first stopping time at different (positive) values.
    npaths: Number of MC paths
    N: lattice size
    H: Hurst coefficient
    mu: deterministic drift
    nu: fbm-specific drift
    m: values to find stopping times of
    """

    lambdas = eigenvalues(N, H)
    dr = drift(mu, nu, N, H)

    stopping_times = np.zeros([npaths, len(m)])

    for i in range(0, npaths):
        X = fbm_path(N, H, lambdas, dr)

        X_max = np.maximum(np.maximum.accumulate(X), 0)  # starting point missing
        stopping_times[i, :] = np.searchsorted(X_max, m)

    return stopping_times


def main(npaths, N, H, mu, nu, m, njobs = -1):

    if njobs == -1:
        njobs = mp.cpu_count()

    args = itertools.repeat([int(npaths/njobs), N, H, mu, nu, m], njobs)

    with mp.Pool(processes=njobs) as pool:
        results = pool.starmap(func, args)

    return np.concatenate(results)


if __name__ == '__main__':

    """
    For a diffusion of the form z_t = x_t + mu*t + nu*t**(2H)
    """

    npaths = 1000000   # Numbers of MC paths
    N = 2**13          # Size of the lattice
    H = 0.55           # Hurst exponent

    mu = 0.0           # Drift
    nu = 0.1           # Autocorrelation drift

    m = np.linspace(0.1, 2, num=20)  # stopping values to store

    start = time.time()
    results = main(npaths, N, H, mu, nu, m)
    end = time.time()

    print('Run time: %.2f' % (end - start))

    if False:
        x = 0.8

        plot_pdf(x, mu, nu, H, N, m, results)
        plot_cdf(x, mu, nu, H, N, m, results)

        plot_G0(x, mu, nu, H, N, m, results)
        plot_G1(x, mu, nu, H, N, m, results)
        plot_G2Mu(x, mu, nu, H, N, m, results)
        plot_G2Nu(x, mu, nu, H, N, m, results)
