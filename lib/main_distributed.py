import time
import numpy as np
import multiprocessing as mp
import itertools
from utilities import eigenvalues, drift, fbm_path
from utilities import plot_pdf, plot_cdf
from utilities import plot_G0, plot_G1, plot_G2Mu, plot_G2Nu, plot_W, plot_G1_y
from utilities import plot_OneTouchPrices


def main_save_fbm_paths(npaths, X0, N, H, mu, nu, b):
    """
    Numerical simulation of a fractional Brownian Motion
    using Davies and Harte's method. Compute paths and saves
    them to the disk. This is a bad idea as it takes a lot
    ot memory.

    npaths: Number of MC paths
    N: lattice size
    H: Hurst coefficient
    mu: deterministic drift
    nu: fbm-specific drift
    m: values to find stopping times of
    """

    lambdas = eigenvalues(N, H)
    dr = drift(mu, nu, N, H)

    paths = np.zeros([npaths, N])

    for i in range(0, npaths):
        paths[i, :] = fbm_path(X0, N, H, lambdas, dr)

#    np.savetxt("fbm_paths.csv", paths, delimiter=",")

    return paths


def main_one_touch_pricing(npaths, X0, N, H, mu, nu, b):
    """
    Numerical simulation of a fractional Brownian Motion
    using Davies and Harte's method. Computation of the price
    of a one touch option. Recall the price is
    E[exp{-r*tau}*I_{tau < T}] where tau is the first hitting
    time of the barrier b.

    npaths: Number of MC paths
    N: lattice size
    H: Hurst coefficient
    mu: deterministic drift (typically discount rate)
    nu: fbm-specific drift
    b: barriers
    """

    lambdas = eigenvalues(N, H)
    dr = drift(mu, nu, N, H)

    prices = np.zeros([npaths, len(b)])

    for i in range(0, npaths):
        X = fbm_path(X0, N, H, lambdas, dr)

        X_max = np.maximum(np.maximum.accumulate(X), 0)
        hitting_times = np.searchsorted(X_max, b)
        prices[i, :] = (hitting_times < len(X_max)) * np.exp(- mu * (hitting_times + 1))

    return prices


def main_first_hitting_time(npaths, X0, N, H, mu, nu, m):
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
        X = fbm_path(X0, N, H, lambdas, dr)

        X_max = np.maximum(np.maximum.accumulate(X), 0)
        stopping_times[i, :] = np.searchsorted(X_max, m)

    return stopping_times


def main_survival(npaths, X0, N, H, mu, nu, m):
    """
    Numerical simulation of a fractional Brownian Motion
    using Davies and Harte's method. Computation of the
    survival probability (proba to be somewhere knowing
    0 hasn't been touched).
    npaths: Number of MC paths
    N: lattice size
    H: Hurst coefficient
    mu: deterministic drift
    nu: fbm-specific drift
    m: values to find stopping times of
    """

    lambdas = eigenvalues(N, H)
    dr = drift(mu, nu, N, H)

    values = np.zeros([npaths])

    for i in range(0, npaths):
        X = fbm_path(X0, N, H, lambdas, dr)

        X_min = np.min(X)

        if X_min > 0:
            values[i] = X[-1]
        else:
            values[i] = np.nan

    return values[~np.isnan(values)]


def main(func, npaths, X0, N, H, mu, nu, m):

    njobs = mp.cpu_count()
    args = itertools.repeat([int(npaths/njobs), X0, N, H, mu, nu, m], njobs)

    with mp.Pool(processes=njobs) as pool:
        results = pool.starmap(func, args)

    return np.concatenate(results)


if __name__ == '__main__':

    """
    For a diffusion of the form z_t = x_t + mu*t + nu*t**(2H)
    """

    npaths = 100000    # Numbers of MC paths
    N = 2**13          # Size of the lattice
    H = 0.50           # Hurst exponent

    mu = 0.0           # Drift
    nu = 0.0           # Autocorrelation drift

    X0 = 0             # starting point of diffusion

    m = np.linspace(0.1, 1.0, 10)      # stopping values to store
    func = main_one_touch_pricing

    start = time.time()
    results = main(func, npaths, X0, N, H, mu, nu, m)
    end = time.time()

    print('Run time: %.2f' % (end - start))

    if False:
        x = 0.75

        # First Passage Time
        plot_pdf(x, mu, nu, H, N, m, results)
        plot_cdf(x, mu, nu, H, N, m, results)

        plot_G0(x, mu, nu, H, N, m, results)
        plot_G1(x, mu, nu, H, N, m, results)
        plot_G2Mu(x, mu, nu, H, N, m, results)
        plot_G2Nu(x, mu, nu, H, N, m, results)

        plot_G1_y(mu, nu, H, N, m, results)

        # Survival
        plot_W(H, N, results)

        # One Touch
        plot_OneTouchPrices(mu, nu, H, N, m, results)
