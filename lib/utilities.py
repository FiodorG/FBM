import numpy as np
import mpmath
import math
import scipy
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as ss

np.random.seed(10000)


###############################################################################
########################## Survival Probabilities #############################
###############################################################################

def R(z, H):
    return R0Plus(z) * math.exp((H - 0.5) * W(z))


def R0Plus(z):
    return z * math.exp(- z * z / 2.)


def W(z):

    return \
        (
            z**4 * float(mpmath.hyp2f2(1., 1., 5. / 2., 3., z * z / 2.)) / 6.
          + math.pi * (1. - z * z) * scipy.special.erfi(z / math.sqrt(2.))
          + math.sqrt(2. * math.pi) * math.exp(z * z / 2.) * z
          + (z * z - 2.) * (math.log(2. * z * z) + np.euler_gamma)
          - 3. * z * z
        )


###############################################################################
############################### First Hitting Time ############################
###############################################################################

def G(m, t, mu, nu, H, N):
    tau = 1. / N
    return \
            math.exp(-N_func(m, t, mu, nu, H, tau)) \
          * G0(m, t) \
          * math.exp(2. * (H - 0.5) * (G1(m, t) - mu * G2Mu(m, t, tau) - nu * G2Nu(m, t, tau)) / G0(m, t))


def G0(m, t):
#    return m / (2. * np.sqrt(math.pi) * t**1.5) * math.exp(-m * m / 4. / t)
    return m / (np.sqrt(2. * math.pi) * t**1.5) * math.exp(-(m + 0.5 * t)**2 / 2. / t)


def G1(m, t):
    z = m / math.sqrt(2. * t)

    return \
        0.5 * G0(m, t) * \
        (
            I_func(z)
          + (z * z - 1.) * math.log(t)
          + z * z * (math.log(2. * z * z) + np.euler_gamma)
          - 4. * math.log(z)
          - 4. * np.euler_gamma
        )

#
#def G(z, H):
#    return G0(z) * math.exp((H - 0.5) * G1(z))
#
#
#def G0(z):
#    return z / np.sqrt(2 * math.pi) * math.exp(- z / 2)
#
#
#def G1(z):
#    return \
#        (
#            z**4 * float(mpmath.hyp2f2(1., 1., 5. / 2., 3., z * z / 2.)) / 6.
#          + math.pi * (1. - z * z) * scipy.special.erfi(z / math.sqrt(2.))
#          + math.sqrt(2. * math.pi) * math.exp(z * z / 2.) * z
#          + (z * z - 2.) * (math.log(2. * z * z) + np.euler_gamma)
#          - 3. * z * z
#        )


def G2Alpha(m, t, tau):
    z = m / math.sqrt(2. * t)

    return \
            math.exp(-z * z / 2.) * z * z * (I_func(z) - 2.) / (2. * math.sqrt(math.pi * t) * (1. - z * z)) \
          + z * scipy.special.erfc(z / math.sqrt(2.)) / (math.sqrt(2. * t) * (z * z - 1.)) \
          - math.exp(-z * z / 2.) * z * z * (math.log(2. * t * z * z / tau) + np.euler_gamma - 1.) / (2. * math.sqrt(math.pi * t))


def G2Beta(m, t, tau):
    z = m / math.sqrt(2. * t)

    return \
            math.exp(-z * z / 2.) * (I_func(z) - 2.) / (2. * math.sqrt(math.pi * t) * (1. - z * z)) \
          + z * scipy.special.erfc(z / math.sqrt(2.)) / (math.sqrt(2. * t) * (z * z - 1.)) \
          + math.exp(-z * z / 2.) * z * z * (1. - math.log(t / tau)) / (2. * math.sqrt(math.pi * t))


def G2Mu(m, t, tau):
    return 0.5 * (G2Alpha(m, t, tau) + G2Beta(m, t, tau))


def G2Nu(m, t, tau):
    return 0.5 * (G2Beta(m, t, tau) - G2Alpha(m, t, tau))

#    z = m / math.sqrt(2. * t)
#
#    return \
#            math.exp(-z * z / 2.) * (I_func(z) - 2.) / (4. * math.sqrt(math.pi * t)) \
#          + math.exp(-z * z / 2.) * z * z * (math.log(2. * z * z) + np.euler_gamma) / (4. * math.sqrt(math.pi * t))


def GHistory(m, t, mu, nu, H, N, window, x_prime):
    tau = 1. / N
    return \
            G(m, t, mu, nu, H, N) \
          + math.exp(- 0.5 * (H - 0.5) * (mu + nu) * integrate.quad(lambda s: math.log(1 - t / s) * x_prime(s), -window, 0)) \
          * math.exp((H - 0.5) * deltaGh(m, t, H, tau) / G0(m, t))


def deltaGh(m, t, h, tau):
    z = m / math.sqrt(2. * t)
    Ksi = h / t

    return \
            math.exp(-z * z / 2.) * z * z / (2. * math.sqrt(math.pi) * t**(5. / 2.) * Ksi * (Ksi + 1.)) \
          + z * math.exp(z * z / 2. / Ksi) * scipy.special.erfc(z * math.sqrt(Ksi + 1.) / math.sqrt(2. * Ksi)) \
          / (2. * math.sqrt(2.) * t**(5. / 2.) * Ksi**(3. / 2.) * (Ksi + 1.)**(3. / 2.))


def I_func(x):
    return \
            x**4 * float(mpmath.hyp2f2(1., 1., 5. / 2., 3., x * x / 2.)) / 6. \
          + math.pi * (1 - x * x) * scipy.special.erfi(x / math.sqrt(2.)) \
          - 3. * x * x \
          + math.sqrt(2. * math.pi) * math.exp(x * x / 2.) * x \
          + 2.


def N_func(m, t, mu, nu, H, tau):
    epsilon = H - 0.5
    D = 2. * H * tau**(2. * H - 1.)

    return m * 0.5 * (mu / D + nu) + \
           t * 0.25 * (mu * mu * t**(-2. * epsilon) + nu * nu * t**(2. * epsilon))


###############################################################################
################################## Pricing ####################################
###############################################################################

def OneTouchPricingBS(X0, B, T, sigma, r):

    if B < X0:
        return 1.0

    alpha = r - 0.5 * sigma * sigma
    beta = math.sqrt(alpha * alpha + 2. * sigma * sigma * r)

    d1 = (math.log(X0 / B) - beta * T) / (sigma * math.sqrt(T))
    d2 = (math.log(X0 / B) + beta * T) / (sigma * math.sqrt(T))

    A = (B / X0)**((alpha + beta) / sigma**2) * ss.norm.cdf(d1)
    B = (B / X0)**((alpha - beta) / sigma**2) * ss.norm.cdf(d2)

    return A + B


#    d1 = (math.log(X0 / B) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
#    d2 = (math.log(X0 / B) - (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
#
#    return (X0 / B) * ss.norm.cdf(d1) + (B / X0)**(2. * r / sigma**2) * ss.norm.cdf(d2)


###############################################################################
################################## Simulation #################################
###############################################################################

def eigenvalues(N, H):
    covariances = np.zeros(2 * N)
    for i in range(0, N):
        covariances[i] = gamma(i, H)
    for i in range(N + 1, 2 * N):
        covariances[i] = gamma(2 * N - i, H)

    return np.sqrt(np.real(np.fft.ifft(covariances)))


def fbm_path(X0, N, H, eigenvalues, drift):
    """
    Technically using different seeds do not guarantee
    that numbers are going to be independant between
    processes.
    """
    np.random.seed()
    V = np.random.normal(loc=0, scale=1, size=2 * N)
    W = np.zeros(2 * N, complex)

    W[0] = V[0]
    W[N] = V[N]

    W[1:N] = (V[1:N] + V[N + 1:2 * N] * 1j) / np.sqrt(2.)
    W2 = (V[1:N] - V[N + 1:2 * N] * 1j) / np.sqrt(2.)
    W[N + 1:2 * N] = W2[::-1]

    Z = np.real(np.fft.fft(eigenvalues * W))
    return (X0 + np.cumsum(Z[0:N])) / N**H + drift
# Survival
#    return (X0 + np.cumsum(Z[0:N])) / math.sqrt(2.) / N**H + drift


def filter_out(array, value):
    return array[array != value]


def index_of_closest(array, value):
    return min(range(len(array)), key=lambda i: abs(array[i] - value))


def gamma(k, H):
    return (abs(k - 1)**(2. * H) + abs(k + 1)**(2. * H) - 2. * abs(k)**(2. * H))


def drift(mu, nu, N, H):
    return [mu * t + nu * (t**(2. * H)) for t in np.linspace(1. / N, 1., num=N)]


def empirical_cdf(array, t):
    return len(array[array < t]) / len(array)


def integration_constant(x, mu, nu, H, N, m):
    """
    Integration constant to make G a density.
    Computed numerically on some defined interval
    """
    return integrate.quad(lambda t: G(x, t, mu, nu, H, N), 0.01, np.inf)[0]


###############################################################################
#################################### Plotting #################################
###############################################################################

def plot_pdf(x, mu, nu, H, N, m, results):
    """
    Plotting PDF of H = inf_{s}{ s | Z_s == x},
    i.e. P(H == t) for t in (0, 1).
    """
    constant = integration_constant(x, mu, nu, H, N, m)
    t_values = np.linspace(0.01, 0.99, num=50)

    values_analytical = [G(x, t, mu, nu, H, N) / constant for t in t_values]
    values_simulated = results[:, index_of_closest(m, x)] / float(N)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.hist(values_simulated, bins=80, range=(0, 1), density=True)
    ax.plot(t_values, values_analytical)
    ax.set_xlim([0, 0.98])
    ax.set_ylim([0, max(values_analytical) + 0.2])
    plt.ylabel('PDF')
    plt.xlabel('Time')
    plt.show()


def plot_cdf(x, mu, nu, H, N, m, results):
    """
    Plotting CDF of H = inf_{s}{ s | Z_s == x},
    i.e. P(H == t) for t in (0, 1).
    """
    constant = integration_constant(x, mu, nu, H, N, m)
    t_values = np.linspace(0.01, 0.99, num=50)

    values_simulated = results[:, index_of_closest(m, x)] / float(N)

    plt.figure()
    plt.plot(t_values, [empirical_cdf(values_simulated, t) for t in t_values], 'o')
    plt.plot(t_values, [integrate.quad(lambda t: G(x, t, mu, nu, H, N), 0.01, T)[0] / constant for T in t_values])
    plt.show()


def plot_G0(x, mu, nu, H, N, m, results):
    """
    H = inf_{s}{ s | Z_s == x}, i.e. we are interested in
    P(H == t) for t in (0, 1).
    Checks G0, historical vs simulated.
    Assumes mu=nu=0, H=0.5
    """
    constant = integration_constant(x, mu, nu, H, N, m)

    values_simulated = results[:, index_of_closest(m, x)] / float(N)
    hist = np.histogram(values_simulated, bins=80, range=(np.min(values_simulated), 1), density=True)

    G_values = hist[0][:-1]
    G_bins = (hist[1][1:-1] + hist[1][:-2]) * 0.5

    G0_simulated = G_values
    G0_analytical = [G0(x, t) / constant for t in G_bins]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(G_bins, G0_simulated, c='r', marker='+')
    ax.plot(G_bins, G0_analytical)
    plt.xlabel('Time')
    plt.ylabel('G0')
    ax.legend(['analytic', 'empirical'])
    ax.set_title('G0(simu) vs G0')
    ax.grid(True)
    plt.show()


def plot_W(H, N, results):
    """
    """
    hist = np.histogram(results, bins=40, range=(np.min(results), 2.5), density=True)

    R_values = hist[0][:-1]
    R_bins = (hist[1][1:-1] + hist[1][:-2]) * 0.5

    R_simulated = [math.log(r / R0Plus(z)) / (H - 0.5) for r, z in zip(R_values, R_bins)]
    R_analytical = [W(z) for z in R_bins]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(R_bins, R_simulated, c='r', marker='+')
    ax.plot(R_bins, R_analytical)
    plt.xlabel('y')
    plt.ylabel('W')
    ax.legend(['analytic', 'empirical'])
    ax.grid(True)
    plt.show()


def plot_G1(x, mu, nu, H, N, m, results):
    """
    H = inf_{s}{ s | Z_s == x}, i.e. we are interested in
    P(H == t) for t in (0, 1).
    Checks G1, historical vs simulated, knowing G0.
    Assumes mu=nu=0.
    """
    constant = integration_constant(x, mu, nu, H, N, m)

    values_simulated = results[:, index_of_closest(m, x)] / float(N)
    hist = np.histogram(values_simulated, bins=50, range=(np.min(values_simulated), 1), density=True)

    G_values = hist[0][:-1]
    G_bins = (hist[1][1:-1] + hist[1][:-2]) * 0.5

    G1_simulated = [math.log(y * constant / G0(x, t)) for y, t in zip(G_values, G_bins)]
    G1_analytical = [2. * (H - 0.5) * G1(x, t) / G0(x, t) for t in G_bins]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(G_bins, G1_simulated, c='r', marker='+')
    ax.plot(G_bins, G1_analytical)
    plt.xlabel('Time')
    plt.ylabel('log(G(simu)/G0)')
    ax.legend(['analytic', 'empirical'])
    ax.set_title('log(G(simu)/G0) vs 2*Eps*G1/G0')
    ax.grid(True)
    plt.show()


def plot_G1_y(mu, nu, H, N, m, results):
    """
    Assumes mu=nu=0.
    """

    bins = 50
    y_intervals = np.linspace(0.1, 5., num=bins)
    y_values = np.zeros(bins)

    for x in m:
        values_simulated = results[:, index_of_closest(m, x)] / float(N)
        hist = np.histogram(values_simulated, bins=bins, range=(0., 1.), density=True)

        t = (hist[1][1:-1] + hist[1][:-2]) * 0.5
        ys = x / 2. / np.power(t, H)
        values = hist[0][:-1] * t

        for i, y in enumerate(ys):
            idx = np.searchsorted(y_intervals, y)

            if idx != len(y_intervals):
                y_values[idx - 1] += values[i]

    y_values /= np.sum(y_values)

    constant = integrate.quad(lambda z: G(z, H), 0.00001, 8)[0]
    R_analytical = [G(z, H) / constant for z in y_intervals]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(y_intervals, y_values, c='r', marker='+')
    ax.plot(y_intervals, R_analytical)
    plt.xlabel('y')
    plt.ylabel('log(G(simu)/G0)')
    ax.legend(['analytic', 'empirical'])
    ax.grid(True)
    plt.show()


def plot_G2Mu(x, mu, nu, H, N, m, results):
    """
    H = inf_{s}{ s | Z_s == x}, i.e. we are interested in
    P(H == t) for t in (0, 1).
    Checks G2Mu, historical vs simulated, knowing G0, G1.
    Assumes nu=0.
    """
    constant = integration_constant(x, mu, nu, H, N, m)

    values_simulated = results[:, index_of_closest(m, x)] / float(N)
    hist = np.histogram(values_simulated, bins=40, range=(np.min(values_simulated), 1), density=True)

    G_values = hist[0][1:-1]
    G_bins = (hist[1][2:-1] + hist[1][1:-2]) * 0.5

    G2Mu_simulated = [
        - ((math.log(y * math.exp(N_func(x, t, mu, nu, H, 1. / N)) * constant / G0(x, t))) / 2. / (H - 0.5) - G1(x, t) / G0(x, t)) / mu
          for y, t in zip(G_values, G_bins) ]
    G2Mu_analytical = [G2Mu(x, t, 1. / N) / G0(x, t) for t in G_bins]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(G_bins, G2Mu_simulated, c='r', marker='+')
    ax.plot(G_bins, G2Mu_analytical)
    plt.xlabel('Time')
    plt.ylabel('G2Mu/G0')
    ax.legend(['analytic', 'empirical'])
    ax.set_title('-((log(G(simu)/G0)+N)/(2*Eps) - G1/G0)/mu vs G2Mu/G0')
    ax.grid(True)
    plt.show()


def plot_G2Nu(x, mu, nu, H, N, m, results):
    """
    H = inf_{s}{ s | Z_s == x}, i.e. we are interested in
    P(H == t) for t in (0, 1).
    Checks G2Nu, historical vs simulated, knowing G0, G1.
    Assumes mu=0.
    """
    constant = integration_constant(x, mu, nu, H, N, m)

    values_simulated = results[:, index_of_closest(m, x)] / float(N)
    hist = np.histogram(values_simulated, bins=40, range=(np.min(values_simulated), 1), density=True)

    G_values = hist[0][1:-1]
    G_bins = (hist[1][2:-1] + hist[1][1:-2]) * 0.5

    G2Mu_simulated = [
        - ((math.log(y * math.exp(N_func(x, t, mu, nu, H, 1. / N)) * constant / G0(x, t))) / 2. / (H - 0.5) - G1(x, t) / G0(x, t)) / nu
          for y, t in zip(G_values, G_bins) ]
    G2Mu_analytical = [G2Nu(x, t, 1. / N) / G0(x, t) for t in G_bins]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(G_bins, G2Mu_simulated, c='r', marker='+')
    ax.plot(G_bins, G2Mu_analytical)
    plt.xlabel('Time')
    plt.ylabel('G2Nu/G0')
    ax.legend(['analytic', 'empirical'])
    ax.set_title('-((log(G(simu)/G0)+N)/(2*Eps) - G1/G0 - muG2Mu/G0)/nu vs G2Nu/G0')
    ax.grid(True)
    plt.show()


def plot_OneTouchPrices(mu, nu, H, N, m, results):
    """
    Plotting One Touch price vs simulation.
    """

    prices_bs = [OneTouchPricingBS(1., np.exp(x), 1., 1., mu) for x in m]
#    prices_analytical = [integrate.quad(lambda t: math.exp(- mu * t) * G(x, t, mu, nu, H, N), 0.01, 1.0)[0] / integration_constant(x, mu, nu, H, N, m) for x in m]
    prices_analytical = [integrate.quad(lambda t: math.exp(- mu * t) * G0(x, t), 0.0, 1.0)[0] for x in m]
    prices_simulated = results.mean(axis=0)

    print('Prices BS:         %s' % np.array2string(np.array(prices_bs), precision=4, separator=', '))
    print('Prices Analytical: %s' % np.array2string(np.array(prices_analytical), precision=4, separator=', '))
    print('Prices Simulated:  %s' % np.array2string(prices_simulated, precision=4, separator=', '))
    print('Diffs:             %s' % np.array2string(np.array(prices_analytical) - prices_simulated, precision=4, separator=', '))

    fig, ax = plt.subplots()
    plt.bar(np.arange(len(m)), prices_bs, 0.35, alpha=0.8, color='r', label='BS')
    plt.bar(np.arange(len(m)) + 0.35, prices_analytical, 0.35, alpha=0.8, color='b', label='Analytical')
    plt.bar(np.arange(len(m)) + 0.7, prices_simulated, 0.35, alpha=0.8, color='g', label='Simulated')
    plt.xticks(np.arange(len(m)) + 0.35, ['%.1f' % x for x in m])
    plt.legend()
    plt.show()
