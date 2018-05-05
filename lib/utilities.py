import numpy as np
import mpmath
import math
import scipy

np.random.seed(10000)


def filter_out(array, value):
    return array[array != value]


def index_of_closest(array, value):
    return min(range(len(array)), key=lambda i: abs(array[i] - value))


def gamma(k, H):
    return (abs(k - 1)**(2. * H) + abs(k + 1)**(2. * H) - 2. * abs(k)**(2. * H))


def G(m, t, mu, nu, H, epsilon, N):
    tau = 1. / N
    return \
            math.exp(-N_func(m, t, mu, nu, H, epsilon, tau)) \
          * G0(m, t) \
          * math.exp(2. * epsilon * (G1(m, t, tau) - mu * G2Mu(m, t, tau) - nu * G2Nu(m, t, tau)) / G0(m, t))


def G0(m, t):
    return m / (2. * np.sqrt(math.pi) * t**1.5) * math.exp(- m * m / 4. / t)


def G1(m, t, tau):
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


def I_func(x):
    return \
            x**4 * float(mpmath.hyp2f2(1., 1., 5. / 2., 3., x * x / 2.)) / 6. \
          + math.pi * (1 - x * x) * scipy.special.erfi(x / math.sqrt(2.)) \
          - 3. * x * x \
          + math.sqrt(2. * math.pi) * math.exp(x * x / 2.) * x \
          + 2.


def N_func(m, t, mu, nu, H, epsilon, tau):
    D = 2. * H * tau**(2. * H - 1.)

    return m * 0.5 * (mu / D + nu) + \
           t * 0.25 * (mu * mu * t**(-2. * epsilon) + nu * nu * t**(2. * epsilon))


def eigenvalues(N, H):
    covariances = np.zeros(2 * N)
    for k in range(0, N):
        covariances[k] = gamma(k, H)
    for k in range(N + 1, 2 * N):
        covariances[k] = gamma(2 * N - k, H)

    return np.sqrt(np.real(np.fft.ifft(covariances)))


def fbm_path(N, H, eigenvalues, drift):
    np.random.seed()
    V = np.random.normal(0, 1, 2 * N)
    W = np.zeros(2 * N, complex)

    W[0] = V[0]
    W[N] = V[N]

    W[1:N] = (V[1:N] + V[N + 1:2 * N] * 1j) / np.sqrt(2.)
    W2 = (V[1:N] - V[N + 1:2 * N] * 1j) / np.sqrt(2.)
    W[N + 1:2 * N] = W2[::-1]

    Z = np.real(np.fft.fft(eigenvalues * W))
    return np.cumsum(Z[0:N]) / N**H + drift


def fbm_noise(N, eigenvalues):
    V = np.random.normal(0, 1, 2 * N)
    W = np.zeros(2 * N, complex)

    W[0] = V[0]
    W[N] = V[N]

    W[1:N] = (V[1:N] + V[N + 1:2 * N] * 1j) / np.sqrt(2.)
    W2 = (V[1:N] - V[N + 1:2 * N] * 1j) / np.sqrt(2.)
    W[N + 1:2 * N] = W2[::-1]

    return np.real(np.fft.fft(eigenvalues * W))


def drift(mu, nu, N, H):
    return [mu * t + nu * t**(2. * H) for t in np.linspace(1. / N, 1., num=N)]


def empirical_cdf(array, t):
    return len(array[array < t]) / len(array)

