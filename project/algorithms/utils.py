import numpy as np
from scipy.stats import multivariate_normal
from numpy.fft import fft2, ifft2


def ft(image, s=None):
    """fft + fftshift for a give image"""
    return np.fft.fftshift(np.fft.fft2(image, s))


def ift(pattern):
    """ifftshift + ifft for a given diffraction pattern"""
    return np.fft.ifft2(np.fft.ifftshift(pattern))


def normalize(arr):
    temp = np.abs(arr).max()
    arr = arr / temp
    return arr


def nrmse(arr1, arr2):
    """
    calutate the normalized-root-mean-square-error
    :param arr1: guessed diffraction pattern
    :param arr2: recorded pattern intensity
    :return:
    """

    res = np.sqrt(np.mean(np.square(np.sqrt(arr2) - np.abs(arr1)))) / np.mean(np.sqrt(arr2))
    return res


def corr(a, b):
    """Cross-correlation coefficient. """
    return np.corrcoef(a.flat, b.flat)[0, 1]


def hsv_convert(img):
    """
    use phase and intensity to produce a hsv image and then convert to rgb image
    :param img: a complex image which will be converted to hsv representation
    :return: a rgb image
    """

    # produce the hsv image
    h = np.angle(img)
    h = np.rad2deg(h) / 360
    h[h < 0] = h[h < 0] + 1
    s = np.ones(img.shape)
    v = np.abs(img)

    # convert the hsv image to rgb image
    hi = (h * 6.0).astype('uint8')
    f = (h * 6.0) - hi
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    conditions = [s == 0.0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5]

    r = np.select(conditions, [v, q, p, p, t, v], default=v)
    g = np.select(conditions, [v, v, v, q, p, p], default=t)
    b = np.select(conditions, [v, p, t, v, v, q], default=p)

    rgb = np.array([r, g, b]).transpose((1, 2, 0))
    return rgb


def circ_aperture(shape: tuple, radius: float = 0.5):
    """
    creates a simple circular binary mask centered in the middle with maximum radius of 1.
    :param shape: a tuple that provides the shape of the beam
    :param radius: radius of the circle
    """

    assert radius <= 1

    x = np.linspace(-1, 1, shape[1])
    y = np.linspace(-1, 1, shape[0])
    Y, X = np.meshgrid(x, y)

    circ = X ** 2 + Y ** 2 <= radius ** 2
    return circ.astype('int')


def circ_gauss(shape: tuple, mu: float = 0.0, sigma: float = 0.3):
    """
    An isotropic bivariate gaussian beam with a covariance of 1.

    :param shape: shape of the beam
    :param mu: mean of the gaussian function
    :param sigma: standard deviation
    """

    # either pass a 2d value of sigma/mu or it copies the same scalar value
    mu = tuple(np.repeat(mu, 2)) if np.isscalar(mu) else mu
    sigma = tuple(np.repeat(sigma, 2)) if np.isscalar(sigma) else sigma

    # isotropic covariance matrix (can also be general if needed)
    covmat = np.diag([sigma[0] ** 2, sigma[1] ** 2])

    # pass some values and create a meshgrid
    x = np.linspace(-1, 1, shape[1])
    y = np.linspace(-1, 1, shape[0])
    X, Y = np.meshgrid(x, y)

    # create a 2d gaussian function
    gauss = multivariate_normal(mu, covmat)
    pdf = gauss.pdf(np.dstack([X, Y]))

    return pdf

def frashift(a, p):
    M, N = a.shape
    x = np.concatenate((np.arange(0, np.floor(N/2)+1), np.arange(-np.ceil(N/2)+1, 0)))
    y = np.concatenate((np.arange(0, np.floor(M/2)+1), np.arange(-np.ceil(M/2)+1, 0)))

    x = np.exp(-2j*np.pi*x*p[1]/N)
    y = np.exp(-2j*np.pi*y*p[0]/M)

    if M % 2 == 0:
        y[M//2] = y[M//2].real
    if N % 2 == 0:
        x[N//2] = x[N//2].real

    X, Y = np.meshgrid(x, y)

    H = X * Y
    b = ifft2(H * fft2(a))
    return b