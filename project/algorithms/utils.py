import numpy as np
from scipy.stats import multivariate_normal


def ft(image):
    """fft + fftshift for a give image"""
    return np.fft.fftshift(np.fft.fft2(image))


def ift(pattern):
    """ifftshift + ifft for a given diffraction pattern"""
    return np.fft.ifft2(np.fft.ifftshift(pattern))


def normalize(arr):
    temp = np.abs(arr).max()
    arr /= temp
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
