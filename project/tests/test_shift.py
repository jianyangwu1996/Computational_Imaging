import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from project.algorithms.reconstruction import TransRefinement


def circshiftft(a, p):
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



lena = np.array(plt.imread('./project/tests/lena.tif'))
shift = circshiftft(lena, (-10.3, 20.7))
shift = np.abs(circshiftft(shift, (-5, -11)))
fig, axes = plt.subplots(1, 2)
axes[0].imshow(np.abs(lena))
axes[1].imshow(np.abs(shift))
plt.show()

syj, sxj = TransRefinement(shift, lena, integer_skip=False)
