import numpy as np
from project.algorithms.utils import ift, ft
import matplotlib.pyplot as plt
from project.algorithms.reconstruction import TransRefinement
from project.algorithms.utils import frashift


def TransRefinement1(im1, im2, integer_skip=False):
    """
    Translate from Prof. Dr. Fucai Zhang's matlab function
    :param im1:
    :param im2:
    :param integer_skip:
    :return:
    """
    N_pass = 5
    us = 10
    win = 1.5 * us
    win = win + 15 % 2
    win_center = win // 2

    CS = ft(im1) * np.conj(ft(im2))
    shift = np.array([0, 0])
    ny, nx = im1.shape
    size = np.array([ny, nx])

    if not integer_skip:
        a = np.abs(ift(CS))
        iy, ix = np.where(a == a.max())
        shift = np.array([np.mean(iy), np.mean(ix)])
        shift = shift - (shift > (size // 2)) * size

    # find fractional shift
    p = nx // 2
    x = np.hstack((np.arange(0, nx - p), np.arange(-p, 0)))
    x = np.reshape(x, (nx, 1))
    q = ny // 2
    y = np.hstack((np.arange(0, ny - q), np.arange(-q, 0)))

    winx = np.arange(0, win)
    winy = np.arange(0, win)
    winy = np.reshape(winy, (len(winy), 1))

    usfac = 1
    for i in range(N_pass - 1):
        usfac *= us
        shift = np.round(shift * usfac)
        offset = win_center - shift

        argx = 2 * np.pi / nx / usfac * np.outer(x, (winx - offset[1]))
        kernel_x = np.exp(1j * argx)
        argy = 2 * np.pi / ny / usfac * np.outer((winy - offset[0]), y)
        kernel_y = np.exp(1j * argy)
        out = np.dot(np.dot(kernel_y, CS), kernel_x)
        aout = np.abs(out)
        ty, tx = np.where(aout == aout.max())
        shift_refine = np.array([np.mean(ty), np.mean(tx)])
        shift_refine = shift_refine - win_center
        shift = (shift + shift_refine) / usfac
    sy, sx = shift
    return sy, sx


# lena = np.array(plt.imread('./project/tests/lena.tif'))
# (K, L) = (161, 161)
# M, N = lena.shape
# p = -np.array([127.5 - M//2, 128.3 - N//2]).astype('float')
# shift = frashift(lena, p)
# obj_shift = shift[M//2-K//2:M//2+K//2+1, N//2-L//2:N//2+L//2+1]
# pad = np.pad(lena, ((K // 2, K // 2), (L // 2, L // 2)))
# obj_pad = pad[34:34 + K, 28:28 + L]
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(np.abs(obj_shift))
# axes[1].imshow(np.abs(obj_pad))
# plt.show()
#
# lena = np.fft.fftshift(lena)
# shift = np.fft.fftshift(shift)
#
# syj, sxj = TransRefinement(lena, shift, integer_skip=False)
# syj1, sxj1 = TransRefinement1(lena, shift)
