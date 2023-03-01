import numpy as np
from project.algorithms.utils import ft, ift
from numpy.fft import fft2
import matplotlib.pyplot as plt

N_pass = 5
us = 10
win = 1.5 * us
win = win + 15 % 2
win_center = win // 2

CS = fft2(lena) * np.conj(fft2(shift))
shift = np.array([0, 0])
ny, nx = img.shape
size = np.array([ny, nx])
integer_skip = False

if not integer_skip:
    a = np.abs(ift(CS))
    iy, ix = np.where(a == a.max())
    shift = np.array([np.mean(iy), np.mean(ix)])
    shift = shift - (shift > (size//2)) * size

# find fractional shift
p = nx // 2
x = np.hstack((np.arange(0, nx-p), np.arange(-p, 0)))
x = np.reshape(x, (nx, 1))
q = ny // 2
y = np.hstack((np.arange(0, ny-q), np.arange(-q, 0)))

winx = np.arange(0, win)
winy = np.arange(0, win)
winy = np.reshape(winy, (len(winy), 1))

usfac = 1.
for i in range(N_pass-1):
    usfac *= us
    shift = np.round(shift * usfac)
    offset = win_center - shift

    argx = 2 * np.pi / nx / usfac * np.outer(x, (winx - offset[1]))
    kernel_x = np.exp(1j * argx)
    argy = 2 * np.pi / ny / usfac * np.outer((winy - offset[0]), y)
    kernel_y = np.exp(1j * argy)
    out = np.dot(np.dot(kernel_y,  CS), kernel_x)
    aout = np.abs(out)
    ty, tx = np.where(aout == aout.max())
    cc = np.mean(out[ty, tx])
    shift_refine = np.array([np.mean(ty), np.mean(tx)])
    shift_refine = shift_refine - win_center
    shift = (shift + shift_refine) / usfac

sy, sx = shift
print(sy, sx)
# plt.imshow(dst)
# plt.show()
# io.savemat('dst.mat', {'dst':dst})
