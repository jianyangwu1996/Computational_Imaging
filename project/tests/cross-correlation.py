from scipy.signal import correlate
import numpy as np
from numpy.fft import fft, fft2, ifft, ifft2
import skimage.data as skdata
from skimage.transform import resize
from project.algorithms.utils import ft, ift, fft_correlate
import matplotlib.pyplot as plt


# 1D cross-correlation with fft
a = np.random.random(9)
b = np.random.random(9)
sc = correlate(a, b, 'same', 'fft')
Fa16_con = np.conj(fft(a, 16))
Fb16 = fft(b, 16)
fc16 = ifft(Fa16_con * Fb16).real
fc = np.hstack((fc16, fc16))
fc = fc[20:11:-1]
np.allclose(fc, sc)

# 2D cross-correlation with fft (random data)
a = np.random.random((81, 81))
b = np.random.random((81, 81))
sc = correlate(a, b, 'same', 'fft')
Fa16_con = np.conj(fft2(a, [128, 128]))
Fb16 = fft2(b, [128, 128])
fc16 = ifft2(Fa16_con * Fb16).real
temp = np.hstack((fc16, fc16))
temp = np.vstack((temp, temp))
fc = temp[88:88+81, 88:88+81]
fc = fc[::-1, ::-1]

# 2D cross-correlation with fft (image)
img1 = resize(skdata.camera(), (81, 81))
img2 = resize(skdata.moon(), (81, 81))
scc = correlate(img1, img2, 'same', 'fft')
fimg1_con = np.conj(fft2(img1, [128, 128]))
fimg2 = fft2(img2, [128, 128])
Fcc = fimg2 * fimg1_con
fcc = ifft2(Fcc).real
temp = np.hstack((fcc, fcc))
temp = np.vstack((temp, temp))
fcc = temp[88:88+81, 88:88+81]
fcc = fcc[::-1, ::-1]

# 2D cross-correlation + upsampling with fft (image)
img2 = resize(skdata.camera(), (81, 81))
img1 = resize(skdata.moon(), (81, 81))
scc = correlate(img1, img2, 'same', 'fft')
fimg1_con = np.conj(ft(img1, [128, 128]))
fimg2 = ft(img2, [128, 128])
Fcc1 = fimg1_con * fimg2
i = 50
Fcc_pad = np.zeros((128*i, 128*i), dtype='complex')
Fcc_pad[64*(i-1):128+64*(i-1), 64*(i-1):64*(i-1)+128] = Fcc1
fcc = ift(Fcc_pad).real
temp = np.hstack((fcc, fcc))
temp = np.vstack((temp, temp))
fcc = temp[88*i:(88*i+81*i), 88*i:(88*i+81*i)]
fcc = fcc[::-1, ::-1]
plt.imshow(fcc)
plt.show()
