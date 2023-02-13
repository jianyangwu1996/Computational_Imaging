import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as color
import skimage.data as skdata
from project.algorithms.simulation import dummy_object, ptychogram, mesh, illumination_beam
from project.algorithms.utils import circ_aperture, normalize, nrmse, ft, ift
from project.algorithms.reconstruction import update_obj, update_probe
from scipy.signal import correlate


intensity = np.array(plt.imread('lena.tif'))
phase = skdata.camera()
obj = dummy_object(intensity=intensity, phase=phase, output_shape=(128, 128))
box_shape = (81, 81)   # the size of reconstruction box
r = 0.75
illumination = illumination_beam(box_shape, beam_radius=r)
illumination = normalize(illumination)
norm = color.Normalize(-np.pi, np.pi)
rainbow = cm.ScalarMappable(norm=norm, cmap='rainbow')
gray = cm.ScalarMappable(cmap='gray')

positions = np.load('positions.npy')
patterns = []
for position in positions:
    pattern = ptychogram(obj, illumination, position)
    patterns.append(pattern)

guess_probe = circ_aperture(box_shape, radius=0.6)*(1+0j)
guess_positions = mesh((128, 128), 30, 0.7, 7, error=False)
guess_obj = circ_aperture(obj.shape) * 0.8 * (1+1j)
# guess_obj = np.ones(obj.shape, dtype="complex")
a = b = 1.0

n_iter = 5
(K, L) = guess_probe.shape
obj_pad = np.pad(guess_obj, ((K // 2, K // 2), (L // 2, L // 2)))

loss = []

for n in range(n_iter):
    NRMSEs = []
    pos_shifts = []
    for pattern, i in zip(patterns, range(len(guess_positions))):
        x = guess_positions[i][1]
        y = guess_positions[i][0]
        obj_scanned = obj_pad[y:y + K, x:x + L]

        # revise the wave function in diffraction plane
        psi = obj_scanned * guess_probe
        Psi = ft(psi)
        phase_Psi = np.exp(1j * np.angle(Psi))
        Psi_corrected = np.sqrt(pattern) * phase_Psi
        psi_corrected = ift(Psi_corrected)
        NRMSE = nrmse(np.abs(Psi), pattern)

        # update the object and probe functions
        diff_psi = psi_corrected - psi
        temp_obj = obj_scanned.copy()
        obj_scanned = update_obj(obj_scanned, guess_probe, diff_psi, learning_rate=a)
        obj_pad[y:y + K, x:x + L] = obj_scanned
        guess_probe = update_probe(guess_probe, temp_obj, diff_psi, learning_rate=b)

        # revise scann positions
        # cc = correlate(temp_obj, obj_scanned, mode='same', method='fft')
        T = np.zeros((4051, 4051), dtype='complex')
        S = np.zeros((4051, 4051), dtype='complex')
        Te = ft(temp_obj)
        Sa = ft(np.conjugate(obj_scanned))
        T[4051//2-40:4051//2+41, 4051//2-40:4051//2+41] = Te
        S[4051 // 2 - 40:4051 // 2 + 41, 4051 // 2 - 40:4051 // 2 + 41] = Sa
        cc = ift(T*S)
        dy, dx = np.where(cc == cc.max())
        dy, dx = round(dy[0]/50) - K // 2, round(dx[0]/50) - L // 2
        y += dy
        x += dx
        if y < 0:
            y = 0
        elif y > guess_obj.shape[0]:
            y = guess_obj.shape[0]

        if x < 0:
            x = 0
        elif x > guess_obj.shape[1]:
            x = guess_obj.shape[1]

    # position += beta[i] * e[i]
        guess_positions[i] = (y, x)
        pos_shift = np.sqrt(dx**2 + dy**2)
        pos_shifts.append(pos_shift)
        NRMSEs.append(NRMSE)
    loss.append(NRMSEs)

plt.figure()
plt.scatter(*np.transpose(positions))
plt.scatter(*np.transpose(guess_positions))
plt.xlim(0, 128)
plt.ylim(0, 128)
plt.show()

plt.figure()
loss = np.array(loss)
plt.plot(np.mean(loss, 1))
plt.show()
# plt.ylim(0.6, 0.7)

plt.figure()
plt.imshow(np.abs(temp_obj))
plt.show()
plt.imshow(np.abs(obj_scanned))
plt.show()
# plt.figure()
# plt.imshow(np.abs(cc))
# plt.show()
print(dx, dy)

