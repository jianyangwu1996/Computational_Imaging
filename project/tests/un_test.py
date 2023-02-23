import matplotlib.pyplot as plt
import numpy as np
import skimage.data as skdata
from project.algorithms.simulation import dummy_object, ptychogram, mesh, illumination_beam
from project.algorithms.utils import circ_aperture, normalize, nrmse, ft, ift, corr
from project.algorithms.reconstruction import update_obj, update_probe, TransRefinement


"ground truth"
intensity = np.array(plt.imread('lena.tif'))
phase = skdata.camera()
obj = dummy_object(intensity=intensity, phase=phase, output_shape=(256, 256))
box_shape = (161, 161)   # the size of reconstruction box
r = 0.625
illumination = illumination_beam(box_shape, beam_radius=r)
illumination = normalize(illumination)

"positions and ptychogram"
positions = mesh((256, 256), 50, 0.9, 7, error=2)
# positions = np.load('positions.npy')
patterns = []
for position in positions:
    pattern = ptychogram(obj, illumination, position)
    patterns.append(pattern)

"initial guess"
guess_probe = circ_aperture(box_shape, radius=0.4).astype('complex')
# guess_positions = mesh((256, 256), 30, 0.9, 7, error=0)
guess_positions = np.array([[128, 128]]*len(positions))
guess_obj = np.ones(obj.shape, dtype="complex")

"initial value of parameters"
a, b = 1.0, 1.0
n_iter = 200
nu = 30
tau = 0.3
beta = np.array([[8000, 8000]]*len(positions)).astype('float32')

"dataset need to be saved"
loss = []   # a list to store NRMSE for each iteration
dpx, dpy, kx, ky = [], [], [], []

"reconstruction"
(K, L) = guess_probe.shape
obj_pad = np.pad(guess_obj, ((K // 2, K // 2), (L // 2, L // 2)))

for n in range(n_iter):
    NRMSEs = []
    pos_shifts = []
    syn, sxn = [], []
    temp_pos = guess_positions.copy()
    for pattern, i in zip(patterns, range(len(guess_positions))):
        x = guess_positions[i][1]
        y = guess_positions[i][0]
        obj_scanned = obj_pad[y:y + K, x:x + L]

        "revise the wave function in diffraction plane"
        psi = obj_scanned * guess_probe
        Psi = ft(psi)
        phase_Psi = np.exp(1j * np.angle(Psi))
        Psi_corrected = np.sqrt(pattern) * phase_Psi
        psi_corrected = ift(Psi_corrected)

        "update the object and probe functions"
        diff_psi = psi_corrected - psi
        temp_obj = obj_scanned.copy()
        obj_scanned = update_obj(obj_scanned, guess_probe, diff_psi, learning_rate=a)
        obj_pad[y:y + K, x:x + L] = obj_scanned
        guess_probe = update_probe(guess_probe, temp_obj, diff_psi, learning_rate=b)

        NRMSE = nrmse(np.abs(Psi), pattern)
        NRMSEs.append(NRMSE)

        "revise scann positions"
        if n >= 1:
            mask = guess_probe > 0.1
            syj, sxj = TransRefinement(obj_scanned, temp_obj)
            dy, dx = round(syj * beta[i, 0]), round(sxj * beta[i, 1])
            y += dy
            x += dx

            if y < 0:
                y = 1
            elif y >= guess_obj.shape[0]:
                y = guess_obj.shape[0] - 1
            if x < 0:
                x = 1
            elif x >= guess_obj.shape[1]:
                x = guess_obj.shape[1] - 1
            guess_positions[i] = np.array([y, x])
            syn.append(syj)
            sxn.append(sxj)

    if n >= 1:
        dpx.append(guess_positions[:, 1] - temp_pos[:, 1])   # the x-direction shift errors in the n-th iteration
        dpy.append(guess_positions[:, 0] - temp_pos[:, 0])   # the y-direction shift errors in the n-th iteration
        temp_pos = guess_positions.copy()
    if len(dpx) >= 2:
        kx.append(corr(dpx[-1], dpx[-2]))
        ky.append(corr(dpy[-1], dpy[-2]))

        "adjust beta"
        # if kx[-1] < -0.3:
        #     beta[:, 1] *= 0.9
        # elif kx[-1] > 0.3:
        #     beta[:, 1] *= 1.3
        #
        # if ky[-1] < -0.3:
        #     beta[:, 0] *= 0.9
        # elif ky[-1] > 0.3:
        #     beta[:, 0] *= 1.3
        #
        #     if (n + 1) > (n_IPs + nu):
        #         n_IPs = n
        #         IPs_index = np.where(NRMSEs > (min(NRMSEs) + tau))
        #         IPs = np.array(guess_positions)[IPs_index]
        #         CPs_index = np.where(NRMSEs <= (min(NRMSEs) + tau))
        #         CPs = np.array(guess_positions)[CPs_index]
        #
        #         intensity_IPs = []
        #         for IP in IPs:
        #             x = IP[1]
        #             y = IP[0]
        #             obj_scanned = obj_pad[y:y + K, x:x + L]
        #             psi = obj_scanned * guess_probe
        #             Psi = ft(psi)
        #             intensity_IPs.append(Psi**2)
        #
        #         intensity_CPs = []
        #         for CP in CPs:
        #             x = CP[1]
        #             y = CP[0]
        #             obj_scanned = obj_pad[y:y + K, x:x + L]
        #             psi = obj_scanned * guess_probe
        #             Psi = ft(psi)
        #             intensity_CPs.append(Psi**2)
        #
        #         for i, iip in zip(IPs_index[0], intensity_IPs):
        #             CCs = []
        #             for icp in intensity_CPs:
        #                 CC = corr(iip, icp)
        #                 CCs.append(CC)
        #             index = np.argmax(CCs)
        #             guess_positions[i] = CPs[index]
        #         beta[:, 0] = np.square(loss[n] / np.min(loss[n]) + 0.005)
        #         beta[:, 0] = np.square(loss[n] / np.min(loss[n]) + 0.005)


    loss.append(NRMSEs)
    guess_obj = obj_pad[K // 2:-K // 2 + 1, L // 2:-L // 2 + 1]

plt.figure()
plt.scatter(*np.transpose(positions))
# plt.scatter(*np.transpose(positions_ori))
plt.scatter(*np.transpose(guess_positions))
plt.title('retrieved positions')
# plt.xlim(0, 256)
# plt.ylim(0, 256)
plt.show()

# plt.figure()
# loss = np.array(loss)
# plt.plot(np.mean(loss, 1))
# plt.show()

plt.title('correlation coefficient of two set of position errors')
plt.plot(kx, label='kx')
plt.plot(ky, label='ky')
plt.xlabel('iteration')
plt.ylabel('correlation coefficient')
plt.legend()
plt.show()
