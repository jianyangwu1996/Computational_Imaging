import numpy as np
from project.algorithms.utils import nrmse, ft, ift

def update_obj(obj: np.ndarray, probe: np.ndarray, diff_psi: np.ndarray, learning_rate: float =1.):
    """
    update recon_object with guess object and guess probe
    :param obj: n-th recon_object
    :param probe: n_th recon_probe
    :param learning_rate:
    :param diff_psi: difference between corrected image and guess image
    :return: (n+1)_th recon_object
    """

    obj += learning_rate * np.conjugate(probe) / np.square(np.abs(probe)).max() * diff_psi
    modulus = np.abs(obj)
    phase = np.angle(obj)
    toohigh = modulus > 1
    obj[toohigh] = 1 * np.exp(1j * phase[toohigh])

    return obj


def update_probe(probe: np.ndarray, obj: np.ndarray, diff_psi: np.ndarray, learning_rate: float =1.):
    """
    update probe with guess object and guess probe
    :param obj: n-th recon_object
    :param probe: n_th recon_probe
    :param learning_rate:
    :param diff_psi: difference between corrected image and guess image
    :return: (n+1)_th recon_probe
    """

    probe += learning_rate * np.conjugate(obj) / np.square(np.abs(obj)).max() * diff_psi

    modulus = np.abs(probe)
    phase = np.angle(probe)
    toohigh = modulus > 1
    probe[toohigh] = 1 * np.exp(1j * phase[toohigh])

    return probe


def epie(ptychogram: np.ndarray, positions, shape_obj, n_iter=100, a=1, b=1, guess_obj=None, guess_probe=None,
         track_error=False, obj_init=None, probe_init=None):
    """
    Reconstruction using the extended pychograhical iterative engine (ePIE) with sliced patterns padding at the edge

    :param ptychogram: experimental diffraction patterns
    :param positions: the scanning map
    :param shape_obj: the object shape
    :param n_iter: the number of iteration
    :param a: learning rate of object reconstruction
    :param b: learning rate of probe reconstruction
    :param guess_obj: initial guess object
    :param guess_probe: initial guess probe
    :param track_error: track the error of reconstruction process or not
    :param obj_init: the real object
    :param probe_init: real probe
    :return:
    """

    if guess_obj is None:
        guess_obj = np.ones(shape_obj, dtype="complex")
    else:
        guess_obj = np.array(guess_obj, dtype="complex")

    K, L = guess_probe.shape
    recon_obj = guess_obj.copy()
    recon_obj_pad = np.pad(recon_obj, ((K // 2, K // 2), (L // 2, L // 2)))

    # corr_intensity_obj = [0]
    # corr_intensity_probe = [0]
    loss = []

    for i in range(n_iter):
        loss_vals = []
        for pattern, position in zip(ptychogram, positions):
            x = position[1]
            y = position[0]

            obj_scanned = recon_obj_pad[y:y + K, x:x + L]
            psi = obj_scanned * guess_probe
            PSI = ft(psi)

            # substitute the intensity of guess with recorded data
            PSI_phase = np.exp(1j * np.angle(PSI))
            PSI_correct = np.sqrt(pattern) * PSI_phase
            psi_correct = ift(PSI_correct)  # corrected diffraction patterns in real domain

            # update guess function
            diff_psi = psi_correct - psi
            guess_obj_old = np.copy(obj_scanned)
            recon_obj_slice = update_obj(obj_scanned, guess_probe, diff_psi, learning_rate=a)
            recon_obj_pad[y:y + K, x:x + L] = recon_obj_slice
            guess_probe = update_probe(guess_probe, guess_obj_old, diff_psi, learning_rate=b)

            loss_vals.append(nrmse(np.abs(PSI), pattern))
        loss.append(np.mean(loss_vals))

        # if track_error is True:
        #     recon_obj = recon_obj_pad[K // 2:-K // 2 + 1, L // 2:-L // 2 + 1]
        #     corr_intensity_obj.append(corr(np.abs(obj_init), np.abs(recon_obj)))
        #     corr_intensity_probe.append(corr(np.abs(probe_init), np.abs(guess_probe)))

    recon_obj = recon_obj_pad[K // 2:-K // 2 + 1, L // 2:-L // 2 + 1]

    if track_error is True:
        return recon_obj, guess_probe, loss
    else:
        return recon_obj, guess_probe


# def epie_unknown_pos(patterns, guess_obj, guess_probe, guess_positions, n_iter=150, delta=2, nu=30, beta=8e3, tau=0.1):
#     """
#     recovery procedure of the unknown-position ePIE
#     :param patterns: the recorded diffraction patterns
#     :param guess_obj: initial randomly guessed object
#     :param guess_probe: initial randomly guessed probe
#     :param guess_positions: all-zero positions
#     :param n_iter: the number of iteration
#     :param delta: the stagnation sign of position refinement process
#     :param nu: the least number of iterations being circled before the next round of the process of identifying
#                 and repositioning the IPs
#     :param beta:
#     :param tau:
#     :return: retrieved complex object, probe and scan position
#     """
#
#     n_IPs = 1
#     NRMSEs = [0] * len(guess_positions)
#     e = [(0,0)] * len(guess_positions)
#     for n in range(n_iter):
#         for position, pattern, i in zip(guess_positions, patterns, range(len(guess_positions))):
#             phi = guess_obj * guess_probe
#             Phi = ft(phi)
#             phase_Phi = np.exp(1j * np.angle(Phi))
#             Phi_corrected = np.sqrt(pattern) * phase_Phi
#             NRMSEs[i] = nrmse(pattern, Phi_corrected)
#             temp_obj = guess_obj.copy()
#             guess_obj = update_obj(guess_obj, guess_probe, position)
#             guess_probe = update_probe(guess_probe, temp_obj, position)
#             e = cross_correlation(guess_obj, temp_obj)
#             position += beta * e[i]
#         maxshift =
#         beta = update_beata()
#
#         if n+1>2 & maxshift<delta & n+1>n_IPs+nu:
#             n_IPs =n
#             IPs_index = np.where(NRMSEs > NRMSEs.min()+tau)
#             IPs = guess_positions[IPs_index]
#             CPs = list(set(guess_positions) - set(IPs))
#             for i in IPs_index:
#                 IP = guess_positions[i]
#                 CCs = []
#                 CC = corr()
#                 index = np.argmax(CCs)
#                 guess_positions[i] = CPs[index]
#             beta = np.square()
