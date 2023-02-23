import numpy as np
from project.algorithms.utils import nrmse, ft, ift, corr
from scipy.signal import correlate
import random


def update_obj(obj: np.ndarray, probe: np.ndarray, diff_psi: np.ndarray, learning_rate: float = 1.):
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


def update_probe(probe: np.ndarray, obj: np.ndarray, diff_psi: np.ndarray, learning_rate: float = 1.):
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
         track_error=False, random_update=False):
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
    :return:
    """

    if guess_obj is None:
        guess_obj = np.ones(shape_obj, dtype="complex")
    else:
        guess_obj = np.array(guess_obj, dtype="complex")

    (K, L) = guess_probe.shape
    recon_obj = guess_obj.copy()
    recon_obj_pad = np.pad(recon_obj, ((K // 2, K // 2), (L // 2, L // 2)))

    # corr_intensity_obj = [0]
    # corr_intensity_probe = [0]
    loss = []

    for i in range(n_iter):
        loss_vals = []
        if random_update is True:
            index = random.sample(range(0, len(positions)), len(positions))
        else:
            index = np.arange(0, len(positions))

        for i in index:
            x = positions[i][1]
            y = positions[i][0]
            pattern = ptychogram[i]

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


def TransRefinement(im1, im2, integer_skip=False):
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
        cc = np.mean(out[ty, tx])
        shift_refine = np.array([np.mean(ty), np.mean(tx)])
        shift_refine = shift_refine - win_center
        shift = (shift + shift_refine) / usfac
    sy, sx = shift
    return sy, sx


def epie_unknown_pos(patterns, guess_probe, guess_positions, shape_obj, guess_obj=None, n_iter=150, delta=2, nu=30,
                     beta=8e3, tau=0.2, a=1., b=1., alpha=0.005):
    """
    recovery procedure of the unknown-position ePIE
    :param a: learning rate of object update
    :param b: learning rate of probe update
    :param alpha:
    :param patterns: the recorded diffraction patterns
    :param guess_obj: initial randomly guessed object
    :param guess_probe: initial randomly guessed probe
    :param guess_positions: all-zero positions
    :param n_iter: the number of iteration
    :param delta: the stagnation sign of position refinement process
    :param nu: the least number of iterations being circled before the next round of the process of identifying
                and repositioning the IPs
    :param beta:
    :param tau:
    :return: retrieved complex object, probe and scan position
    """

    if guess_obj is None:
        guess_obj = np.ones(shape_obj, dtype="complex")
    else:
        guess_obj = np.array(guess_obj, dtype="complex")

    (K, L) = guess_probe.shape
    obj_pad = np.pad(guess_obj, ((K // 2, K // 2), (L // 2, L // 2)))
    n_IPs = 1
    beta = np.array([[beta, beta]]*len(guess_positions)).astype('float32')
    loss = []
    sy, sx, kxs, kys = [], [], [], []
    guess_obj = obj_pad[K // 2:-K // 2 + 1, L // 2:-L // 2 + 1]
    for n in range(n_iter):
        NRMSEs = []
        # pos_shifts = []
        syn, sxn = [], []
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
            if n >= 5:
                mask = guess_probe > 0.1
                syj, sxj = TransRefinement(obj_scanned*mask, temp_obj*mask)
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

                # pos_shift = np.sqrt(dx**2 + dy**2)
                # pos_shifts.append(pos_shift)
        loss.append(NRMSEs)
        # if n > 5:
        #     maxshift = max(pos_shifts)
        # else:
        #     maxshift = 0
        sy.append(syn)
        sx.append(sxn)

        "update beta"
        if n > 5:
            kx = corr(np.array(sx[-1]), np.array(sx[-2]))
            ky = corr(np.array(sy[-1]), np.array(sy[-2]))
            if kx < -0.3:
                beta[:, 1] *= 0.9
            elif kx > 0.3:
                beta[:, 1] *= 1.3
            kxs.append(kx)

            if ky < -0.3:
                beta[:, 0] *= 0.9
            elif ky > 0.3:
                beta[:, 0] *= 1.3
            kys.append(ky)

        "repositioning"
        if (n + 1) > (n_IPs + nu):
            n_IPs = n
            IPs_index = np.where(NRMSEs > (min(NRMSEs) + tau))
            IPs = np.array(guess_positions)[IPs_index]
            CPs_index = np.where(NRMSEs <= (min(NRMSEs) + tau))
            CPs = np.array(guess_positions)[CPs_index]

            intensity_IPs = []
            for IP in IPs:
                x = IP[1]
                y = IP[0]
                obj_scanned = obj_pad[y:y + K, x:x + L]
                psi = obj_scanned * guess_probe
                Psi = ft(psi)
                intensity_IPs.append(Psi**2)

            intensity_CPs = []
            for CP in CPs:
                x = CP[1]
                y = CP[0]
                obj_scanned = obj_pad[y:y + K, x:x + L]
                psi = obj_scanned * guess_probe
                Psi = ft(psi)
                intensity_CPs.append(Psi**2)

            for i, iip in zip(IPs_index[0], intensity_IPs):
                CCs = []
                for icp in intensity_CPs:
                    CC = corr(iip, icp)
                    CCs.append(CC)
                index = np.argmax(CCs)
                guess_positions[i] = CPs[index]
            beta[:, 0] = np.square(loss[n] / np.min(loss[n]) + alpha)
            beta[:, 0] = np.square(loss[n] / np.min(loss[n]) + alpha)

    guess_obj = obj_pad[K // 2:-K // 2 + 1, L // 2:-L // 2 + 1]
    return guess_obj, guess_probe, guess_positions, loss
