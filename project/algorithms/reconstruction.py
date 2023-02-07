import numpy as np
from project.algorithms.simulation import propagate
from utils import nrmse


def epie_unknown_pos(patterns, guess_obj, guess_probe, guess_positions, n_iter=150, delta=2, nu=30, beta=8e3, tau=0.1):
    """
    recovery procedure of the unknown-position ePIE
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

    n_IPs = 1
    NRMSEs = [0] * len(guess_positions)
    e = [(0,0)] * len(guess_positions)
    for n in range(n_iter):
        for position, pattern, i in zip(guess_positions, patterns, range(len(guess_positions))):
            phi = guess_obj * guess_probe
            Phi = propagate_for(phi)
            phase_Phi = np.exp(1j * np.angle(Phi))
            Phi_corrected = np.sqrt(pattern) * phase_Phi
            NRMSEs[i] = nrmse(pattern, Phi_corrected)
            temp_obj = guess_obj.copy()
            guess_obj = update_obj(guess_obj, guess_probe, position)
            guess_probe = update_probe(guess_probe, temp_obj, position)
            e = cross_correlation(guess_obj, temp_obj)
            position += beta * e[i]
        maxshift =
        beta = update_beata()

        if n+1>2 & maxshift<delta & n+1>n_IPs+nu:
            n_IPs =n
            IPs_index = np.where(NRMSEs > NRMSEs.min()+tau)
            IPs = guess_positions[IPs_index]
            CPs = list(set(guess_positions) - set(IPs))
            for i in IPs_index:
                IP = guess_positions[i]
                CCs = []
                CC = corr()
                index = np.argmax(CCs)
                guess_positions[i] = CPs[index]
            beta = np.square()
