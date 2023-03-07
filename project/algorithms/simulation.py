import numpy as np
import skimage.data as skdata
from skimage.transform import resize
from scipy.signal import convolve2d
from project.algorithms.utils import ft, normalize, frashift, circ_aperture, circ_gauss


def dummy_object(n_periods=1, intensity=None, phase=None, output_shape=None):
    """
    creat a dummy object for experiment
    """

    if intensity is None:
        intensity = skdata.carema()
    intensity = normalize(intensity)

    if phase is None:
        phase = skdata.brick()
    phase = normalize(phase)

    if output_shape is None:
        output_shape = (128, 128)
    intensity = resize(intensity, output_shape)
    phase = resize(phase, output_shape)

    obj = intensity * np.exp(2j * np.pi * n_periods * phase)
    return obj


def illumination_phase(beam_shape: tuple, focal_length: float):
    """
    Adding quadratic phase based on wave propagation theory. The phase can be controlled with
    focal length with this function. You can also control this with changing experimental geometry.
    :param beam_shape: tuple
                    shape of the aperture
    :param focal_length: float
                    focal length of the lens
    :return: phase_factor: ndarray
                    2d phase factor which is passed to `experimental_beam`
    """

    # setting some experimental geometry
    wavelength = 632.8e-9  # wavelength in nm
    zo = 5e-2  # distance between sample and detector
    Np = beam_shape[0]

    # Calculate the length of the detector
    dxd = 16 * 4.5e-6  # pixel size of the detector
    Ld = Np * dxd  # length of the detector

    # probe estimation
    dxp = wavelength * zo / Ld
    xp = np.arange(-Np // 2, Np // 2) * dxp
    Yp, Xp = np.meshgrid(xp, xp)

    # quadratic phase that can be changed by the focal length
    phase_factor = 2 * np.pi / wavelength * (Xp ** 2 + Yp ** 2) / (2 * focal_length)

    return phase_factor


def illumination_beam(beam_shape: tuple, beam_radius: float, kernel_shape: tuple = (10, 10), kernel_mu: float = 0.0,
                      kernel_sigma: float = 0.6, add_phase: bool = True, focal_length: float = 5e-3):
    """
    Experimental beam is simply an aperture that is convolved with a small guassian kernel to deblurr
    the edges of the aperture. The beam can also have a quadratic phase based on the setup (2f/4f).

    :param beam_shape: tuple
                    shape of the aperture
    :param beam_radius: float
                    radius of the aperture
    :param kernel_shape: tuple
                    shape of the gaussian kernel being used
    :param kernel_mu: float
                    mean of the gaussian kernel
    :param kernel_sigma: float
                    standard deviation of the gaussian kernel
    :param add_phase: bool
                    whether to add phase
    :return: probe: ndarray
                    aperture whose edges are blurred and a phase is added (optional).
    :param focal_length: float
                    controls the quadratic phase by changing the focal length of the lens
    """

    # creating an aperture and a gaussian kernel
    aperture = circ_aperture(beam_shape, beam_radius)
    gauss2d = circ_gauss(kernel_shape, kernel_mu, kernel_sigma)

    # convolving these two for blurring the edges of the aperture
    probe = convolve2d(aperture, gauss2d, mode='same')

    # add quadratic phase if needed
    if add_phase:
        factor = illumination_phase(beam_shape, focal_length)
        probe = probe * np.exp(-1j * factor)
        return probe

    return probe


def mesh(object_size: tuple, radius, overlap: float, n: int, error: int):
    """
    define the position of scan points with mesh-grid
    :param object_size: the size of scanned object
    :param radius: the radius of illumination beam
    :param overlap: overlap rate
    :param n: the number of scan points per row
    :param error: the maximum position error
    :return: the positions of scann points
    """

    distance = int(2 * radius * (1-overlap))
    x = distance * np.arange(n)
    start = (object_size[1] - x.max()) // 2
    x += start
    X, Y = np.meshgrid(x, x)
    if error != 0:
        xerror = np.random.randint(-error, error+1, size=X.shape)
        X += xerror
        yerror = np.random.randint(-error, error+1, size=X.shape)
        Y += yerror
    positions = list(zip(X.flat, Y.flat))

    return positions


def ptychogram_pad(obj, probe, position):
    """
    pad half shape of probe to object and then produce ptychogram with slicing_based scanning
    :param obj: the experiment object
    :param probe: the illumination probe
    :param position: the position of scanning area
    :return: intensity of diffraction pattern
    """

    Y, X = probe.shape
    y, x = position

    #pad object with half shape of probe in the 4 edges to keep the diffraction patterns in the same size
    obj = np.pad(obj, ((Y//2, Y//2), (X//2, X//2)))

    obj_slice = obj[y:y+Y, x:x+Y]
    esw = obj_slice * probe
    pattern = ft(esw)
    data = np.square(np.abs(pattern))

    return data


def ptychogram_shift(obj, probe, position):
    """
    produce ptychogram with object shift
    :param obj: the experiment object
    :param probe: the illumination probe
    :param position: the position of scanning area
    :return: intensity of diffraction pattern
    """

    K, L = probe.shape
    M, N = obj.shape
    y, x = position

    # shift the scan points to the center of the object
    shift = -np.array([y - M // 2, x - N // 2]).astype('float')
    obj_shift = frashift(obj, shift)
    obj_scanned = obj_shift[M // 2 - K // 2: M // 2 + K // 2 + 1, N // 2 - L // 2: N // 2 + L // 2 + 1]

    esw = obj_scanned * probe
    pattern = ft(esw)
    data = np.square(np.abs(pattern))

    return data
