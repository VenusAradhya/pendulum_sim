
"""
Function to convert an Amplitude Spectral Density (ASD) into a
randomised timeseries.

Could be a class - after core functionality is written.
Could have 'bands' of the probabilistic ASD be randomly assigned.
 > Low frequency < ~ 50 mHz
 > Microseismic ~ 50 mHz to ~ 500 mHz
 > Anthropogenic > ~ 500 mHz
 * Randomise precise cut-off.
 * Generate, low order, blending filters in F Domain.
   Stability is not a requirement.
   Order can be randomised.
 * Generate a z score for each band.
 * Create an ASD for each z score.
 * F Domain filter the created ASDs.
 * Sum the filtered ASDs.
"""


import numpy as np
import scipy.signal as sg
import scipy.integrate as si
# Needs replacing/updating - not updated/supported.
# https://docs.scipy.org/doc/scipy-1.17.0/tutorial/interpolate/extrapolation_examples.html
from scipy.interpolate import interp1d
from datetime import datetime


__all__ = ['disturbance_noise_file',
           'sensor_noise_file',
           'asd_from_asd_statistics',
           'asd_to_timeseries',
           'calculate_RMS']


# Disturbance noise file.
disturbance_noise_file = '2013.Charles.40m.elog8786.20130628seismicNoiseMeters.csv'

# Sensor noise file.
sensor_noise_file = 'aosem_noise.csv'


def asd_from_asd_statistics(mean_asd : np.typing.ArrayLike,
                            stddev_asd : np.typing.ArrayLike,
                            deterministic : bool = False,
                            z_score : float | None = None,
                            seed : int | np.random.Generator = datetime.now().microsecond) \
                           -> np.typing.NDArray:

    """
    Randomly generate an amplitude spectral density from a statistical range of
    amplitude spectral densities - see probabilitic power spectral density.
    
    :param mean_asd: Per bin, mean value of the amplitude spectral density.
    :type mean_asd: np.typing.ArrayLike[float]
    :param stddev_asd: Per bin, standard deviation of the amplitude spectral density.
    :type stddev_asd: np.typing.ArrayLike[float]
    :param deterministic: Option to generate a deterministic amplitude spectral density.
    :type deterministic: bool
    :param z_score: Deterministic z score, ignored if deterministic is False.
    :type z_score: float | None
    :param seed: Seed for randomisation.
    :type seed: int | np.random.Generator
    :return: Randomised, if instructed, amplitude spectral density.
    :rtype: NDArray[float]
    """

    # Check seeding
    if issubclass(type(seed), np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed = seed)

    # Check deterministic kwargs.
    if deterministic and (z_score is None):
        err = 'Keyword argument z_score must be provided when keyword' + \
              ' argument deterministic is True.'
        raise RuntimeError(err)
    

    # Get the z score to use.
    if deterministic:
        Zscore = float(z_score)
    else:
        Zscore = rng.normal(0, 1, size = None)
    

    # Convert ASDs to dB PSDs.
    mean_dB_PSD = 20*np.log10(mean_asd)
    stddev_dB_PSD = 20*np.log10(mean_asd + stddev_asd) - mean_dB_PSD
    

    # Z score to sample.
    PSD_dB = stddev_dB_PSD * Zscore + mean_dB_PSD
    

    # Probabilistic ASD
    return 10**(PSD_dB / 20)


def asd_to_timeseries(duration : float, sample_rate : float,
                      frequencies : np.typing.ArrayLike,
                      amplitude_spectral_density : np.typing.ArrayLike,
                      seed : int | np.random.Generator = datetime.now().microsecond) \
                      -> np.typing.NDArray:
    
    """
    Function to convert from an amplitude spectral density to a timeseries.
    
    :param duration: Duration, s, of the generated timeseries.
    :type duration: float
    :param sample_rate: Sample rate of the generated timeseries (Hz).
    :type sample_rate: float
    :param frequencies: Frequency vector (Hz).
    :type frequencies: np.typing.ArrayLike[float]
    :param amplitude_spectral_density: Amplitude spectral density.
    :type amplitude_spectral_density: np.typing.ArrayLike[float]
    :param seed: Seed for randomisation of the timeseries phase.
    :type seed: int | np.random.Generator
    :return: Random timeseries with the given aplitude spectral density.
    :rtype: NDArray[float]
    """

    # Check seeding
    if issubclass(type(seed), np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed = seed)
    

    # Number of points.
    # A power of 2 is used for faster FFT computations
    num_points = int(duration * sample_rate)
    gen_points = 2**(int(np.ceil(np.log2(2 * num_points))) + 1)

    # Frequencies that will be generated
    gen_freq = np.fft.rfftfreq(gen_points, d = 1 / sample_rate)
    mask_gen = np.greater(gen_freq, 0)


    # Random phase - white spectrum
    random_noise = rng.standard_normal(size = gen_points)
    random_phase = np.fft.rfft(random_noise * \
                               sg.get_window(('tukey', 0.2), gen_points))
    # No DC power.
    random_phase[0] = 0
    

    # Mask - 0 Hz is invalid.
    mask_in = np.greater(frequencies, 0)

    # Generate the asd points.
    asd_gen = interp1d(np.log10(frequencies[mask_in]),
                       np.log10(amplitude_spectral_density[mask_in]),
                       kind = 'linear', bounds_error = False,
                       fill_value = 'extrapolate')
    target_asd = asd_gen(np.log10(gen_freq[mask_gen]))

    # Scale random phase.
    random_phase *= np.sqrt(sample_rate / 2)
    random_phase[mask_gen] *= target_asd


    # Convert to time domain.
    timeseries = np.fft.irfft(random_phase)

    # Cut out the circular transients in the 1st/last quarter.
    return timeseries[gen_points//4 : gen_points//4 + num_points]


def calculate_RMS(x : np.typing.ArrayLike,
                  y : np.typing.ArrayLike) \
                 -> np.typing.NDArray:

    """
    Calculate the RMS of an amplitude sprectral density or
    timeseries signal.
    
    :param x: Frequency, in Hz, or zeroed time, in seconds - ordered.
    :type x: np.typing.ArrayLike[float]
    :param y: Amplitude spectral density or time series - ordered.
    :type y: np.typing.ArrayLike[float]
    :return: Cumulative RMS of the signal
    :rtype: NDArray[float]
    """

    return np.flip(np.sqrt(si.cumulative_trapezoid(np.flip(y**2),
                               -1*np.flip(x), initial = 0)))


if __name__ == '__main__':
    pass