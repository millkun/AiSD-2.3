def compute_fft(audio_data):
    """
    Computes the Fast Fourier Transform (FFT) of the given audio data.

    Parameters:
    audio_data (numpy.ndarray): The audio signal data.

    Returns:
    numpy.ndarray: The frequency components of the audio signal.
    """
    from numpy.fft import fft
    return fft(audio_data)

def compute_magnitude_spectrum(fft_data):
    """
    Computes the magnitude spectrum from the FFT data.

    Parameters:
    fft_data (numpy.ndarray): The FFT of the audio signal.

    Returns:
    numpy.ndarray: The magnitude spectrum of the audio signal.
    """
    return np.abs(fft_data)