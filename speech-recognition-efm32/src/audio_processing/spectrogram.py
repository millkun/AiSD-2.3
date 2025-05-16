def generate_spectrogram(audio_data, sample_rate):
    from scipy import signal
    import numpy as np

    nperseg = 512
    noverlap = nperseg // 2

    freqs, times, Sxx = signal.spectrogram(
        audio_data,
        fs=sample_rate,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='spectrum'
    )

    return freqs, times, 10 * np.log10(Sxx)