import numpy as np
import sounddevice as sd
from scipy import signal
import pytest
from src.audio_processing.fft import compute_fft
from src.audio_processing.dct import compute_dct
from src.audio_processing.spectrogram import generate_spectrogram

@pytest.fixture
def audio_sample():
    fs = 44100  # Sample rate
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # A 440 Hz sine wave
    return audio

def test_compute_fft(audio_sample):
    freq_components = compute_fft(audio_sample)
    assert freq_components.shape[0] > 0  # Ensure some frequency components are returned

def test_compute_dct(audio_sample):
    dct_output = compute_dct(audio_sample)
    assert dct_output.shape[0] > 0  # Ensure some DCT output is returned

def test_generate_spectrogram(audio_sample):
    freqs, times, Sxx = generate_spectrogram(audio_sample, fs=44100)
    assert freqs.shape[0] > 0  # Ensure frequency bins are returned
    assert times.shape[0] > 0  # Ensure time bins are returned
    assert Sxx.shape[0] > 0 and Sxx.shape[1] > 0  # Ensure spectrogram data is returned