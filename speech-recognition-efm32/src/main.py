import sounddevice as sd
import numpy as np
from audio_processing.fft import compute_fft
from audio_processing.dct import compute_dct
from audio_processing.spectrogram import generate_spectrogram
from command_recognition.recognizer import CommandRecognizer
from command_recognition.reference_loader import load_references

SAMPLE_RATE = 44100
DURATION = 2.0

def record_audio():
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def main():
    print("Recording audio...")
    audio_data = record_audio()
    
    print("Processing audio...")
    spectrogram_data = generate_spectrogram(audio_data, SAMPLE_RATE)
    
    references = load_references()
    recognizer = CommandRecognizer(references)
    
    recognized_command = recognizer.recognize_command(spectrogram_data)
    
    if recognized_command:
        print(f"Recognized command: {recognized_command}")
    else:
        print("No command recognized.")

if __name__ == "__main__":
    main()