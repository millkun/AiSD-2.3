# Speech Recognition for EFM32 Microcontroller

This project implements a speech recognition system designed specifically for the EFM32 Room 32 38-pin microcontroller. The system processes audio input from a microphone using Fourier and cosine transforms to recognize commands.

## Project Structure

```
speech-recognition-efm32
├── src
│   ├── main.py                     # Entry point of the application
│   ├── audio_processing             # Module for audio processing
│   │   ├── __init__.py             # Package initialization
│   │   ├── fft.py                   # Fast Fourier Transform functions
│   │   ├── dct.py                   # Discrete Cosine Transform functions
│   │   └── spectrogram.py           # Spectrogram generation functions
│   ├── command_recognition          # Module for command recognition
│   │   ├── __init__.py              # Package initialization
│   │   ├── recognizer.py            # Command recognition logic
│   │   └── reference_loader.py      # Reference command loading
│   ├── utils                        # Utility functions
│   │   ├── __init__.py              # Package initialization
│   │   └── file_utils.py            # File operations utilities
├── tests                            # Unit tests for the project
│   ├── test_audio_processing.py      # Tests for audio processing
│   ├── test_command_recognition.py   # Tests for command recognition
│   └── test_utils.py                 # Tests for utility functions
├── requirements.txt                 # Project dependencies
├── README.md                        # Project documentation
└── .gitignore                       # Git ignore file
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd speech-recognition-efm32
   ```

2. **Install dependencies**:
   Use the following command to install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the speech recognition program, execute the following command:
```
python src/main.py
```

## Examples

1. **Recognizing Commands**:
   After running the program, speak a command into the microphone. The system will process the audio and attempt to recognize the command based on pre-loaded references.

2. **Testing**:
   You can run the unit tests to ensure everything is functioning correctly:
   ```
   pytest tests/
   ```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.