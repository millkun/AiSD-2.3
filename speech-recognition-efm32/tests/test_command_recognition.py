import unittest
from src.command_recognition.recognizer import CommandRecognizer
from src.command_recognition.reference_loader import load_references

class TestCommandRecognition(unittest.TestCase):

    def setUp(self):
        self.recognizer = CommandRecognizer()
        self.references = load_references()

    def test_recognize_command(self):
        # Assuming we have a method to generate a test spectrogram
        test_spectrogram = self.generate_test_spectrogram()
        recognized_command = self.recognizer.recognize_command(test_spectrogram)
        
        # Replace 'expected_command' with the actual command you expect
        expected_command = 'test_command'
        self.assertEqual(recognized_command, expected_command)

    def test_load_references(self):
        self.assertIsInstance(self.references, dict)
        self.assertGreater(len(self.references), 0)

    def generate_test_spectrogram(self):
        # Placeholder for generating a test spectrogram
        # This should return a spectrogram that can be used for testing
        return None  # Replace with actual spectrogram generation logic

if __name__ == '__main__':
    unittest.main()