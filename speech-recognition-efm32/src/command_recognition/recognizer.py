class CommandRecognizer:
    def __init__(self, references):
        self.references = references

    def recognize_command(self, spectrogram_data):
        best_match = None
        best_score = -1
        
        for command, ref_spectrogram in self.references.items():
            score = self.compare_spectrograms(spectrogram_data, ref_spectrogram)
            if score > best_score:
                best_score = score
                best_match = command
        
        return best_match if best_score > 0.8 else None

    def compare_spectrograms(self, current_spec, reference_spec):
        current_flat = current_spec.flatten()
        ref_flat = reference_spec.flatten()
        
        current_norm = current_flat / np.linalg.norm(current_flat)
        ref_norm = ref_flat / np.linalg.norm(ref_flat)
        
        similarity = np.dot(current_norm, ref_norm)
        return similarity