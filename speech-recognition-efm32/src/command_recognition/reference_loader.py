def load_references(reference_dir):
    import os
    import json
    import numpy as np

    references = {}
    for filename in os.listdir(reference_dir):
        if filename.endswith("_spectrogram.json"):
            command = filename.split('-')[1].split('_')[0]
            with open(os.path.join(reference_dir, filename), 'r') as f:
                data = json.load(f)
                freqs = np.array(data["freqs"])
                times = np.array(data["times"])
                Sxx = np.array(data["Sxx"])
                references[command] = (freqs, times, Sxx)
    return references