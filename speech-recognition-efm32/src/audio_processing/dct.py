def compute_dct(audio_data):
    from scipy.fftpack import dct
    return dct(audio_data, type=2, norm='ortho')

def compute_idct(dct_data):
    from scipy.fftpack import idct
    return idct(dct_data, type=2, norm='ortho')