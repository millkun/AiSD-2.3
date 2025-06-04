import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import pickle
import os



N_FFT = 512
N_MELS = 64
HOP_LEN = 256
SAMPLE_RATE = 16000
DURATION = 1.5


def get_signal():

    print("Запись началась...")
    signal = sd.rec(int(DURATION * SAMPLE_RATE), 
                   samplerate=SAMPLE_RATE, 
                   channels=1,
                   dtype='float32')
    sd.wait()
    print("Запись завершена!")

    return signal.flatten()


def create_mel_spec(signal):
    fft = librosa.stft(y = signal, n_fft=N_FFT, hop_length=HOP_LEN)
    power_spec = np.abs(fft) ** 2
    mel_spec = librosa.feature.melspectrogram(S=power_spec, sr =SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LEN, n_mels = N_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def normalize(mel_spec_db):
    return (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))


def save_spec(mel_spec, filename):
    with open(filename, 'wb') as f:
        pickle.dump(mel_spec, f)



def fill(FROM = 0, N = 100, debug = False, classes = ['алгоритмы', 'дата', 'время']):
    norm_path = './data/norm/'
    wo_norm_path = './data/wo_normalize/'
    raw_path = './data/raw/'
    for cls in classes:
        for i in range(FROM, FROM + N):
            print(f'запись {i} для класса: {cls}')
            input('enter for record')
            signal = get_signal()
            if not debug or bool(int(input("Пишем? 1- ДА, 0 - НЕТ:\n"))):
                mel_spec = create_mel_spec(signal)
                norm_mel_spec = normalize(mel_spec)
                save_spec(mel_spec, wo_norm_path + cls + f'/{i}.pkl')
                save_spec(norm_mel_spec, norm_path + cls + f'/{i}.pkl')
                save_spec(signal, raw_path + cls + f'/{i}.pkl')
            else:
                i -= 1

def fill_noise(FROM = 0, N = 100):
    norm_path = './data/norm/'
    wo_norm_path = './data/wo_normalize/'
    raw_path = './data/raw/'
    print('запись для класса:\t шум')
    for i in range(FROM, FROM + N):
        #input('enter for record')
        signal = get_signal()
        mel_spec = create_mel_spec(signal)
        norm_mel_spec = normalize(mel_spec)
        save_spec(mel_spec, wo_norm_path + 'шум' + f'/{i}.pkl')
        save_spec(norm_mel_spec, norm_path + "шум" + f'/{i}.pkl')
        save_spec(signal, raw_path + "шум" + f'/{i}.pkl')

def augmentation_data():
    norm_path = './data/norm/'
    wo_norm_path = './data/wo_normalize/'
    raw_path = './data/raw/'
    paths = [norm_path, wo_norm_path]
    classes = ['алгоритмы', 'дата', 'время']
    f = open(norm_path + 'алгоритмы/1.pkl', 'rb')
    audio = pickle.load(f)
    f.close()
    noise = np.random.normal(0, 0.06, audio.shape)

    for path in paths:
        for c in classes:
            for filename in os.listdir(path + c):
                filepath = os.path.join(path + c, filename)

                f = open(filepath,  'rb')
                audio = pickle.load(f)
                f.close()
                audio_noisy = normalize(audio + noise)
                f = open(filepath[:-4] + 'A.pkl',  'wb')
                pickle.dump(audio_noisy, f)
                f.close()



if __name__ ==  '__main__':
    #print(normalize(create_mel_spec(get_signal())).shape)
    #fill(FROM=400, N = 100)
    #fill_noise(FROM=300)
    augmentation_data()


# 1 8 15 22 ... 

# signal = get_signal()
# fig, ax = plt.subplots()
# print(signal.shape)
# ax.plot(signal)
# plt.show()


# fft = librosa.stft(y = signal, n_fft=N_FFT, hop_length=HOP_LEN)
# power_spec = np.abs(fft) ** 2
# mel_spec = librosa.feature.melspectrogram(S=power_spec, sr =SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LEN, n_mels = N_MELS)
# mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
# mel_spec_db = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))

# fig, ax = plt.subplots(figsize = (12, 8))
















