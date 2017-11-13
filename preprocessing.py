import scipy.signal as signal
from scipy.io import wavfile
import glob
import numpy as np
import os


def spectrogram(pathname):
    samples = wavfile.read(pathname)
    t, f, Sxx = signal.spectrogram(samples[1])
    return np.ndarray.flatten(np.transpose(Sxx))

def build_data():
    languages_key = {'arabic':0, 'english_american':1, 'french':2, 'chinese':3,
                    'hebrew':4, 'korean':5, 'russian':6, 'spanish':7}

    X, Y = [], []
    for path in glob.glob("./arabic/**/*.wav", recursive = True):
        print(path)
        lang = os.path.normpath(path).split(os.sep)[0]

        x = spectrogram(path)
        y = languages_key[lang]

        X.append(x)
        Y.append(y)


    return np.array(X), np.array(Y)

if __name__ == "__main__":
    X, Y = build_data()
