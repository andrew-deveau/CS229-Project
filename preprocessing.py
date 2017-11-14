import scipy.signal as signal
from scipy.io import wavfile
import glob
import numpy as np
import os
from multiprocessing import Pool

def spectrogram(pathname):
    samples = wavfile.read(pathname)
    t, f, Sxx = signal.spectrogram(samples[1])
    return np.ndarray.flatten(np.transpose(Sxx))

def build_data():
    languages_key = {'arabic':0, 'english_american':1, 'french':2, 'chinese':3,
                    'hebrew':4, 'korean':5, 'russian':6, 'spanish':7}

    data = p.map(glob.glob("./arabic/**/*.wav", recursive = True), parse)
    X = np.vstack([tup[0] for tup in data if len(tup[0]) == 126936])
    y = np.array(tup[1] for tup in data if len(tup[0]) == 126936)

    return X, y

def parse(path):

    lang = os.path.normpath(path).split(os.sep)[0]

    x = spectrogram(path)
    y = languages_key[lang]

    return x, y

if __name__ == "__main__":
    X, Y = build_data()
