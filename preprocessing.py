import scipy.signal as signal
from scipy.io import wavfile
import glob
import numpy as np
import os
from pathos.multiprocessing import ProcessPool
import sys
from python_speech_features import mfcc
from python_speech_features import logfbank


def spectrogram(pathname):
    samples = wavfile.read(pathname)
    t, f, Sxx = signal.spectrogram(samples[1])
    return np.ndarray.flatten(np.transpose(Sxx))

def mfcc(pathnam):
    samples = wavfile.read(pathname)
    mfcc_feat = mfcc(samples[1],samples[0], winlen=0.025, winstep=0.01)
    return mfcc_feat
    
def build_data(language):
    p = ProcessPool(5)
    data = p.map(parse, glob.glob("./{}/**/*.wav".format(language), recursive = True)) 
    X = np.vstack([spect for spect in data if len(spect) == 126936])
    return X

def parse(path):
    lang = os.path.normpath(path).split(os.sep)[0]
    x = spectrogram(path)
    return x

if __name__ == "__main__":
    lang = sys.argv[1]
    X = build_data(lang)
    print(X.shape)
    np.save("{}_features.npy".format(lang), X)

