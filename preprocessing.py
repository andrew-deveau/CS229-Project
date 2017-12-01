import scipy.signal as signal
from scipy.io import wavfile
import glob
import numpy as np
import os
from pathos.multiprocessing import ProcessPool
import sys
from python_speech_features.base import mfcc

def calc_mfcc(pathname):
    samprate, samples = wavfile.read(pathname)
    return mfcc(samples, samplerate = samprate, appendEnergy = False)

def spectrogram(pathname):
    samples = wavfile.read(pathname)
    t, f, Sxx = signal.spectrogram(samples[1])
    return np.ndarray.flatten(np.transpose(Sxx))

def build_data(language, train_fraction = .6, dev_fraction = .2):
    pth = "/farmshare/user_data/adeveau/calls"
    p = ProcessPool(5)
    data = p.map(parse, glob.glob(pth + "/{}/**/*.wav".format(language), recursive = True))
    split_pt1 = int(len(data)*train_fraction)
    split_pt2 = int(split_pt1 + len(data)*dev_fraction)
    X_train, X_dev, X_test = np.vstack(data[:split_pt1]), np.vstack(data[split_pt1:split_pt2]), np.vstack(data[split_pt2:])
    return X_train, X_dev, X_test

def parse(path):
    lang = os.path.normpath(path).split(os.sep)[0]
    x = calc_mfcc(path)
    return x

if __name__ == "__main__":
    for lang in ['english', 'farsi', 'french', 'german', 'hindi', 'japanese', 'korean', 'mandarin', 'spanish', 'tamil', 'vietnam']:
        print(lang)
        X_train, X_dev, X_test = build_data(lang)
        np.save("{}_train_features.npy".format(lang), X_train)
        np.save("{}_test_features.npy".format(lang), X_test)
        np.save("{}_dev_features.npy".format(lang), X_dev)
