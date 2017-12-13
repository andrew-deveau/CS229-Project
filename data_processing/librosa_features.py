import numpy as np
import scipy
import librosa


def librosa_features(path, n_mel_filters=22, normalize=True):
    '''
    path: path to .wav file
    n_mel_filters: integer which I believe represents the number of bins
        the cepstrum is cut up into when computing the mfcss...?
    normalize: if True, normalize by maximum amplitude
    '''
    sample_rate, signal = scipy.io.wavfile.read(path)
    if normalize:
        signal = signal/signal.max()
    S = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=n_mel_filters, fmax=5000)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate)
    ss = np.abs(librosa.stft(signal))
    contrast = librosa.feature.spectral_contrast(S=ss, sr=sample_rate, n_bands=5)
    return np.concatenate([mfcc, centroid, rolloff, contrast], axis=0).T



