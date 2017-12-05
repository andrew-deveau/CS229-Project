

import numpy as np
import scipy
import matplotlib.pyplot as plt
from python_speech_features.base import mfcc
import librosa
from sklearn.decomposition import PCA
import glob

def get_mfcc(sample_rate, signal):
    '''
    Returns Mel Frequency Cepstral Coefficients
    Provides information about sinusoids that constitute sound wave,
    adjusted to account for the way human's perceive sound
    '''
    mfccs = mfcc(signal, samplerate=sample_rate, appendEnergy=False)
    mfcc_cov = np.cov(mfccs.T)
    dim = mfcc_cov.shape[0]

    # Get means
    mean = mfccs.mean(axis=0)

    # Get variances (i.e. diagonal of covariance matrix)
    var_mask = np.nonzero(np.eye(dim))
    var = mfcc_cov[var_mask]

    # Get off-diagonal covariances
    cov_mask = np.nonzero(np.tri(dim) - np.eye(dim))
    cov = mfcc_cov[cov_mask]

    # NOTE: librosa also provides an MFCC function, but I believe it
    # requires passing as input some complicated information
    return mean, var, cov


def get_spec_centroid(sample_rate, signal):
    '''
    Measure of "center of gravity" of frequencies,
    i.e. amplitude-weighted average of frequencies
    Returns:
        Mean of spectral centroid time series
        Standard deviation of spectral centroid time series
    '''
    centroid = librosa.feature.spectral_centroid(signal, sr=sample_rate)[0,:]
    return np.array([centroid.mean()]), np.array([np.std(centroid)])


def get_spec_contrast(sample_rate, signal):
    '''
    Meaure of the difference between peaks and troughs of wave
    Returns:
    Mean of spectral contrast time series
    Standard deviation of spectral time series
    '''
    contrast = librosa.feature.spectral_contrast(signal, sr=sample_rate)
    return np.array([contrast.mean()]), np.array([np.std(contrast)])


def get_mfcc_pca(sample_rate, signal, num_components):
    '''
    Returns the N largest principal components of input multivariate time series
    Required input format: each time series arranged in a column vector
    '''
    mfccs = mfcc(signal, samplerate=sample_rate, appendEnergy=False)
    pca = PCA(n_components = num_components)
    pca.fit(mfccs)
    components = pca.components_  # each row is a component
    return components.flatten()



'''
This script provides methods of constructing feature sets.
Here, I assume that our dataset is organized as following:

    --> Data Folder
        --> language folders (e.g. English, Spanish, German, ...)
            --> raw audio signal (numpy array) of 10 second audio clips
'''


# Keys in this dictionary should be language_folder names
label_dict = {'english': 0,
          'french': 1,
          'spanish': 2,
          'german': 3,
          'tamil':4,
          'vietnam':5,
          'korean':6,
          'farsi':7,
          'mandarin':8,
          'japanese':9,
          }
labels = list()


# Store each feature set in a list
mfcc_mean_list = list()
mfcc_var_list = list()
mfcc_cov_list = list()  # off diagonal entries in correlation matrix

centroid_mean_list = list()
centroid_std_list = list()

#contrast_mean_list = list()
contrast_std_list = list()

pca_list = list()



n_sec = 10
# NOTE: THIS ISN'T FUNCTIONING CODE, JUST AN OUTLINE
for language_folder in label_dict:  # i.e. English, French, Spanish, ...

    for clip_file in glob.glob("{}/*.wav".format(language_folder)):

        # Convert .wav into raw audio signal
        sample_rate, signal = scipy.io.wavfile.read(clip_file)

        for idx in range(0, len(signal), n_sec*sample_rate):
            # Generate feature vectors for each 10 second clip
            chunk = signal[idx:idx+n_sec*sample_rate]
            mfcc_mean, mfcc_var, mfcc_cov = get_mfcc(sample_rate, chunk)
            centroid_mean, centroid_std = get_spec_centroid(sample_rate, chunk)
            #contrast_mean, contrast_std = get_spec_contrast(sample_rate, chunk)
            pca = get_mfcc_pca(sample_rate, signal, num_components=4)

            # Append each feature set list
            mfcc_mean_list.append(mfcc_mean)
            mfcc_var_list.append(mfcc_var)
            mfcc_cov_list.append(mfcc_cov)

            centroid_mean_list.append(centroid_mean)
            centroid_std_list.append(centroid_std)

            #contrast_mean_list.append(contrast_mean)
            #contrast_std_list.append(contrast_std)

            pca_list.append(pca)

            # Append correct lable to label list
            labels.append(label_dict[language_folder])

np.save('../features/mfcc_means', np.array(mfcc_mean_list))
np.save('../features/mfcc_vars', np.array(mfcc_var_list))
np.save('../features/mfcc_covs', np.array(mfcc_cov_list))

np.save('../features/centroid_means', np.array(centroid_mean_list))
np.save('../features/centroid_stds', np.array(centroid_std_list))

#np.save('../features/contrast_means', np.array(contrast_mean_list))
#np.save('../features/contrast_stds', np.array(contrast_std_list))

np.save('../features/mfcc_principal_components', np.array(pca_list))
np.save('../features/labels', np.array(labels))


#==============================================================================
#           FUNCTIONS TO CREATE FEATURES
#==============================================================================
# Each function will take as input a numpy array and output a numpy array
# Input will be raw audio signal numpy array
# Output will be either 1x1 numpy array (e.g. the mean of something)
#     or output will be normal vector numpy array (e.g. covariance
#     entries or principal components)



