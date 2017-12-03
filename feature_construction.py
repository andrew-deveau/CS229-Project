

import numpy as np
import scipy
import matplotlib.pyplot as plt
from python_speech_features.base import mfcc
import librosa
from sklearn.decomposition import PCA


'''
This script provides methods of constructing feature sets.
Here, I assume that our dataset is organized as following:
    
    --> Data Folder
        --> language folders (e.g. English, Spanish, German, ...)
            --> raw audio signal (numpy array) of 10 second audio clips
'''



# Keys in this dictionary should be language_folder names
label_dict = {'English': 0,
          'French': 1,
          'Spanish': 2,
          'German': 3}
labels = list()


# Store each feature set in a list
mfcc_mean_list = list()
mfcc_var_list = list()
mfcc_cov_list = list()  # off diagonal entries in correlation matrix

centroid_mean_list = list()
centroid_std_list = list()

contrast_mean_list = list()
contrast_std_list = list()

pca_list = list()


# NOTE: THIS ISN'T FUNCTIONING CODE, JUST AN OUTLINE
for language_folder in folders:  # i.e. English, French, Spanish, ...
    
    for clip_file in language_folder:
        
        # Convert .wav into raw audio signal
        sample_rate, signal = scipy.io.wavfile.read(clip_file)
        
        # Generate feature vectors for each 10 second clip
        mfcc_mean, mfcc_var, mfcc_cov = get_mfcc(signal)
        centroid_mean, centroid_std = get_spec_centroid(signal)
        contrast_mean, contrast_std = get_spec_contrast(signal)
        pca = get_mfcc_pca(signal, num_components=4)
        
        # Append each feature set list
        mfcc_mean_list.append(mfcc_mean)
        mfcc_var_list.append(mfcc_var)
        mfcc_cov_list.append(mfcc_cov)
        
        centroid_mean_list.append(centroid_mean)
        centroid_std_list.append(centroid_std)
        
        contrast_mean_list.append(contrast_mean)
        contrast_std_list.append(contrast_std)
        
        pca_list.append(pca)
        
        # Append correct lable to label list
        labels.append(label_dict[language_folder])



np.save('mfcc_means', np.array(mfcc_mean_list))
np.save('mfcc_vars', np.array(mfcc_var_list))
np.save('mfcc_covs', np.array(mfcc_cov_list))

np.save('centroid_means', np.array(centroid_mean_list))
np.save('centroid_stds', np.array(centroid_std_list))

np.save('contrast_means', np.array(contrast_mean_list))
np.save('contrast_stds', np.array(contrast_std_list))

np.save('mfcc_principal_components', np.array(pca_list))



#==============================================================================
#           FUNCTIONS TO CREATE FEATURES
#==============================================================================
# Each function will take as input a numpy array and output a numpy array
# Input will be raw audio signal numpy array
# Output will be either 1x1 numpy array (e.g. the mean of something)
#     or output will be normal vector numpy array (e.g. covariance 
#     entries or principal components)


def get_mfcc(signal):
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


def get_spec_centroid(signal):
    '''
    Measure of "center of gravity" of frequencies,
    i.e. amplitude-weighted average of frequencies
    Returns:
        Mean of spectral centroid time series
        Standard deviation of spectral centroid time series
    '''
    centroid = librosa.feature.spectral_centroid(signal)[0,:]
    return np.array([centroid.mean()]), np.array([np.std(centroid)])


def get_spec_contrast(signal):
    '''
    Meaure of the difference between peaks and troughs of wave
    Returns:
    Mean of spectral contrast time series
    Standard deviation of spectral time series
    '''
    contrast = librosa.feature.spectral_contrast(signal)
    return np.array([contrast.mean()]), np.array([np.std(contrast)])


def get_mfcc_pca(signal, num_components):
    '''
    Returns the N largest principal components of input multivariate time series
    Required input format: each time series arranged in a column vector
    '''
    mfccs = mfcc(signal, samplerate=sample_rate, appendEnergy=False)
    pca = PCA(n_components = num_components)
    pca.fit(mfccs)
    components = pca.components_  # each row is a component
    return components.flatten()


#a = np.random.random((50,4))
#pca = PCA(n_components=4)
#pca.fit(a)
#components = pca.components_
#new_cov = np.cov(components.dot(a.T))  # off diagonal entries should be zero
#
#pca = PCA(n_components=3)
#pca.fit(a)
#components2 = pca.components_









'''
for language_folder in folders:  # i.e. English, French, Spanish, ...
    
    for clip in language_folder:  # clip is a file name of an *entire* phone call

        sample_rate, signal = scipy.io.wavfile.read(clip)
        # break up signal into 10-second segments

        for ten_second_segment in signal:

            # Save the np array containing raw audio signal to the proper folder
            # To reiterate, this raw audio signal is the result of scipy.io.wavfile.read()
            np.save(something)

'''



root = '/Users/justinpyron/Google Drive/Stanford/Fall 2017/CS 229/Project/'  # mac
file_name = 'test_audio.wav'
sample_rate, signal = scipy.io.wavfile.read(root + file_name)
signal = signal[1000:21000]


#==============================================================================
#            TEST DIFFERENCES IN MFCC COMPUTATIONS
#==============================================================================

# python_speech_features MFCCs
mfccs_1 = mfcc(signal, samplerate=sample_rate, appendEnergy=False)

# librosa MFCCs
# the parameters here were taken from website example
S = librosa.feature.melspectrogram(y=signal, n_mels=128, fmax=8000)
mfccs_2 = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=13)

cov = np.cov(mfccs_1.T)
upper_tri = np.triu_indices(cov.shape[0])
mfcc_cov = np.cov(mfccs_1.T)[upper_tri] # Only take upper triangle




