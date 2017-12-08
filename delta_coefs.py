
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import time


# EXPLANATION
# Inside the directory
#       /Users/justinpyron/Google Drive/Stanford/Fall 2017/
#           CS 229/Project/features/train/ungrouped
# there is a pickle for each language. Each pickle contains a list of numpy 
# arrays, each corresponding to a single phone call. Each numpy array contains 
# the mfcc vector time series. 


def get_features(data, P=3, K=7, demean=True):
    '''
    Input data: a numpy array of mfcc coefficient vector time series,
        where each row is a 13 entry vector corresponding to a single frame
    Input P: an integer representing the step between differences
    Input K: an integer representig the number of differences to take
    Returns a numpy array of feature vectors, i.e. an array where each row
        contains de-meaned mfccs, and delta coefficients
    '''
    if demean == True:
        data = data - data.mean(axis=0)
    
    downshift = pd.DataFrame(data).shift(1, axis=0).as_matrix()
    upshift = pd.DataFrame(data).shift(-1, axis=0).as_matrix()
    diffs = 0.5*(upshift - downshift)
    
    features = list()
    for row in range(1, len(data)-2-K*P):
        new_array = data[row]
        for i in range(K):
            new_array = np.concatenate([new_array, diffs[row + P*i]])
        features.append(new_array)
    return np.array(features)


#===============================================
#     SET  PARAMETERS  FOR  DATA  CONSTRUCTION
#===============================================

os.chdir('/Users/justinpyron/Google Drive/Stanford/Fall 2017/CS 229/Project/features/')

# Load files
def load(file_name):
  with open(file_name, 'rb+') as f:
    return pickle.load(f)

# Insert the languages you'll use here, along with corresponding labels
label_map = {'english': 0.0,
             'mandarin': 1.0}
min_len_thresh = 300   # only construct features for phone calls of this length or longer


#===============================================
#     CREATE  TRAIN  SET
#===============================================

train_labels = list()
train_data = list()

start = time.time()
path_root = 'train/ungrouped/'
for lang in label_map.keys():
    for file_name in os.listdir(path_root):
        if lang in file_name:
            array_list = load(path_root + file_name)

            for phone_call in array_list:
                if phone_call.shape[0] < min_len_thresh:
                    continue  # don't take features from short phone calls
                features = get_features(phone_call)  # create new features
                train_data.append(features)
                train_labels.append( np.ones(features.shape[0]) * label_map[lang])
train_labels = np.concatenate(train_labels)
train_data = np.concatenate(train_data, axis=0)
end = time.time()-start
print('Training set construction took {:.2f} seconds = {:.2f} minutes'.format(end, end/60.0))
np.save('train_data', train_data)
np.save('train_labels', train_labels)

#===============================================
#     CREATE  DEV  SET
#===============================================
dev_labels = list()
dev_data = list()

start = time.time()
path_root = 'dev/ungrouped/'
for lang in label_map.keys():
    for file_name in os.listdir(path_root):
        if lang in file_name:
            array_list = load(path_root + file_name)

            for phone_call in array_list:
                if phone_call.shape[0] < min_len_thresh:
                    continue  # don't take features from short phone calls
                features = get_features(phone_call)  # create new features
                dev_data.append(features)
                dev_labels.append( np.ones(features.shape[0]) * label_map[lang])
dev_labels = np.concatenate(dev_labels)
dev_data = np.concatenate(dev_data, axis=0)
end = time.time()-start
print('Dev set construction took {:.2f} seconds = {:.2f} minutes'.format(end, end/60.0))
np.save('dev_data', dev_data)
np.save('dev_labels', dev_labels)



#===============================================
#     CREATE  TEST  SET
#===============================================

# NOTE SOME DIFFERENCES HERE
# Here, instead of combining all mfccs together into one giant numpy array,
# I keep the mfccs for each phone call separate. The entire feature array 
# will be used to make predictions on a call-level basis.

# So, inside of 'test_data.npy' will be a list of numpy arrays, one per phone
# call, and inside of 'test_labels.npy' will be a list of labels (scalars).

test_labels = list()
test_data = list()

start = time.time()
path_root = 'test/ungrouped/'
for lang in label_map.keys():
    for file_name in os.listdir(path_root):
        if lang in file_name:
            array_list = load(path_root + file_name)

            for phone_call in array_list:
                if phone_call.shape[0] < min_len_thresh:
                    continue  # don't take features from short phone calls
                features = get_features(phone_call)  # create new features
                test_data.append(features)
                test_labels.append( label_map[lang] )

end = time.time()-start
print('Test set construction took {:.2f} seconds = {:.2f} minutes'.format(end, end/60.0))
np.save('test_data', test_data)
np.save('test_labels', test_labels)








