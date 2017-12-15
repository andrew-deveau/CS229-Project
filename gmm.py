
import numpy as np
import pandas as pd
import os
import time
import pickle
from sklearn.mixture import GaussianMixture
import datetime

#os.chdir('/Users/justinpyron/Google Drive/Stanford/Fall 2017/CS 229/Project/features')

#==============================================================================

def output_results(output_data):
    os.chdir('/farmshare/user_data/pyron/CS229/')
    data = list()
#    data.append(['Results where features contain delta-delta cepstra as well', '', '', '', '', '', ''])
    data.append(['Time', 'Num Eng. clusters', 'Num Chin. clusters', 'GMM train size', 'Train Accuracy', 'Test Accuracy', 'Dev Accuracy'])
    data = data + output_data
    time = datetime.datetime.now().strftime('%m-%d-%H_%M')
    output_file = 'GMM_tuning_{}.txt'.format(time)
    with open(output_file, 'w') as f:
        for row in data:
            try:
                f.write('{:.2f}  {:>3}                 {:>3}                  {:.0f}            {:.4f}           {:.4f}          {:.4f} \n'.format(
                        row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
            except:
                f.write('{}   {}   {}   {}   {}   {}   {}\n'.format(row[0], row[1], row[2], row[3], row[4], row[5], row[6]))

#==============================================================================

def load(file_name):
  with open(file_name, 'rb') as f:
    return pickle.load(f)


# The method below constructs features that only contain first differences
# i.e. contain only delta cepstra and NOT delta-delta cepstra

def get_features(data, P=3, K=7, demean=True):
    #    Input data: a numpy array of mfcc coefficient vector time series,
    #        where each row is a 13 entry vector corresponding to a single frame
    #    Input P: an integer representing the step between differences
    #    Input K: an integer representig the number of differences to take
    #    Returns a numpy array of feature vectors, i.e. an array where each row
    #        contains de-meaned mfccs, and delta coefficients
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



# The method below constructs features with first AND second differences as features
# i.e. has both delta cepstra AND delta-delta cepstra
'''
def get_features(data, P=3, K=7, demean=True):
#    Input data: a numpy array of mfcc coefficient vector time series,
#        where each row is a 13 entry vector corresponding to a single frame
#    Input P: an integer representing the step between differences
#    Input K: an integer representig the number of differences to take
#    Returns a numpy array of feature vectors, i.e. an array where each row
#        contains de-meaned mfccs, and delta coefficients
    if demean == True:
        data = data - data.mean(axis=0)

    features = list()
    
    # Single differences  (~ first derivative)
    downshift = pd.DataFrame(data).shift(1, axis=0).as_matrix()
    upshift = pd.DataFrame(data).shift(-1, axis=0).as_matrix()
    diffs = upshift - downshift

    # Second differences  (~ second derivative)
    downshift = pd.DataFrame(diffs).shift(1, axis=0).as_matrix()
    upshift = pd.DataFrame(diffs).shift(-1, axis=0).as_matrix()
    two_diffs = upshift - downshift

    for row in range(2, len(data)-2-K*P-1):  # adjust to account for double diffs
        new_array = data[row] # add static mfccs 
        for i in range(K):
            new_array = np.concatenate([new_array, 
                                        diffs[row + P*i], 
                                        two_diffs[row + P*i] ])
        features.append(new_array)

    return np.array(features)

'''



#===============================================
#     CREATE  TRAIN  SET
#===============================================

os.chdir('/farmshare/user_data/adeveau/CS229-Project/features/')
#os.chdir('/Users/justinpyron/Google Drive/Stanford/Fall 2017/CS 229/Project/features')

min_len_thresh = 300   # only construct features for phone calls of this length or longer

english_train_data = list()
chinese_train_data = list()

label_map = {'english': 0.0, 'mandarin':1.0}

train_labels = list()
train_call_set = list()   # this is used to do call-level prediction on training set

print('Constructing train set...')
start = time.time()
path_root = 'train/ungrouped/'
for lang in ['english','mandarin']:
    for file_name in os.listdir(path_root):
        if lang in file_name:
            array_list = load(path_root + file_name)

            for phone_call in array_list:
                if phone_call.shape[0] < min_len_thresh:
                    continue  # don't take features from short phone calls
                features = get_features(phone_call)  # create new features

                if 'english' in file_name:
                    english_train_data.append(features)
                    train_labels.append(label_map['english'])
                elif 'mandarin' in file_name:
                    chinese_train_data.append(features)
                    train_labels.append(label_map['mandarin'])
                else:
                    pass
                train_call_set.append(features)


english_train_data = np.concatenate(english_train_data, axis=0)
chinese_train_data = np.concatenate(chinese_train_data, axis=0)
end = time.time()-start
print('Training set construction took {:.2f} seconds = {:.2f} minutes'.format(end, end/60.0))

np.random.shuffle(english_train_data)
np.random.shuffle(chinese_train_data)

#===============================================
#     CREATE  TEST  SET
#===============================================

test_labels = list()
test_data = list()

print('Constructing test set...')
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


#===============================================
#     CREATE  DEV  SET
#===============================================

dev_labels = list()
dev_data = list()

print('Constructing dev set...')
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
                dev_labels.append( label_map[lang] )

end = time.time()-start
print('Dev set construction took {:.2f} seconds = {:.2f} minutes'.format(end, end/60.0))

#===============================================
#     TUNE GMM PARAMETERS
#===============================================

english_clusters = [30, 40, 50, 60]
chinese_clusters = [30, 40, 50, 60]
#english_clusters = [2]
#chinese_clusters = [2]
num_examples = 100000

results_list = list()

for english_num_cluster in english_clusters:
    for chinese_num_cluster in chinese_clusters:
        print('Current loop: {} english clusters, {} chinese clusters'.format(english_num_cluster, chinese_num_cluster))
        start = time.time()

        english_gmm = GaussianMixture(n_components=english_num_cluster)
        start = time.time()
        english_gmm.fit(english_train_data[:num_examples])
        
        # Fit chinese parameters
        chinese_gmm = GaussianMixture(n_components=chinese_num_cluster)
        start = time.time()
        chinese_gmm.fit(chinese_train_data[:num_examples])  # chinese only has a little over 490,000 examples

        # Get train set accuracy
        train_results = list()
        for i in range(len(train_call_set)):
            loop_array = train_call_set[i]
            loop_label = train_labels[i]
            english_probs = english_gmm.score_samples(loop_array)
            chinese_probs = chinese_gmm.score_samples(loop_array)
#            most_likely_lang = 0.5 * np.ones(loop_array.shape[0])
#            most_likely_lang[english_probs > chinese_probs] = label_map['english']
#            most_likely_lang[english_probs < chinese_probs] = label_map['mandarin']
#            pred = 0 if most_likely_lang.sum()/most_likely_lang.shape[0] < 0.5 else 1

            pred = 0 if english_probs.sum() > chinese_probs.sum() else 1
            if (pred == loop_label):
                train_results.append(1)
            else:
                train_results.append(0)
        train_results = np.array(train_results)
        train_pct_correct = train_results.sum()/train_results.shape[0]

        # Get test set accuracy
        test_results = list()        
        for i in range(len(test_data)):
            loop_array = test_data[i]
            loop_label = test_labels[i]
            english_probs = english_gmm.score_samples(loop_array)
            chinese_probs = chinese_gmm.score_samples(loop_array)
#            most_likely_lang = 0.5 * np.ones(loop_array.shape[0])
#            most_likely_lang[english_probs > chinese_probs] = label_map['english']
#            most_likely_lang[english_probs < chinese_probs] = label_map['mandarin']
#            pred = 0 if most_likely_lang.sum()/most_likely_lang.shape[0] < 0.5 else 1
            pred = 0 if english_probs.sum() > chinese_probs.sum() else 1
            if (pred == loop_label):
                test_results.append(1)
            else:
                test_results.append(0)            
        test_results = np.array(test_results)
        test_pct_correct = test_results.sum()/test_results.shape[0]

        # Get dev set accuracy
        dev_results = list()        
        for i in range(len(dev_data)):
            loop_array = dev_data[i]
            loop_label = dev_labels[i]
            english_probs = english_gmm.score_samples(loop_array)
            chinese_probs = chinese_gmm.score_samples(loop_array)
#            most_likely_lang = 0.5 * np.ones(loop_array.shape[0])
#            most_likely_lang[english_probs > chinese_probs] = label_map['english']
#            most_likely_lang[english_probs < chinese_probs] = label_map['mandarin']
#            pred = 0 if most_likely_lang.sum()/most_likely_lang.shape[0] < 0.5 else 1
            pred = 0 if english_probs.sum() > chinese_probs.sum() else 1
            if (pred == loop_label):
                dev_results.append(1)
            else:
                dev_results.append(0)            
        dev_results = np.array(dev_results)
        dev_pct_correct = dev_results.sum()/dev_results.shape[0]


        duration = (time.time()-start)/60.0
        
        # Save results
        intermediate_results_list = [duration, english_num_cluster, chinese_num_cluster,
                                     num_examples, train_pct_correct, test_pct_correct, dev_pct_correct]
        results_list.append(intermediate_results_list)

output_results(results_list)


print('\nNumber of train set calls: {}'.format(len(train_call_set)))
print('\nNumber of dev set calls: {}'.format(len(dev_data)))
print('\nNumber of test set calls: {}'.format(len(test_data)))

