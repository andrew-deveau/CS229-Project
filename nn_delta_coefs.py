
import numpy as np
import os
import time

#       LOAD DATA
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')

dev_data = np.load('dev_data.npy')
dev_labels = np.load('dev_labels.npy')

#       SHUFFLE DATA
train_combine = np.concatenate([train_labels.reshape((train_labels.shape[0],1)),
                                train_data], axis=1)

np.random.shuffle(train_combine)
train_data = train_combine[:,1:]
train_labels = train_combine[:,0].reshape((train_labels.shape[0],))

#==============================================================================

#       TRAIN NEURAL NETWORK

from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

dimension = train_data.shape[1]  # should be 104 ( = 13*8)

model = Sequential()
model.add(Dense(units=200, activation='relu', input_dim=dimension, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(units=100, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(units=50, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

start = time.time()
model.fit(train_data, train_labels, epochs=15, batch_size=500)
end = time.time() - start
print('\nTook {:.2f} seconds = {:.2f} minutes to train network'.format(end, end/60.0))

#       CALCULATE DEV SET ERROR
predictions = model.predict(dev_data).reshape((dev_data.shape[0],))
predictions[predictions < 0.5] = 0
predictions[predictions > 0.5] = 1
results = (predictions == dev_labels).mean()
print('\nDev set percentage of correct predictions: {}'.format(results))


#==============================================================================

#       PREDICT PHONE CALLS

os.chdir('/Users/justinpyron/Google Drive/Stanford/Fall 2017/CS 229/Project/features/')

test_calls = np.load('test_data.npy')
test_labels = np.load('test_labels.npy')
call_results = list()

# Here, prediction is performed on each mfcc vector inside of a phone call. 
# The prediction vector output from Keras is then converted from a probability 
# to either 0 or 1. To compute a prediction for the entire call, calculate 
# the percentage of 1 predictions vs 0 predictions. Whichever one is greatest
# is what is predicted.

start = time.time()
for i in range(len(test_calls)):
    #print('Phone call #{}, data shape: {}, label: {}'.format(i, test_calls[i].shape, test_labels[i]))
    phone_call = test_calls[i]
    pred = model.predict(phone_call).reshape((phone_call.shape[0],))
    pred[pred < 0.5] = 0
    pred[pred > 0.5] = 1
    call_prediction = 0 if pred.sum()/pred.shape[0] < 0.5 else 1
    output = 1 if (call_prediction == test_labels[i]) else 0  # majority vote
    call_results.append(output)
end = time.time() - start
print('\nTook {:.2f} seconds = {:.2f} minutes to calculate call-level predictions'.format(end, end/60.0))
call_results = np.array(call_results)
pct_correct = call_results.sum()/call_results.shape[0]  # percentage of phone calls correctly predicted
print('\nPercentage of calls correctly predicted: {}'.format(pct_correct))









