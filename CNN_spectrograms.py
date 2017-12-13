from keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.models import Sequential
import pickle
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import numpy as np
import sys

def load(path):
    with open(path, 'rb+') as f:
        return pickle.load(f)

def load_data(n, languages, which = 'train') :
    path = "/farmshare/user_data/adeveau/CS229-Project/spectrograms_and_mfccs/{}/".format(which)
    data = {}
    for lang in languages:
        data[lang] = [x[1] for x in load(path+'{}_{}_features.pkl'.format(lang, which))]
    
   
    for lang in data:
        data[lang] = [x[:n,:] for x in data[lang] if x.shape[1] > n]
    
    m = min(len(data[lang]) for lang in data)
    for lang in data:
        data[lang] = data[lang][:m]

    for lang in data:
        data[lang] = np.dstack(data[lang])
        data[lang] = np.swapaxes(data[lang], 0,2)
        data[lang] = np.reshape(data[lang], (m, n, 13, 1))
    y = []
    for i,lang in enumerate(data):
        try:
            X = np.concatenate((X,data[lang]), axis = 0)
        except UnboundLocalError:
            X = data[lang]
        y.extend([i]*m)

    y = np.array(y)
    y = to_categorical(y)
    return X, y

def model_constr(n, n_classes, n_neurons, filter_shape):
    model = Sequential()
    for n_neur, fs in zip(n_neurons, filter_shape):
        model.add(Conv2D(n_neur, fs,  input_shape = (n, 13, 1), kernel_initializer='he_normal', activation='tanh'))
    model.add(Dropout(rate = .1))
    model.add(Flatten())
    model.add(Dense(units = n_classes, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
    return model


if __name__ == '__main__':
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 300
    if len(sys.argv) > 2:
        epochs = int(sys.argv[2])
    else:
        epochs = 5

    if len(sys.argv) > 3:
        langs = sys.argv[3:]
    else:
        langs = ['english', 'french', 'vietnam','japanese', 'mandarin', 'spanish', 'german', 'korean', 'farsi', 'tamil']

    X, y = load_data(n, languages = langs)
    print(X.shape, y.shape)
    model = model_constr(n, len(langs), [5], [(10, 1)])
    model.fit(X, y, epochs = epochs, batch_size = 64)
    X_dev, y_dev = load_data(n, languages = langs, which = 'dev')
    print("\n Train error: {}".format(model.evaluate(X,y)))
    print("\n Dev error: {}".format(model.evaluate(X_dev, y_dev)))

