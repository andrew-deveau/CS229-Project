from keras.layers import Dense, Flatten, Conv2D
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
    path = "/farmshare/user_data/adeveau/CS229-Project/features/{}/ungrouped/".format(which)
    data = {}
    for lang in languages:
        data[lang] = load(path+'{}_{}_features.pkl'.format(lang, which))
   
    for lang in data:
        data[lang] = [x[:n,:] for x in data[lang] if x.shape[0] > n]
    
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

def model_constr(n, n_classes):
    model = Sequential()
    model.add(Conv2D(5, (10,1),  input_shape = (300, 13, 1), kernel_initializer='he_normal', activation='sigmoid'))
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
        langs = sys.argv[2:]
    else:
        langs = ['english', 'french']
    X, y = load_data(n, languages = langs)
    print(X.shape, y.shape)
    model = model_constr(n, len(langs))
    model.fit(X, y, epochs = 5, batch_size = 10)
