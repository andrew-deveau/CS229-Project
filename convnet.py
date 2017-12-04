from keras.layers import Dense, Flatten, Conv2D
from keras.models import Sequential
import pickle
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import numpy as np


def load(path):
    with open(path, 'rb+') as f:
        return pickle.load(f)

def load_data(n):
    path = "/farmshare/user_data/adeveau/CS229-Project/features/train/ungrouped/"
    train = {}
    train['english'] = load(path+'english_train_features.pkl')
    train['french'] = load(path+'french_train_features.pkl')
   
    for lang in train:
        train[lang] = [x[:n,:] for x in train[lang] if x.shape[0] > n]
    
    m = len(train['french'])

    train['english'] = train['english'][:m]

    for lang in train:
        train[lang] = np.dstack(train[lang])
        train[lang] = np.swapaxes(train[lang], 0,2)
        train[lang] = np.reshape(train[lang], (m, n, 13, 1))

    X_train = np.concatenate((train['english'], train['french']), axis = 0)
    y_train = [0]*m+[1]*m
    y_train = np.array(y_train)
    y_train = to_categorical(y_train)
    return X_train, y_train

def model_constr(n):
    model = Sequential()
    model.add(Conv2D(5, (10,1),  input_shape = (300, 13, 1), kernel_initializer='he_normal', activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(units = 2, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
    return model


if __name__ == '__main__':
    X_train, y_train = load_data(300)
    print(X_train.shape, y_train.shape)
    model = model_constr(300)
    model.fit(X_train, y_train, epochs = 5, batch_size = 10)
