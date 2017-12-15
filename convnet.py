from keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.models import Sequential
import pickle
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import numpy as np
import sys
import random
import itertools
from keras.regularizers import l2


def load(path):
    with open(path, 'rb+') as f:
        return pickle.load(f)

def load_data(n, languages, which = 'train') :
    path = "/farmshare/user_data/adeveau/CS229-Project/features/{}/ungrouped/".format(which)
    data = {}
    key = {}
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
        key[lang] = i
    y = np.array(y)
    return X, y, to_categorical(y), key

def merged_data(n, l1 = ['english', 'german'], l2 = ['mandarin'], which = 'train'):
    X, y, cat_y, key = load_data(n, languages = l1 + l2, which = which)
    l1_idx, l2_idx = [],[]
    for l in l1:
        l1_idx.append(key[l])
    for l in l2:
        l2_idx.append(key[l])

    X_l1, X_l2 = X[np.isin(y, l1_idx),:], X[np.isin(y, l2_idx),:]

    if X_l1.shape[0] > X_l2.shape[0]:
        X_l1 = X_l1[np.random.choice(range(X_l1.shape[0]), size = X_l2.shape[0]), :] 
    else:
        X_l2 = X_l2[np.random.choice(range(X_l2.shape[0]), size = X_l1.shape[0]), :]

    y = np.array([0]*X_l2.shape[0] + [1]*X_l2.shape[0])
    y = to_categorical(y)

    return np.concatenate((X_l1, X_l2), axis = 0), y, {tuple(l1):0, tuple(l2):1}

def model_constr(n, n_classes, n_neurons, filter_shape, dropout_rate, Lambda):
    model = Sequential()
    for n_neur, fs in zip(n_neurons, filter_shape):
        model.add(Conv2D(n_neur, fs,  input_shape = (n, 13, 1), kernel_initializer='he_normal', activation='tanh', kernel_regularizer = l2(Lambda)))
    model.add(Dropout(rate = dropout_rate))
    model.add(Flatten())
    model.add(Dense(units = n_classes, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
    return model

def cross_validate(shapes, lambdas, dropout_rates):
    X_train, y_train, cat_y_train, _ = load_data(300, languages = ['english', 'mandarin'], which = 'train')
    X_dev, y_dev, cat_y_dev, _ = load_data(300, languages = ['english', 'mandarin'], which = 'dev')
    dev_accr = []
    for shape, Lambda, do_rate in itertools.product(shapes, lambdas, dropout_rates):
        cur_model = model_constr(300, 2, [16,8], shape, do_rate, Lambda)
        cur_model.fit(X_train, cat_y_train, verbose = 0, epochs = 5, batch_size = 32)
        accr = cur_model.evaluate(X_dev, cat_y_dev, verbose = 0)
        tag = "_".join([str(s) for s in (shape, Lambda, do_rate)])
        dev_accr.append(tag + " " + str(accr[1]))
        cur_model.save("saved_models/c_net_{}_{}.mdl".format("_".join(x[:2] for x in langs), tag))

    with open('dev_accr.txt', 'wt') as f:
        f.write("\n".join(dev_accr))

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
 
    shapes = [[(x,1), (x,1)] for x in [5, 10, 20, 30]]
    Lambdas = [.001, .01, .1, 1]
    do_rates = [.01, .2, .3, .5]

    #cross_validate(shapes, Lambdas, do_rates)   
    X_train, y_train, cat_y_train, key = load_data(n, languages = langs)
    print(key)
    model = model_constr(n, len(langs), [16,8], [(5, 1), (5,1)], 1, .3)
    model.fit(X_train, cat_y_train, epochs = epochs, batch_size = 10)
    X_dev, y_dev, cat_y_dev, _ = load_data(n, languages = langs, which = 'dev')
    print("\n Train error: {}".format(model.evaluate(X_train,cat_y_train)))
    print("\n Dev error: {}".format(model.evaluate(X_dev, cat_y_dev)))
    #X_test, y_test = load_data(n, languages = langs, which = 'test')
    #print("\n Test error: {}".format(model.evaluate(X_test, y_test)))
    model.save("saved_models/c_net_full.mdl")
    
