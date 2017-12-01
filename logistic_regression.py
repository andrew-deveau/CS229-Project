import sklearn.linear_model
from load_data import load_data
import numpy as np
import sys
from sklearn.externals import joblib

def log_regression(X_train, y_train, X_test, y_test):
    model = sklearn.linear_model.LogisticRegression()
    model.fit(X_train, y_train)
    y_test = model.predict(X_test)
    return score(X_test, y_test)


if __name__ == "__main__":
    n = int(sys.argv[1])
    X_train, y_train = load_data()
    p = np.random.permutation(len(y_train))
    X_train, y_train = X_train[p,:], y_train[p]
    X_train, y_train = X_train[:n,:], y_train[:n]
    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(X_train, y_train)
    joblib.dump(clf, "logistic_model2.pkl")
    print(clf.score(X_train, y_train))
