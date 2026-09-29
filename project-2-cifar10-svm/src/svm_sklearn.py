# src/svm_sklearn.py
from sklearn.svm import SVC, LinearSVC
import time

def svm_linear(x_train, y_train, x_test, C=1.0):
    model = LinearSVC(C=C)
    start = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - start

    start = time.time()
    preds = model.predict(x_test)
    test_time = time.time() - start

    return preds, train_time, test_time

def svm_rbf(x_train, y_train, x_test, C=1.0, gamma='scale'):
    model = SVC(kernel='rbf', C=C, gamma=gamma)
    start = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - start

    start = time.time()
    preds = model.predict(x_test)
    test_time = time.time() - start

    return preds, train_time, test_time

def svm_poly(x_train, y_train, x_test, C=1.0, degree=3):
    model = SVC(kernel='poly', C=C, degree=degree)
    start = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - start

    start = time.time()
    preds = model.predict(x_test)
    test_time = time.time() - start

    return preds, train_time, test_time
