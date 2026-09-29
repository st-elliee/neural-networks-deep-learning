# baselines.py
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, confusion_matrix

def baseline_nn(Z_train, y_train, Z_test, y_test, k=1):
    t0 = time.time()
    clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    clf.fit(Z_train, y_train)
    y_pred = clf.predict(Z_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return {"acc": acc, "cm": cm, "time": time.time() - t0}

def baseline_ncc(Z_train, y_train, Z_test, y_test):
    t0 = time.time()
    clf = NearestCentroid(metric="euclidean")
    clf.fit(Z_train, y_train)
    y_pred = clf.predict(Z_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return {"acc": acc, "cm": cm, "time": time.time() - t0}
