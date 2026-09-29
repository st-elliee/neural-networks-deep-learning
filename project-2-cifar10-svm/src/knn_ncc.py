# src/knn_ncc.py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time


# 1. kNN with scikit-learn

def knn_sklearn(x_train, y_train, x_test, k=1):
    model = KNeighborsClassifier(n_neighbors=k)
    start = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - start

    start = time.time()
    preds = model.predict(x_test)
    test_time = time.time() - start

    return preds, train_time, test_time


# 2. Custom kNN implementation (brute force)

def knn_custom(x_train, y_train, x_test, k=1):
    """
    Simple kNN implementation:
    - Computes distances from each test sample to all training samples
    - Takes the k nearest neighbors
    - Predicts the most frequent class among them
    """

    start_test = time.time()
    preds = []

    for x in x_test:
        # distances: (num_train,)
        dists = np.linalg.norm(x_train - x, axis=1)

        # indices of the k smallest distances
        nn_idx = np.argsort(dists)[:k]

        # labels of the k nearest neighbors
        nn_labels = y_train[nn_idx]

        # most frequent class
        values, counts = np.unique(nn_labels, return_counts=True)
        pred = values[np.argmax(counts)]
        preds.append(pred)

    test_time = time.time() - start_test
    train_time = 0.0  # no training phase

    return np.array(preds), train_time, test_time



# 3. Nearest Class Centroid (NCC)

def nearest_class_centroid(x_train, y_train, x_test):
    """
    NCC:
    - Computes the centroid (mean vector) of each class
    - Assigns each test sample to the nearest centroid
    """

    classes = np.unique(y_train)
    centroids = {}

    # compute class centroids
    for c in classes:
        centroids[c] = x_train[y_train == c].mean(axis=0)

    # stack into a matrix for vectorized distance computation
    centroid_matrix = np.vstack([centroids[c] for c in classes])

    start = time.time()

    # distances: (num_test, num_classes)
    dists = np.linalg.norm(x_test[:, None, :] - centroid_matrix[None, :, :], axis=2)

    preds = classes[np.argmin(dists, axis=1)]
    test_time = time.time() - start

    return preds, 0.0, test_time
