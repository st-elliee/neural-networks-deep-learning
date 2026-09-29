# src/features.py
import numpy as np
from sklearn.decomposition import PCA

def apply_pca(x_train, x_test, variance_threshold=0.90):
    pca = PCA(variance_threshold)
    pca.fit(x_train)

    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    print("Original dim:", x_train.shape[1])
    print("Reduced dim:", x_train_pca.shape[1])

    return x_train_pca, x_test_pca, pca
