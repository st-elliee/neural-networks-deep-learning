# data.py
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

def load_mnist(flatten=True, batch_size=4096, limit=None, seed=42):
    np.random.seed(seed)
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    X_train = train_set.data.numpy().astype(np.float32) / 255.0
    y_train = train_set.targets.numpy().astype(np.int64)
    X_test  = test_set.data.numpy().astype(np.float32) / 255.0
    y_test  = test_set.targets.numpy().astype(np.int64)

    if flatten:
        X_train = X_train.reshape(len(X_train), -1)
        X_test  = X_test.reshape(len(X_test), -1)

    if limit is not None:
        idx_train = np.random.permutation(len(X_train))[:limit]
        idx_test  = np.random.permutation(len(X_test))[:min(limit//6, len(X_test))]
        X_train, y_train = X_train[idx_train], y_train[idx_train]
        X_test, y_test   = X_test[idx_test], y_test[idx_test]

    return (X_train, y_train), (X_test, y_test)

def fit_pca(X_train, variance=0.95, svd_solver="full", random_state=42):
    pca = PCA(n_components=variance, svd_solver=svd_solver, random_state=random_state)
    pca.fit(X_train)
    return pca

def apply_pca(pca, X_train, X_test):
    Z_train = pca.transform(X_train)
    Z_test  = pca.transform(X_test)
    return Z_train, Z_test
