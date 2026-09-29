# rbf.py
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix

def gaussian_rbf(dist_sq, sigma):
    return np.exp(-dist_sq / (2.0 * (sigma ** 2) + 1e-12))

def pairwise_sq_dists(X, C):
    # X: [N, D], C: [K, D]
    # returns [N, K] squared distances
    X2 = np.sum(X**2, axis=1, keepdims=True)     # [N, 1]
    C2 = np.sum(C**2, axis=1, keepdims=True).T   # [1, K]
    XC = X @ C.T                                 # [N, K]
    return X2 + C2 - 2*XC

def choose_centers(Z_train, y_train, K, mode="kmeans", seed=42):
    np.random.seed(seed)
    if mode == "kmeans":
        km = KMeans(n_clusters=K, random_state=seed, n_init=10)
        km.fit(Z_train)
        C = km.cluster_centers_
    elif mode == "random":
        idx = np.random.permutation(len(Z_train))[:K]
        C = Z_train[idx]
    else:
        raise ValueError("Unknown centers mode")
    return C

def estimate_sigma(C, Z_train, mode="median", k_neighbors=10):
    if mode == "median":
        D = pairwise_sq_dists(C, C)
        # guard against small negative values from floating-point error
        dists = np.sqrt(np.maximum(D, 0.0))
        Kc = len(C)
        up = []
        for i in range(Kc):
            for j in range(i+1, Kc):
                up.append(dists[i, j])
        # fallback if there are no center pairs (e.g. K=1)
        if len(up) == 0:
            sigma = 1.0
        else:
            # no division by sqrt(2K), to avoid an overly small sigma
            base = np.median(up)
            # lower bound for numerical safety
            sigma = max(base, 1e-3)
        return sigma, None
    elif mode == "per_center":
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(Z_train)), metric="euclidean")
        nbrs.fit(Z_train)
        distances, _ = nbrs.kneighbors(C)
        sigma_i = np.mean(distances, axis=1)
        # clip to avoid very small / zero widths
        sigma_i = np.clip(sigma_i, 1e-3, None)
        return None, sigma_i
    else:
        raise ValueError("Unknown sigma mode")

def design_matrix(Z, C, sigma=None, sigma_i=None):
    D2 = pairwise_sq_dists(Z, C)  # [N, K]
    if sigma_i is None:
        Phi = gaussian_rbf(D2, sigma)
    else:
        # broadcast each column with its sigma_i
        Phi = np.exp(-D2 / (2.0 * (sigma_i[np.newaxis, :]**2) + 1e-12))
    # Add bias column
    Phi = np.concatenate([Phi, np.ones((Phi.shape[0], 1))], axis=1)
    return Phi  # [N, K+1]

def one_hot(y, num_classes=10):
    Y = np.zeros((len(y), num_classes), dtype=np.float64)
    Y[np.arange(len(y)), y] = 1.0
    return Y

def train_rbf_classifier(Z_train, y_train, Z_val, y_val, K, centers_mode="kmeans",
                         sigma_mode="median", ridge=1e-3, seed=42):
    t0 = time.time()
    C = choose_centers(Z_train, y_train, K=K, mode=centers_mode, seed=seed)
    sigma, sigma_i = estimate_sigma(C, Z_train, mode=sigma_mode)

    Phi_train = design_matrix(Z_train, C, sigma=sigma, sigma_i=sigma_i)
    Phi_val   = design_matrix(Z_val,   C, sigma=sigma, sigma_i=sigma_i)

    Y_train = one_hot(y_train, num_classes=len(np.unique(y_train)))

    # Ridge regression closed-form: W = (Phi^T Phi + λI)^(-1) Phi^T Y
    I = np.eye(Phi_train.shape[1])
    W = np.linalg.solve(Phi_train.T @ Phi_train + ridge * I, Phi_train.T @ Y_train)

    # Predictions
    logits_val = Phi_val @ W
    y_pred_val = np.argmax(logits_val, axis=1)
    acc_val = accuracy_score(y_val, y_pred_val)
    cm_val = confusion_matrix(y_val, y_pred_val)

    return {
        "centers": C,
        "sigma": sigma,
        "sigma_i": sigma_i,
        "W": W,
        "time_train": time.time() - t0,
        "val_acc": acc_val,
        "val_cm": cm_val,
    }

def predict_rbf(Z, model):
    C = model["centers"]
    Phi = design_matrix(Z, C, sigma=model["sigma"], sigma_i=model["sigma_i"])
    logits = Phi @ model["W"]
    y_pred = np.argmax(logits, axis=1)
    return y_pred
