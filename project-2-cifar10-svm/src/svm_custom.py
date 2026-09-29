# src/svm_custom.py
import numpy as np

class LinearSVM:
    def __init__(self, lr=1e-3, C=1.0, epochs=10):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.W = None
        self.b = None

    def fit(self, X, y):
        """
        X: (N, D)
        y: (N,) labels in {-1, +1}
        """
        N, D = X.shape
        self.W = np.zeros(D)
        self.b = 0.0

        for epoch in range(self.epochs):
            for i in range(N):
                xi = X[i]
                yi = y[i]

                condition = yi * (np.dot(self.W, xi) + self.b)

                if condition >= 1:
                    # only regularization
                    dW = self.W
                    db = 0
                else:
                    # hinge loss gradient
                    dW = self.W - self.C * yi * xi
                    db = -self.C * yi

                # SGD update
                self.W -= self.lr * dW
                self.b -= self.lr * db

    def predict(self, X):
        scores = X.dot(self.W) + self.b
        return np.where(scores >= 0, 1, -1)
    
class OneVsRestSVM:
    def __init__(self, lr=1e-3, C=1.0, epochs=10):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.classifiers = {}

    def fit(self, X, y):
        classes = np.unique(y)

        for c in classes:
            # binary labels: +1 for class c, -1 for others
            y_binary = np.where(y == c, 1, -1)

            svm = LinearSVM(lr=self.lr, C=self.C, epochs=self.epochs)
            svm.fit(X, y_binary)
            self.classifiers[c] = svm

    def predict(self, X):
        # compute scores for each classifier
        scores = []
        for c, svm in self.classifiers.items():
            score = X.dot(svm.W) + svm.b
            scores.append(score)

        scores = np.vstack(scores)  # shape: (num_classes, num_samples)
        preds = np.argmax(scores, axis=0)
        return preds
