# src/mlp_numpy.py
import numpy as np

class MLP_Hinge:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3):
        self.lr = lr

        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        # X: (N, D)
        self.z1 = X.dot(self.W1) + self.b1      # (N, H)
        self.a1 = self.relu(self.z1)            # (N, H)
        self.scores = self.a1.dot(self.W2) + self.b2  # (N, C)
        return self.scores

    def hinge_loss(self, scores, y):
        """
        scores: (N, C)
        y: (N,)
        """
        N = scores.shape[0]
        correct_scores = scores[np.arange(N), y].reshape(-1, 1)

        margins = np.maximum(0, scores - correct_scores + 1)
        margins[np.arange(N), y] = 0  # no loss for correct class

        loss = np.mean(np.sum(margins, axis=1))
        return loss, margins

    def backward(self, X, y, margins):
        N = X.shape[0]
        C = margins.shape[1]

        # gradient of scores
        dscores = np.zeros_like(margins)
        dscores[margins > 0] = 1
        dscores[np.arange(N), y] -= np.sum(margins > 0, axis=1)

        dscores /= N

        # gradients for W2, b2
        dW2 = self.a1.T.dot(dscores)
        db2 = np.sum(dscores, axis=0)

        # backprop into hidden layer
        da1 = dscores.dot(self.W2.T)
        dz1 = da1 * self.relu_grad(self.z1)

        # gradients for W1, b1
        dW1 = X.T.dot(dz1)
        db1 = np.sum(dz1, axis=0)

        # SGD update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def fit(self, X, y, epochs=10, batch_size=128):
        N = X.shape[0]

        for epoch in range(epochs):
            # shuffle
            idx = np.random.permutation(N)
            X = X[idx]
            y = y[idx]

            for i in range(0, N, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                scores = self.forward(X_batch)
                loss, margins = self.hinge_loss(scores, y_batch)
                self.backward(X_batch, y_batch, margins)

            print(f"Epoch {epoch+1}/{epochs}, loss={loss:.4f}")

    def predict(self, X):
        scores = self.forward(X)
        return np.argmax(scores, axis=1)
