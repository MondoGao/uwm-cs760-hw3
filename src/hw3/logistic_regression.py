import numpy as np
from dataclasses import dataclass


@dataclass
class LogisticRegression:
    learning_rate: float = 0.1
    max_iterations: int = 1000
    theta = None

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        self.gradient_descent(X, y)

    def predict_proba(self, X):
        z = np.dot(X, self.theta)
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int)
        return predictions

    def gradient_descent(self, X, y):
        data_len, feat_num = X.shape
        self.theta = np.zeros(feat_num)

        for _ in range(self.max_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / data_len
            self.theta -= self.learning_rate * gradient

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, h):
        epsilon = 1e-15
        return -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

