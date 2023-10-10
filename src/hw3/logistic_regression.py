import numpy as np
from dataclasses import dataclass


@dataclass
class LogisticRegression:
    learning_rate: float = 0.1
    num_iterations: int = 1000
    theta = None

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        self.theta = gradient_descent(
            X, y, self.theta, self.learning_rate, self.num_iterations
        )

    def predict(self, X_test):
        return logistic_regression(self.theta, X_test)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression(theta, X):
    return sigmoid(np.dot(X, theta))


def cost_function(y, y_pred):
    m = len(y)
    return -1 / m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    for iteration in range(num_iterations):
        y_pred = logistic_regression(theta, X)

        gradient = np.dot(X.T, (y_pred - y)) / m

        theta -= learning_rate * gradient

        cost = cost_function(y, y_pred)

    return theta
