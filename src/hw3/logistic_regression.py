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


def cost_function(Y_real, Y_pred):
    data_len = len(Y_real)
    return -1 / data_len * np.sum(Y_real * np.log(Y_pred) + (1 - Y_real) * np.log(1 - Y_pred)) 



def gradient_descent(X, Y, theta, learning_rate, num_iterations):
    m = len(Y)
    for iteration in range(num_iterations):
        Y_pred = logistic_regression(theta, X)

        gradient = np.dot(X.T, (Y_pred - Y)) / m

        theta -= learning_rate * gradient

        cost = cost_function(Y, Y_pred)

    return theta


