import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as met


def main():
    data = np.loadtxt(
        fname="data/emails.csv",
        delimiter=",",
        skiprows=1,
        usecols=range(1, 3002),
        dtype=int,
    )
    train_data = data[:2500]
    test_data = data[2500:]
    knn(train_data, test_data)
    lg(train_data, test_data)
    plt.show()


def knn(train_data, test_data):
    X_train = train_data[:, 0:-1]
    Y_train = train_data[:, -1].flat
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, Y_train)

    X_test = test_data[:, 0:-1]
    Y_real = test_data[:, -1].flat

    Y_prob = knn.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = met.roc_curve(Y_real, Y_prob)
    auc = met.auc(fpr, tpr)

    plt.plot(fpr, tpr, 'b')
    plt.xlabel("False Positive Rate (Positive label: 1)")
    plt.ylabel("True Positive Rate (Positive label: 1)")

def lg(train_data, test_data):
    X_train = train_data[:, 0:-1]
    Y_train = train_data[:, -1].flat
    lg = LogisticRegression(max_iter=1000)
    lg.fit(X_train, Y_train)

    X_test = test_data[:, 0:-1]
    Y_real = test_data[:, -1].flat

    Y_prob = lg.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = met.roc_curve(Y_real, Y_prob)
    auc = met.auc(fpr, tpr)

    plt.plot(fpr, tpr, 'y')
    plt.xlabel("False Positive Rate (Positive label: 1)")
    plt.ylabel("True Positive Rate (Positive label: 1)")


if __name__ == "__main__":
    main()
