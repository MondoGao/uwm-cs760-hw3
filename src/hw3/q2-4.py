import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as met


def main():
    data = np.loadtxt(
        fname="data/emails.csv",
        delimiter=",",
        skiprows=1,
        usecols=range(1, 3002),
        dtype=int,
    )
    ks = [1, 3, 5, 7, 10]
    # ks = [1, 3]
    accuracies = [run_knn(data, k) for k in ks]
    # print(accuracies)

    plt.plot(ks, accuracies, marker="o")
    for idx, k in enumerate(ks):
        plt.annotate(f"{accuracies[idx]:0.4f}", (k, accuracies[idx]))
        print(f"k={k}, accuracy={accuracies[idx]}")
    plt.show()


def run_knn(data, k: int):
    cross_ranges = [
        [0, 1000],
        [1000, 2000],
        [2000, 3000],
        [3000, 4000],
        [4000, 5000],
    ]

    return np.mean([single_fold(tuple(r), data, k) for r in cross_ranges])


def single_fold(train_range: tuple, data, k: int) -> float:
    start, end = train_range
    train_data = data[start:end]
    test_data = np.concatenate((data[:start], data[end:]))

    X_train = train_data[:, 0:-1]
    Y_train = train_data[:, -1].flat
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)

    X_test = test_data[:, 0:-1]
    Y_real = test_data[:, -1].flat
    Y_predict = knn.predict(X_test)

    accuracy = met.accuracy_score(Y_real, Y_predict)

    return accuracy


if __name__ == "__main__":
    main()
