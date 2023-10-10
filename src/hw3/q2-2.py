import math
import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics as met

from hw3.knn import KNN


def main():
    data = np.loadtxt(
        fname="data/emails.csv",
        delimiter=",",
        skiprows=1,
        usecols=range(1, 3002),
        dtype=int,
    )
    # print(data.shape)
    cross_ranges = [
        [0, 1000],
        [1000, 2000],
        [2000, 3000],
        [3000, 4000],
        [4000, 5000],
    ]
    # debug
    # data = data[:100]
    cross_ranges = [
        [0, 1000],
    ]

    for idx, train_range in enumerate(cross_ranges):
        accuracy, precision, recall = single_fold(tuple(train_range), data)

        print(
            f"Fold {idx + 1}: accuracy: {accuracy}, precision: {precision}, recall: {recall}"
        )


def single_fold(test_range: tuple[int, int], data):
    start, end = test_range
    train_data = np.concatenate((data[:start], data[end:]))
    test_data = data[start:end]

    X_train = train_data[:, 0:-1]
    Y_train = train_data[:, -1].flat
    nn = KNN(k=1)
    nn.fit(X_train, Y_train)

    X_test = test_data[:, 0:-1]
    Y_real = test_data[:, -1].flat
    Y_predict = nn.predict(X_test)

    accuracy = met.accuracy_score(Y_real, Y_predict)
    precision = met.precision_score(Y_real, Y_predict)
    recall = met.recall_score(Y_real, Y_predict)

    return accuracy, precision, recall


if __name__ == "__main__":
    main()
