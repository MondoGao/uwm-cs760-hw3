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
    print(data.shape)
    cross_ranges = [
        [0, 1000],
        [1000, 2000],
        [2000, 3000],
        [3000, 4000],
        [4000, 5000],
    ]

    for idx, train_range in enumerate(cross_ranges):
        accuracy, precision, recall = single_fold(tuple(train_range), data)

        print(
            f"Fold {idx + 1}: accuracy: {accuracy}, precision: {precision}, recall: {recall}"
        )


def single_fold(test_range: tuple[int, int], data):
    start, end = test_range
    test_data = data[start:end]
    train_data = np.concatenate((data[:start], data[end:]))
    # print(train_data.shape)
    # print(test_data.shape)

    X_train = train_data[:, 0:-1]
    Y_train = train_data[:, -1].flat
    nn = KNeighborsClassifier(n_neighbors=1)
    nn.fit(X_train, Y_train)

    X_test = test_data[:, 0:-1]
    Y_real = test_data[:, -1].flat
    Y_predict = nn.predict(X_test)

    # print(
    #     f"unique Y_real: {np.unique(Y_real)}, unique Y_predict: {np.unique(Y_predict)}"
    # )
    # print(f"Y_real: {list(test_data)[2178]}")
    # print(list(Y_predict))
    # print(list(Y_real))

    accuracy = met.accuracy_score(Y_real, Y_predict)
    precision = met.precision_score(Y_real, Y_predict)
    recall = met.recall_score(Y_real, Y_predict)

    # tp = sum(np.logical_and(Y_predict, Y_real))
    # fp = sum(np.logical_and(Y_predict, np.logical_not(Y_real)))
    # tn = sum(np.logical_and(np.logical_not(Y_predict), np.logical_not(Y_real)))
    # fn = sum(np.logical_and(Y_real, np.logical_not(Y_predict)))
    # accuracy = (tp + tn) / (tp + tn + fp + fn)
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)

    return accuracy, precision, recall


if __name__ == "__main__":
    main()
