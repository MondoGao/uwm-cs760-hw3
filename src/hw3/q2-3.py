import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

from hw3.logistic_regression import LogisticRegression


def main():
    data = np.loadtxt(
        fname="data/emails.csv",
        delimiter=",",
        skiprows=1,
        usecols=range(1, 3002),
        dtype=int,
    )

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


def single_fold(
    test_range: tuple[int, int], data, learning_rate=0.1, max_iterations=1000
):
    start, end = test_range
    test_data = data[start:end]
    train_data = np.concatenate((data[:start], data[end:]))

    X_train = train_data[:, 0:-1]
    Y_train = train_data[:, -1].flatten()
    X_test = test_data[:, 0:-1]
    Y_test = test_data[:, -1].flatten()

    lg = LogisticRegression(learning_rate=learning_rate, max_iterations=max_iterations)
    lg.fit(X_train, Y_train)

    Y_pred = lg.predict(X_test)

    accuracy = accuracy_score(Y_test, (Y_pred > 0.5).astype(int))
    precision = precision_score(Y_test, (Y_pred > 0.5).astype(int))
    recall = recall_score(Y_test, (Y_pred > 0.5).astype(int))

    return accuracy, precision, recall


if __name__ == "__main__":
    main()
