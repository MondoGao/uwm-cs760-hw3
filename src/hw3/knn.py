import numpy as np
from dataclasses import dataclass


@dataclass
class KNN:
    k: int = 3

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, x):
        class_votes = self.predict_vote(x)
        # we choose to use a simple majority vote
        most_common_class = max(class_votes, key=class_votes.get)
        return most_common_class

    # use the same name with skilearn
    def predict_proba():
        class_votes = self.predict_vote(x)

        class_probabilities = {}
        total_neighbors = len(self.k)
        for label, count in class_votes.items():
            class_probabilities[label] = count / total_neighbors

        return class_probabilities

    def predict_vote(self, x):
        # [[dist, label], ...]
        distances = []
        data_len = len(self.X_train)
        for i in range(data_len):
            dist = self._euclidean_distance(x, self.X_train[i])
            distances.append((dist, self.Y_train[i]))

        distances.sort(key=lambda x: x[0])
        # distances may have ties, simply ignore
        k_nearest_neighbors = distances[: self.k]

        # label -> count
        class_votes = {}
        for neighbor in k_nearest_neighbors:
            label = neighbor[1]
            if label in class_votes:
                class_votes[label] += 1
            else:
                class_votes[label] = 1
        return class_votes

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
