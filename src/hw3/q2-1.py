import math
import numpy as np
from matplotlib import pyplot as plt

from hw3.knn import KNN


def main():
    data = np.loadtxt(fname="data/D2z.txt", delimiter=" ")
    X_train = data[:, 0:2]
    Y_train = data[:, 2].flat
    nn = KNN(k=1)
    nn.fit(X_train, Y_train)

    plot_step = 0.1
    X1_plot = np.arange(-2, 2 + plot_step, plot_step)
    X2_plot = np.arange(-2, 2 + plot_step, plot_step)
    X1_grid, X2_grid = np.meshgrid(X1_plot, X2_plot)
    X_plot = np.stack((X1_grid, X2_grid), axis=-1).reshape(-1, 2)
    # print(
    #     "grid",
    #     X1_grid,
    #     X2_grid,
    # )
    # print("plot", X_plot)
    Y_plot = nn.predict(X_plot)
    # print(Y_plot)
    plt.margins(0)
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=Y_plot, marker=".", s=20, cmap="RdYlBu")

    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, marker="x", cmap="RdYlBu")
    plt.show()


if __name__ == "__main__":
    main()
