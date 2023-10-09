from matplotlib import pyplot as plt
import numpy as np
import math


def main():
    data = np.array(
        [
            [0.95, 1],
            [0.85, 1],
            [0.8, 0],
            [0.7, 1],
            [0.55, 1],
            [0.45, 0],
            [0.4, 1],
            [0.3, 1],
            [0.2, 0],
            [0.1, 0],
        ]
    )
    observations = data[:, 0]
    real_class = data[:, 1]

    recalls = []
    fprs = []
    ob_recalls = []
    ob_fprs = []
    ob_txt = []

    thres_step = 0.01
    T = np.arange(0, 1, thres_step)
    for idx, t in enumerate(T):
        # ! float comparison is not accurate
        threshold = round(t, 2)
        prediction = [1 if c >= threshold else 0 for c in observations]
        # if prediction and real_class are both 1, then tp += 1
        tp = sum(np.logical_and(prediction, real_class))
        # if prediction and real_class are both 0, then fp += 1
        fp = sum(
            np.logical_and(
                prediction,
                np.logical_not(real_class),
            )
        )
        tn = sum(np.logical_and(np.logical_not(real_class), np.logical_not(prediction)))
        fn = sum(np.logical_and(real_class, np.logical_not(prediction)))
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)
        recalls.append(recall)
        fprs.append(fpr)
        if len(list(filter(lambda o: math.isclose(o, threshold), observations))):
            ob_recalls.append(recall)
            ob_fprs.append(fpr)
            ob_txt.append(f"threshold: {threshold:.2f}")
            print(
                f"threshold: {threshold}, tp:{tp}, tn: {tn}, fp: {fp}, fn: {fn}, fpr: {fpr:.2f}, recall: {recall:.2f}"
            )

    plt.plot(ob_fprs, ob_recalls, "o")
    for i, txt in enumerate(ob_txt):
        plt.annotate(txt, (ob_fprs[i], ob_recalls[i]))

    plt.plot(fprs, recalls)
    plt.xlabel("FP-rate")
    plt.ylabel("Recall")
    plt.show()


if __name__ == "__main__":
    main()
