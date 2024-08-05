# Plot score for signal and background, comparing training and testing
from math import log, sqrt

import matplotlib.pyplot as plt
import numpy as np


def compare_train_test(
    y_pred_train,
    y_train,
    y_pred,
    y_test,
    high_low=(0, 1),
    bins=30,
    xlabel="",
    ylabel="Arbitrary units",
    title="",
    weights_train=None,
    weights_test=None,
    density=True,
):
    if weights_train.size is not None:
        weights_train_signal = weights_train[y_train == 1]
        weights_train_background = weights_train[y_train == 0]
    else:
        weights_train_signal = None
        weights_train_background = None
    plt.hist(
        y_pred_train[y_train == 1],
        color='r',
        alpha=0.5,
        range=high_low,
        bins=bins,
        histtype='stepfilled',
        density=density,
        label='S (train)',
        weights=weights_train_signal,
    )  # alpha is transparancy
    plt.hist(
        y_pred_train[y_train == 0],
        color='b',
        alpha=0.5,
        range=high_low,
        bins=bins,
        histtype='stepfilled',
        density=density,
        label='B (train)',
        weights=weights_train_background,
    )

    if weights_test is not None:
        weights_test_signal = weights_test[y_test == 1]
        weights_test_background = weights_test[y_test == 0]
    else:
        weights_test_signal = None
        weights_test_background = None
    hist, bins = np.histogram(y_pred[y_test == 1], bins=bins, range=high_low, density=density, weights=weights_test_signal)
    scale = len(y_pred[y_test == 1]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

    hist, bins = np.histogram(
        y_pred[y_test == 0], bins=bins, range=high_low, density=density, weights=weights_test_background
    )
    scale = len(y_pred[y_test == 0]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')


def amsasimov(s, b):  # asimov (or Poisson) significance
    if b <= 0 or s <= 0:
        return 0
    try:
        return sqrt(2 * ((s + b) * log(1 + float(s) / b) - s))
    except ValueError:
        print(1 + float(s) / b)
        print(2 * ((s + b) * log(1 + float(s) / b) - s))
    # return s/sqrt(s+b)
