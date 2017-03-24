import numpy as np
from keras.utils.np_utils import to_categorical
from voc2011.score import (
    accuracy,
    mean_accuracy,
    mean_IU,
    freq_weighted_IU
)


def test_voc_scores():
    y_true = to_categorical(np.array([[0, 2, 0], [0, 2, 0], [0, 2, 0]]),
                            3).reshape(3, 3, 3)
    y_pred = to_categorical(np.array([[0, 2, 2], [0, 2, 2], [0, 2, 2]]),
                            3).reshape(3, 3, 3)
    acc = accuracy(y_true, y_pred)
    assert abs(acc - 2/float(3)) < 1e-8
    avgacc = mean_accuracy(y_true, y_pred)
    assert abs(avgacc - 0.75) < 1e-8
    avgiu = mean_IU(y_true,  y_pred)
    assert abs(avgiu - 0.5) < 1e-8
    fwavacc = freq_weighted_IU(y_true, y_pred)
    assert abs(fwavacc - 0.5) < 1e-8
    y_true = to_categorical(np.array([[1, 2, 0], [0, 2, 1], [2, 2, 0]]),
                            3).reshape(3, 3, 3)
    y_pred = to_categorical(np.array([[1, 2, 0], [0, 2, 1], [2, 2, 0]]),
                            3).reshape(3, 3, 3)
    acc = accuracy(y_true, y_pred)
    assert abs(acc - 1) < 1e-8
    avgacc = mean_accuracy(y_true, y_pred)
    assert abs(avgacc - 1) < 1e-8
    avgiu = mean_IU(y_true,  y_pred)
    assert abs(avgiu - 1) < 1e-8
    fwavacc = freq_weighted_IU(y_true, y_pred)
    assert abs(fwavacc - 1) < 1e-8
